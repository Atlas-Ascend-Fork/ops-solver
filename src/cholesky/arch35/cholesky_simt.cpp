/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cholesky_internal.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/math_functions.h"

namespace Cholesky
{
__aicore__ __attribute__((always_inline)) inline float *Pointer(float *single, float **array, int first, int q)
{
    return array ? array[first + q] : single;
}
__aicore__ __attribute__((always_inline)) inline int64_t Offset(int ld, int upper, int i, int j)
{
    return upper ? int64_t(i) * ld + j : int64_t(j) * ld + i;
}
__aicore__ __attribute__((always_inline)) inline int64_t Diagonal(int ld, int i)
{
    return int64_t(i) * (int64_t(ld) + 1);
}
__global__ __launch_bounds__(1024) void InfoKernel(int *info, int value, int count, float **aArray, float **bArray,
                                                   int batchCount)
{
    if (aArray)
    {
        // The compiler supports this shared attribute in pure SIMT mode.
        __attribute__((shared)) unsigned missing[1024];
        unsigned mask = 0;
        for (int q = threadIdx.x; q < batchCount; q += blockDim.x)
        {
            if (!aArray[q]) mask |= 1;
            if (!bArray[q]) mask |= 2;
        }
        missing[threadIdx.x] = mask;
        asc_syncthreads();
        if (threadIdx.x == 0)
        {
            for (unsigned t = 1; t < blockDim.x; ++t) mask |= missing[t];
            info[0] = (mask & 1) ? -4 : (mask & 2) ? -6 : 0;
        }
        return;
    }
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) info[i] = value;
}
__global__ __launch_bounds__(512) void SmallFactorKernel(float *aSingle, float **aArray, int n, int lda, int upper,
                                                         int firstA, int count, int *info)
{
    for (int q = blockIdx.x * blockDim.x + threadIdx.x; q < count; q += gridDim.x * blockDim.x)
    {
        float *p = Pointer(aSingle, aArray, firstA, q);
        if (!p)
        {
            info[q] = -3;
            continue;
        }
        int failure = 0;
        for (int k = 0; k < n; ++k)
        {
            float d = p[Diagonal(lda, k)];
            for (int t = 0; t < k; ++t)
            {
                float f = p[Offset(lda, upper, k, t)];
                d = fmaf(-f, f, d);
            }
            if (!(d > 0))
            {
                failure = k + 1;
                break;
            }
            // SIMT sqrtf may differ by one ulp. Refine with the FMA residual
            // before storing the FP32 pivot used by both triangular solves.
            float root = sqrtf(d);
            float correction = fmaf(-root, root, d);
            d = fmaf(correction, 0.5f / root, root);
            p[Diagonal(lda, k)] = d;
            for (int i = k + 1; i < n; ++i)
            {
                float x = p[Offset(lda, upper, i, k)];
                for (int t = 0; t < k; ++t) x = fmaf(-p[Offset(lda, upper, i, t)], p[Offset(lda, upper, k, t)], x);
                p[Offset(lda, upper, i, k)] = x / d;
            }
        }
        info[q] = failure;
    }
}
__global__ __launch_bounds__(128) void SolveKernel(float *aSingle, float **aArray, float *bSingle, float **bArray,
                                                   int n, int lda, int ldb, int upper, int firstA, int firstB,
                                                   int count, int nrhs, int *info)
{
    if (info[0]) return;
    for (int qr = blockIdx.x * blockDim.x + threadIdx.x; qr < count * nrhs; qr += gridDim.x * blockDim.x)
    {
        int q = qr / nrhs, r = qr % nrhs;
        float *pa = Pointer(aSingle, aArray, firstA, q), *pb = Pointer(bSingle, bArray, firstB, q);
        if (!pa || !pb) continue;
        pb += int64_t(r) * ldb;
        for (int i = 0; i < n; ++i)
        {
            float x = pb[i];
            for (int j = 0; j < i; ++j) x = fmaf(-pa[Offset(lda, upper, i, j)], pb[j], x);
            pb[i] = x / pa[Diagonal(lda, i)];
        }
        for (int i = n - 1; i >= 0; --i)
        {
            float x = pb[i];
            for (int j = i + 1; j < n; ++j) x = fmaf(-pa[Offset(lda, upper, j, i)], pb[j], x);
            pb[i] = x / pa[Diagonal(lda, i)];
        }
    }
}
__global__ __launch_bounds__(128) void InverseStoreKernel(float *aSingle, float **aArray, int n, int lda, int upper,
                                                          int firstA, int count, const float *w, int pitch,
                                                          const int *info)
{
    if (info[0]) return;
    for (int ij = blockIdx.x * blockDim.x + threadIdx.x; ij < n * n; ij += gridDim.x * blockDim.x)
    {
        int i = ij / n, j = ij % n;
        if (j <= i) aSingle[Offset(lda, upper, i, j)] = w[int64_t(j) * pitch + i];
    }
}

__global__ __launch_bounds__(128) void IdentityKernel(const float *g, float *b, int factorLd, int rhsPitch, int n,
                                                      int *info)
{
    // Only one thread examines the factor; the next kernel observes info in stream order.
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        info[0] = 0;
        for (int i = 0; i < n; ++i)
        {
            float diagonal = g[Diagonal(factorLd, i)];
            if (!(diagonal > 0.f) || !isfinite(diagonal))
            {
                info[0] = i + 1;
                break;
            }
        }
    }
    // Each block writes contiguous rows of a column of the identity right-hand side.
    for (int col = blockIdx.x; col < n; col += gridDim.x)
    {
        for (int row = threadIdx.x; row < n; row += blockDim.x)
            b[int64_t(col) * rhsPitch + row] = row == col ? 1.f : 0.f;
    }
}

__global__ __launch_bounds__(1024) void SubtractKernel(float *b, const float *product, int ldb, int productPitch, int n,
                                                       int nrhs, int start, int width, int reverse, const int *info)
{
    if (info[0]) return;
    int rows = reverse ? start : n - start - width;
    for (int r = blockIdx.x; r < nrhs; r += gridDim.x)
    {
        for (int i = threadIdx.x; i < rows; i += blockDim.x)
            b[int64_t(r) * ldb + i + (reverse ? 0 : start + width)] -= product[int64_t(r) * productPitch + i];
    }
}

void SetInfo(int *info, int value, int count, aclrtStream s)
{
    InfoKernel<<<1, 128, 0, s>>>(info, value, count, nullptr, nullptr, 0);
}
void SetSolveInfo(int *info, float *a[], float *b[], int count, aclrtStream s)
{
    unsigned threads = 128;
    while (threads < unsigned(count) && threads < 1024) threads *= 2;
    InfoKernel<<<1, threads, 0, s>>>(info, 0, 1, a, b, count);
}
void SmallFactor(const MatrixBatch &a, int *info, aclrtStream s)
{
    if (a.n > 8)
    {
        SmallFactorVector(a, info, s);
        return;
    }
    SmallFactorKernel<<<VectorCores(), 512, 0, s>>>(a.single, a.array, a.n, a.ld, a.upper, a.first, a.count, info);
}
void Solve(const MatrixBatch &a, const MatrixBatch &b, int nrhs, int *info, aclrtStream s, float *product,
           int productPitch)
{
    if (a.n <= 128 && ((a.n > 32 && nrhs == 1) || (a.n > 8 && unsigned(a.count) <= VectorCores())))
    {
        SmallSolve(a, b, nrhs, info, s);
        return;
    }
    if (a.n <= 128)
    {
        SolveKernel<<<VectorCores(), 128, 0, s>>>(a.single, a.array, b.single, b.array, a.n, a.ld, b.ld, a.upper,
                                                  a.first, b.first, a.count, nrhs, info);
        return;
    }
    for (int reverse = 0; reverse < 2; ++reverse)
    {
        int panels = (a.n + PANEL - 1) / PANEL;
        for (int step = 0; step < panels; ++step)
        {
            int start = (reverse ? panels - 1 - step : step) * PANEL;
            int width = a.n - start < PANEL ? a.n - start : PANEL;
            SolvePanel(a, b, nrhs, start, width, reverse, info, s);
            int rows = reverse ? start : a.n - start - width;
            if (rows && nrhs == 1)
                SolveUpdateVector(a, b, start, width, reverse, info, s);
            else if (rows)
            {
                RhsUpdate(a, b, nrhs, product, productPitch, start, width, reverse, info, CubeCores(), s);
                SubtractKernel<<<VectorCores(), 1024, 0, s>>>(b.single, product, b.ld, productPitch, a.n, nrhs, start,
                                                              width, reverse, info);
            }
        }
    }
}
void Invert(const MatrixBatch &a, float *w, int pitch, int *info, aclrtStream s)
{
    if (a.n <= 128)
    {
        IdentityKernel<<<VectorCores(), 128, 0, s>>>(a.single, w, a.ld, pitch, a.n, info);
        MatrixBatch rhs{w, nullptr, a.n, pitch, 0, 0, 1};
        SmallSolve(a, rhs, a.n, info, s);
        InverseStoreKernel<<<VectorCores(), 128, 0, s>>>(a.single, a.array, a.n, a.ld, a.upper, a.first, a.count, w,
                                                         pitch, info);
        return;
    }
    float *b = w + int64_t(pitch) * pitch;
    float *product = b + int64_t(pitch) * pitch;
    Pack(a, w, pitch, info, s);
    IdentityKernel<<<VectorCores(), 128, 0, s>>>(w, b, pitch, pitch, a.n, info);
    MatrixBatch factor{w, nullptr, a.n, pitch, 1, 0, 1};
    MatrixBatch rhs{b, nullptr, a.n, pitch, 0, 0, 1};
    for (int reverse = 0; reverse < 2; ++reverse)
    {
        int panels = (a.n + PANEL - 1) / PANEL;
        for (int step = 0; step < panels; ++step)
        {
            int start = (reverse ? panels - 1 - step : step) * PANEL;
            int width = a.n - start < PANEL ? a.n - start : PANEL;
            SolvePanel(factor, rhs, a.n, start, width, reverse, info, s);
            int rows = reverse ? start : a.n - start - width;
            if (!rows) continue;
            TriUpdate(w, b, product, pitch, a.n, start, width, reverse, info, CubeCores(), s);
            SubtractKernel<<<VectorCores(), 1024, 0, s>>>(b, product, pitch, pitch, a.n, a.n, start, width, reverse,
                                                          info);
        }
    }
    InverseStoreKernel<<<VectorCores(), 128, 0, s>>>(a.single, a.array, a.n, a.ld, a.upper, a.first, a.count, b, pitch,
                                                     info);
}
}  // namespace Cholesky
