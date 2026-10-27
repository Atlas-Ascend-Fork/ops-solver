/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cholesky_device.h"

namespace Cholesky
{
namespace
{
template <int BLOCK>
__simt_callee__ inline void SelectInversePivot(int row, const float (&re)[BLOCK / WARP_SIZE],
                                               const float (&im)[BLOCK / WARP_SIZE],
                                               const float (&inverseDiagonal)[BLOCK / WARP_SIZE], float& xr, float& xi,
                                               float& inverseD)
{
    xr = row ? re[1] : re[0];
    xi = row ? im[1] : im[0];
    inverseD = row ? inverseDiagonal[1] : inverseDiagonal[0];
    if constexpr (BLOCK == MAX_SOLVE_BLOCK)
    {
        xr = row < 2 ? xr : (row == 2 ? re[2] : re[3]);
        xi = row < 2 ? xi : (row == 2 ? im[2] : im[3]);
        inverseD = row < 2 ? inverseD : (row == 2 ? inverseDiagonal[2] : inverseDiagonal[3]);
    }
}

template <int BLOCK>
__simt_callee__ inline void InvertTileStep(__ubuf__ aclsolverFloatComplex* d, int k, int width,
                                           float (&re)[BLOCK / WARP_SIZE], float (&im)[BLOCK / WARP_SIZE],
                                           const float (&inverseDiagonal)[BLOCK / WARP_SIZE])
{
    int lane = threadIdx.x % WARP_SIZE;
    constexpr int diagonalStride = BLOCK + 1;

    int row = k / WARP_SIZE;
    float xr = re[0], xi = im[0], inverseD = inverseDiagonal[0];
    if constexpr (BLOCK >= SINGLE_RHS_BLOCK)
    {
        SelectInversePivot<BLOCK>(row, re, im, inverseDiagonal, xr, xi, inverseD);
    }

    inverseD = asc_shfl(inverseD, k % WARP_SIZE);
    float pr = asc_shfl(xr, k % WARP_SIZE) * inverseD, pi = asc_shfl(xi, k % WARP_SIZE) * inverseD;
#pragma unroll
    for (int j = 0; j < BLOCK / WARP_SIZE; ++j)
    {
        if (lane == k % WARP_SIZE && row == j)
        {
            re[j] = pr;
            im[j] = pi;
        }
    }
#pragma unroll
    for (int j = 0; j < BLOCK / WARP_SIZE; ++j)
    {
        int r = lane + j * WARP_SIZE;
        if (r < width && r > k)
        {
            auto v = SubProduct({re[j], im[j]}, ReadUB(d, r * diagonalStride + k), {pr, pi});
            re[j] = v.real;
            im[j] = v.imag;
        }
    }
}

template <int BLOCK>
__simt_callee__ inline void InvertTileColumn(__ubuf__ aclsolverFloatComplex* d, int rhs, int width,
                                             const float (&inverseDiagonal)[BLOCK / WARP_SIZE],
                                             __gm__ aclsolverFloatComplex* out, __gm__ aclsolverFloatComplex* adjoint)
{
    int lane = threadIdx.x % WARP_SIZE;
    constexpr int block = BLOCK;

    float re[BLOCK / WARP_SIZE], im[BLOCK / WARP_SIZE];
#pragma unroll
    for (int j = 0; j < BLOCK / WARP_SIZE; ++j)
    {
        re[j] = rhs < width && lane + j * WARP_SIZE == rhs ? 1 : 0;
        im[j] = 0;
    }
    for (int k = rhs; k < width; ++k)
    {
        InvertTileStep<BLOCK>(d, k, width, re, im, inverseDiagonal);
    }
#pragma unroll
    for (int j = 0; j < BLOCK / WARP_SIZE; ++j)
    {
        int r = lane + j * WARP_SIZE;
        if (r < block)
        {
            WriteLower(out, block, r, rhs, false, {re[j], im[j]});
            WriteLower(adjoint, block, rhs, r, false, {re[j], -im[j]});
        }
    }
}

template <int BLOCK>
__simt_callee__ inline void LoadInverseDiagonal(const __gm__ aclsolverFloatComplex* a, int width, int lda, int offset,
                                                bool upper, __ubuf__ aclsolverFloatComplex* d,
                                                float (&inverseDiagonal)[BLOCK / WARP_SIZE],
                                                __gm__ aclsolverFloatComplex* inverse,
                                                __gm__ aclsolverFloatComplex*& out,
                                                __gm__ aclsolverFloatComplex*& adjoint, int64_t tile, int64_t tiles)
{
    constexpr int block = BLOCK, diagonalStride = BLOCK + 1;
    int lane = threadIdx.x % WARP_SIZE;
    for (int p = threadIdx.x; p < block * block; p += INVERSE_THREADS)
    {
        int r = upper ? p / block : p % block, c = upper ? p % block : p / block;
        if (r < width && c <= r)
        {
            WriteUB(d, r * diagonalStride + c, ReadLower(a, lda, offset + r, offset + c, upper));
        }
    }
    asc_syncthreads();
    out = inverse + tile * block * block;
    adjoint = out + tiles * block * block;
#pragma unroll
    for (int j = 0; j < BLOCK / WARP_SIZE; ++j)
    {
        int r = lane + j * WARP_SIZE;
        inverseDiagonal[j] = r < width ? 1.0f / d[r * diagonalStride + r].real : 0;
    }
}

template <int BLOCK>
__simt_vf__ __aicore__ __launch_bounds__(INVERSE_THREADS) void InvertTilesVF(
    const __gm__ aclsolverFloatComplex* single, __gm__ const uint64_t* array, int n, int nrhs, int lda, bool upper,
    int batch, int groups, __gm__ float* work, const __gm__ int* check, __ubuf__ aclsolverFloatComplex* d)
{
    if (check && *check != 0)
    {
        return;
    }
    constexpr int block = BLOCK;
    int panels = (n + block - 1) / block;
    int rhsPad = (nrhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int64_t nativeCount = static_cast<int64_t>(NATIVE_RHS_ROWS + COMPLEX_COMPONENTS * capacity) * rhsPad * batch;
    auto inverse = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work) + nativeCount;
    int64_t tiles = static_cast<int64_t>(batch) * panels;
    int group = blockIdx.x % groups, warp = threadIdx.x / WARP_SIZE;
    for (int64_t tile = blockIdx.x / groups; tile < tiles; tile += gridDim.x / groups)
    {
        int m = tile / panels, offset = (tile % panels) * block;
        int width = n - offset < block ? n - offset : block;
        auto a = array ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(array[m]) : single;
        if (offset == 0 && group == 0 && nrhs <= SMALL_RHS_LIMIT)
        {
            for (int p = threadIdx.x; p < NATIVE_RHS_ROWS * rhsPad; p += INVERSE_THREADS)
            {
                work[static_cast<int64_t>(m) * NATIVE_RHS_ROWS * rhsPad + p] = 0;
                work[static_cast<int64_t>(batch + m) * NATIVE_RHS_ROWS * rhsPad + p] = 0;
            }
        }
        float inverseDiagonal[BLOCK / WARP_SIZE];
        __gm__ aclsolverFloatComplex* out = nullptr;
        __gm__ aclsolverFloatComplex* adjoint = nullptr;
        LoadInverseDiagonal<BLOCK>(a, width, lda, offset, upper, d, inverseDiagonal, inverse, out, adjoint, tile,
                                   tiles);
        for (int rhs = group * (INVERSE_THREADS / WARP_SIZE) + warp; rhs < block;
             rhs += groups * (INVERSE_THREADS / WARP_SIZE))
        {
            InvertTileColumn<BLOCK>(d, rhs, width, inverseDiagonal, out, adjoint);
        }
        asc_syncthreads();
    }
}
__global__ __aicore__ __vector__ void InvertTilesKernel(const __gm__ aclsolverFloatComplex* a,
                                                        __gm__ const uint64_t* array, int n, int nrhs, int lda,
                                                        bool upper, int batch, int groups, __gm__ float* work,
                                                        const __gm__ int* check)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    int block = SolveBlockSize(nrhs, batch);
    pipe.InitBuffer(buffer, block * (block + 1) * sizeof(aclsolverFloatComplex));
    auto d = buffer.Get<uint8_t>();
    if (block == SMALL_SOLVE_BLOCK)
    {
        asc_vf_call<InvertTilesVF<SMALL_SOLVE_BLOCK>>(
            dim3(INVERSE_THREADS), a, array, n, nrhs, lda, upper, batch, groups, work, check,
            reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(d.GetPhyAddr()));
    }
    else if (block == SINGLE_RHS_BLOCK)
    {
        asc_vf_call<InvertTilesVF<SINGLE_RHS_BLOCK>>(dim3(INVERSE_THREADS), a, array, n, nrhs, lda, upper, batch,
                                                     groups, work, check,
                                                     reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(d.GetPhyAddr()));
    }
    else
    {
        asc_vf_call<InvertTilesVF<MAX_SOLVE_BLOCK>>(dim3(INVERSE_THREADS), a, array, n, nrhs, lda, upper, batch, groups,
                                                    work, check,
                                                    reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(d.GetPhyAddr()));
    }
}
__simt_callee__ inline aclsolverFloatComplex ProductValue(__gm__ float* real, __gm__ float* imag, int64_t index,
                                                          bool column)
{
    if (!column)
    {
        return {real[index], imag[index]};
    }
    auto x = asc_ldcg(reinterpret_cast<__gm__ float2*>(real + index));
    auto y = asc_ldcg(reinterpret_cast<__gm__ float2*>(imag + index));
    return {x.x - y.y, x.y + y.x};
}
__simt_callee__ inline void PackRhs(__gm__ float* real, __gm__ float* imag, int rhsPad, int row, int rhs, bool column,
                                    aclsolverFloatComplex value)
{
    int64_t index = static_cast<int64_t>(column ? row : 2 * row) * rhsPad + rhs;
    real[index] = value.real;
    imag[index] = value.imag;
    if (!column)
    {
        real[index + rhsPad] = value.imag;
        imag[index + rhsPad] = -value.real;
    }
}
// Stages: pack RHS, apply trailing product, copy diagonal product, or small-RHS diagonal multiply.
template <int inverseStride, bool PREFETCH>
__simt_callee__ inline aclsolverFloatComplex MultiplyInverseRow(__ubuf__ aclsolverFloatComplex* localInverse,
                                                                __ubuf__ aclsolverFloatComplex* shared, int width)
{
    float re[RHS_ACCUMULATORS] = {0}, im[RHS_ACCUMULATORS] = {0};
    for (int first = 0; first < width; first += RHS_ACCUMULATORS)
    {
#pragma unroll
        for (int j = 0; j < RHS_ACCUMULATORS; ++j)
        {
            int k = first + j;
            if (k < width)
            {
                auto z = ReadUB(localInverse,
                                PREFETCH ? k * inverseStride + threadIdx.x : threadIdx.x * inverseStride + k),
                     x = ReadUB(shared, k);
                re[j] = fmaf(z.real, x.real, re[j]);
                re[j] = fmaf(-z.imag, x.imag, re[j]);
                im[j] = fmaf(z.real, x.imag, im[j]);
                im[j] = fmaf(z.imag, x.real, im[j]);
            }
        }
    }
    aclsolverFloatComplex sum{(re[0] + re[1]) + (re[2] + re[3]), (im[0] + im[1]) + (im[2] + im[3])};
    return sum;
}

template <int BLOCK, bool PREFETCH>
__simt_callee__ inline void LoadInverseTile(__ubuf__ aclsolverFloatComplex* localInverse,
                                            const __gm__ aclsolverFloatComplex* inv, int width)
{
    constexpr int block = BLOCK, inverseStride = BLOCK + (PREFETCH ? NATIVE_PREFETCH_PADDING : 1);

    for (int p = threadIdx.x; p < width * width; p += THREADS)
    {
        int r = p % width, c = p / width;
        WriteUB(localInverse, r * inverseStride + c, ReadLower(inv, block, r, c, false));
    }
    asc_syncthreads();
}

template <int BLOCK, bool PREFETCH>
__simt_callee__ inline void SolveInitialDiagonal(__gm__ aclsolverFloatComplex* b, int ldb, int m, int panels,
                                                 int offset, bool backward, int batch,
                                                 __gm__ aclsolverFloatComplex* inverse,
                                                 __ubuf__ aclsolverFloatComplex* shared, int width, int group,
                                                 int groups, int activeRhs, __gm__ float* real, __gm__ float* imag,
                                                 int rhsPad, bool column)
{
    constexpr int block = BLOCK, inverseStride = BLOCK + (PREFETCH ? NATIVE_PREFETCH_PADDING : 1);

    int64_t tile = static_cast<int64_t>(m) * panels + offset / block;
    auto inv = inverse + (tile + (backward ? static_cast<int64_t>(batch) * panels : 0)) * block * block;
    auto localInverse = shared + NATIVE_SHARED_ELEMENTS;
    if constexpr (!PREFETCH)
    {
        LoadInverseTile<BLOCK, PREFETCH>(localInverse, inv, width);
    }
    for (int rhs = group; rhs < activeRhs; rhs += groups)
    {
        if (threadIdx.x < width)
        {
            WriteUB(shared, threadIdx.x, ReadLower(b, ldb, offset + threadIdx.x, rhs, false));
        }
        asc_syncthreads();
        if (threadIdx.x < width)
        {
            auto sum = MultiplyInverseRow<inverseStride, PREFETCH>(localInverse, shared, width);
            WriteLower(b, ldb, offset + threadIdx.x, rhs, false, sum);
            PackRhs(real, imag, rhsPad, threadIdx.x, rhs, column, sum);
        }
        asc_syncthreads();
    }
}

__simt_callee__ inline void PackNativeRhs(__gm__ aclsolverFloatComplex* b, int ldb, int offset, int width,
                                          int activeRhs, int group, int groups, NativeStage stage, int capacity,
                                          int rhsPad, bool column, __gm__ float* real, __gm__ float* imag,
                                          __gm__ float* outReal, __gm__ float* outImag,
                                          __ubuf__ aclsolverFloatComplex* shared)
{
    int activePad = (activeRhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int rhsTiles = (activePad + WARP_SIZE - 1) / WARP_SIZE;
    int rowTiles = (width + WARP_SIZE - 1) / WARP_SIZE;
    for (int tile = group; tile < rhsTiles * rowTiles; tile += groups)
    {
        int rowBase = (tile / rhsTiles) * WARP_SIZE, rhsBase = (tile % rhsTiles) * WARP_SIZE;
        for (int p = threadIdx.x; p < WARP_SIZE * WARP_SIZE; p += THREADS)
        {
            int r = rowBase + p % WARP_SIZE, rhs = rhsBase + p / WARP_SIZE;
            aclsolverFloatComplex v{0, 0};
            if (r < width && rhs < activeRhs)
            {
                if (stage == NativeStage::CopyDiagonal)
                {
                    v = ProductValue(outReal, outImag, static_cast<int64_t>(rhs) * 2 * capacity + 2 * r, true);
                    WriteLower(b, ldb, offset + r, rhs, false, v);
                }
                else
                {
                    v = ReadLower(b, ldb, offset + r, rhs, false);
                }
            }
            WriteUB(shared, (p % WARP_SIZE) * (WARP_SIZE + 1) + p / WARP_SIZE, v);
        }
        asc_syncthreads();
        for (int p = threadIdx.x; p < WARP_SIZE * WARP_SIZE; p += THREADS)
        {
            int r = rowBase + p / WARP_SIZE, rhs = rhsBase + p % WARP_SIZE;
            if (r < width && rhs < activePad)
            {
                PackRhs(real, imag, rhsPad, r, rhs, stage == NativeStage::PackRhs ? true : column,
                        ReadUB(shared, (p / WARP_SIZE) * (WARP_SIZE + 1) + p % WARP_SIZE));
            }
        }
        asc_syncthreads();
    }
}

template <int inverseStride, bool PREFETCH>
__simt_callee__ inline void ApplyNextDiagonal(__gm__ aclsolverFloatComplex* b, int ldb, int next, int nextWidth,
                                              int relative, int rhs, int capacity, bool column, int rhsPad,
                                              __gm__ float* real, __gm__ float* imag, __gm__ float* outReal,
                                              __gm__ float* outImag, __ubuf__ aclsolverFloatComplex* shared,
                                              __ubuf__ aclsolverFloatComplex* localInverse, int group, int owner)
{
    if (group == owner)
    {
        if (threadIdx.x < nextWidth)
        {
            int r = relative + threadIdx.x;
            auto product =
                ProductValue(outReal, outImag, static_cast<int64_t>(rhs) * 2 * capacity + (column ? 2 * r : r), column);
            auto v = ReadLower(b, ldb, next + threadIdx.x, rhs, false);
            WriteUB(shared, threadIdx.x, {v.real - product.real, v.imag - product.imag});
        }
        asc_syncthreads();
        if (threadIdx.x < nextWidth)
        {
            auto sum = MultiplyInverseRow<inverseStride, PREFETCH>(localInverse, shared, nextWidth);
            WriteLower(b, ldb, next + threadIdx.x, rhs, false, sum);
            PackRhs(real, imag, rhsPad, threadIdx.x, rhs, column, sum);
        }
        asc_syncthreads();
    }
}

template <int BLOCK, bool PREFETCH>
__simt_callee__ inline void ApplySmallRhs(__gm__ aclsolverFloatComplex* b, int n, int ldb, int m, int panels,
                                          int offset, int width, bool backward, int batch, int group, int groups,
                                          int activeRhs, __gm__ aclsolverFloatComplex* inverse,
                                          __ubuf__ aclsolverFloatComplex* shared, int capacity, bool column, int rhsPad,
                                          __gm__ float* real, __gm__ float* imag, __gm__ float* outReal,
                                          __gm__ float* outImag)
{
    constexpr int block = BLOCK, inverseStride = BLOCK + (PREFETCH ? NATIVE_PREFETCH_PADDING : 1);

    int rows = backward ? offset : n - offset - width;
    int first = backward ? 0 : offset + width;
    int next = backward ? offset - block : offset + width;
    int nextWidth = n - next < block ? n - next : block;
    int relative = next - first;
    int owner = (relative / THREADS) % groups;
    int64_t tile = static_cast<int64_t>(m) * panels + next / block;
    auto inv = inverse + (tile + (backward ? static_cast<int64_t>(batch) * panels : 0)) * block * block;
    auto localInverse = shared + NATIVE_SHARED_ELEMENTS;
    if (group == owner)
    {
        if constexpr (!PREFETCH)
        {
            LoadInverseTile<BLOCK, PREFETCH>(localInverse, inv, nextWidth);
        }
    }
    for (int rhs = 0; rhs < activeRhs; ++rhs)
    {
        for (int r = group * THREADS + threadIdx.x; r < rows; r += groups * THREADS)
        {
            if (r >= relative && r < relative + nextWidth)
            {
                continue;
            }
            auto product =
                ProductValue(outReal, outImag, static_cast<int64_t>(rhs) * 2 * capacity + (column ? 2 * r : r), column);
            auto v = ReadLower(b, ldb, first + r, rhs, false);
            WriteLower(b, ldb, first + r, rhs, false, {v.real - product.real, v.imag - product.imag});
        }
        ApplyNextDiagonal<inverseStride, PREFETCH>(b, ldb, next, nextWidth, relative, rhs, capacity, column, rhsPad,
                                                   real, imag, outReal, outImag, shared, localInverse, group, owner);
    }
}

__simt_callee__ inline void ApplyLargeRhs(__gm__ aclsolverFloatComplex* b, int n, int ldb, int offset, int width,
                                          bool backward, int group, int groups, int activeRhs, int capacity,
                                          bool column, __gm__ float* outReal, __gm__ float* outImag)
{
    int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;

    int rows = backward ? offset : n - offset - width;
    int first = backward ? 0 : offset + width;
    for (int rhs = group * RHS_WARPS + warp; rhs < activeRhs; rhs += groups * RHS_WARPS)
    {
        for (int r = lane; r < rows; r += WARP_SIZE)
        {
            auto product =
                ProductValue(outReal, outImag, static_cast<int64_t>(rhs) * 2 * capacity + (column ? 2 * r : r), column);
            auto v = ReadLower(b, ldb, first + r, rhs, false);
            WriteLower(b, ldb, first + r, rhs, false, {v.real - product.real, v.imag - product.imag});
        }
    }
}

template <int BLOCK, bool PREFETCH = false>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) void NativeSolveVF(
    __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n, int nrhs, int ldb, bool upper,
    int batch, int groups, int offset, int width, bool backward, NativeStage stage, int activeRhs, __gm__ float* work,
    const __gm__ int* check, __ubuf__ aclsolverFloatComplex* shared)
{
    if (check && *check != 0)
    {
        return;
    }
    int group = blockIdx.x % groups;
    int rhsPad = (nrhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT,
        capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int64_t bStride = static_cast<int64_t>(NATIVE_RHS_ROWS) * rhsPad,
            cStride = static_cast<int64_t>(2) * capacity * rhsPad;
    bool column = upper == backward;
    bool smallRhs = BLOCK <= SINGLE_RHS_BLOCK || nrhs <= SMALL_RHS_LIMIT;
    int panels = (n + BLOCK - 1) / BLOCK;
    auto inverse = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work) +
                   static_cast<int64_t>(NATIVE_RHS_ROWS + COMPLEX_COMPONENTS * capacity) * rhsPad * batch;
    for (int m = blockIdx.x / groups; m < batch; m += gridDim.x / groups)
    {
        auto b = barray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(barray[m]) : singleB;
        auto real = work + m * bStride, imag = real + batch * bStride;
        auto outReal = work + 2 * batch * bStride + m * cStride;
        auto outImag = outReal + batch * cStride;
        if (smallRhs && stage == NativeStage::SolveDiagonal)
        {
            SolveInitialDiagonal<BLOCK, PREFETCH>(b, ldb, m, panels, offset, backward, batch, inverse, shared, width,
                                                  group, groups, activeRhs, real, imag, rhsPad, column);
        }
        else if (!smallRhs && (stage == NativeStage::PackRhs || stage == NativeStage::CopyDiagonal))
        {
            PackNativeRhs(b, ldb, offset, width, activeRhs, group, groups, stage, capacity, rhsPad, column, real, imag,
                          outReal, outImag, shared);
        }
        else if (smallRhs)
        {
            ApplySmallRhs<BLOCK, PREFETCH>(b, n, ldb, m, panels, offset, width, backward, batch, group, groups,
                                           activeRhs, inverse, shared, capacity, column, rhsPad, real, imag, outReal,
                                           outImag);
        }
        else
        {
            ApplyLargeRhs(b, n, ldb, offset, width, backward, group, groups, activeRhs, capacity, column, outReal,
                          outImag);
        }
        asc_syncthreads();
    }
}
__aicore__ inline void PrefetchNativeInverse(int n, int nrhs, int batch, int groups, int offset, int width,
                                             bool backward, NativeStage stage, int block, int inverseStride,
                                             __gm__ float* work, AscendC::LocalTensor<float> local)
{
    int core = AscendC::GetBlockIdx(), m = core / groups, group = core % groups;
    int next = stage == NativeStage::SolveDiagonal ? offset : (backward ? offset - block : offset + width);
    int nextWidth = stage == NativeStage::SolveDiagonal ? width : (n - next < block ? n - next : block);
    int first = backward ? 0 : offset + width;
    int owner = stage == NativeStage::SolveDiagonal ? group : ((next - first) / THREADS) % groups;
    if (m < batch && (stage == NativeStage::SolveDiagonal || group == owner))
    {
        int capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT,
            rhsPad = (nrhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT, panels = (n + block - 1) / block;
        auto inverse = reinterpret_cast<__gm__ aclsolverFloatComplex*>(work) +
                       static_cast<int64_t>(NATIVE_RHS_ROWS + COMPLEX_COMPONENTS * capacity) * rhsPad * batch;
        int64_t tile = static_cast<int64_t>(m) * panels + next / block;
        auto inv = inverse + (tile + (backward ? static_cast<int64_t>(batch) * panels : 0)) * block * block;
        AscendC::GlobalTensor<float> global;
        global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inv));

        AscendC::DataCopyExtParams copy{static_cast<uint16_t>(nextWidth),
                                        static_cast<uint32_t>(nextWidth * COMPLEX_BYTES),
                                        (static_cast<int64_t>(block) - nextWidth) * COMPLEX_BYTES,
                                        static_cast<uint32_t>(UbPaddingBlocks(inverseStride, nextWidth)), 0};
        AscendC::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        AscendC::DataCopyPad(local, global, copy, padding);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

__global__ __aicore__ __vector__ void NativeSolveKernel(__gm__ aclsolverFloatComplex* b, __gm__ const uint64_t* barray,
                                                        int n, int nrhs, int ldb, bool upper, int batch, int groups,
                                                        int offset, int width, bool backward, NativeStage stage,
                                                        int activeRhs, __gm__ float* work, const __gm__ int* check)
{
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    int block = SolveBlockSize(nrhs, batch);
    int inverseStride = block + (block == SINGLE_RHS_BLOCK ? NATIVE_PREFETCH_PADDING : 1);
    pipe.InitBuffer(buffer, NativeSolveBytes(block));
    auto d = buffer.Get<uint8_t>();
    // Block64 is the small-batch single-RHS path: every core owns one matrix.
    // DMA keeps the inverse in column-major UB storage, avoiding a SIMT transpose.
    if (block == SINGLE_RHS_BLOCK && (stage == NativeStage::ApplyTrailing || stage == NativeStage::SolveDiagonal) &&
        (!check || *check == 0))
    {
        PrefetchNativeInverse(n, nrhs, batch, groups, offset, width, backward, stage, block, inverseStride, work,
                              buffer.Get<float>()[NATIVE_SHARED_ELEMENTS * COMPLEX_COMPONENTS]);
    }

    if (block == SMALL_SOLVE_BLOCK)
    {
        asc_vf_call<NativeSolveVF<SMALL_SOLVE_BLOCK>>(
            dim3(THREADS), b, barray, n, nrhs, ldb, upper, batch, groups, offset, width, backward, stage, activeRhs,
            work, check, reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(d.GetPhyAddr()));
    }
    else if (block == SINGLE_RHS_BLOCK)
    {
        asc_vf_call<NativeSolveVF<SINGLE_RHS_BLOCK, true>>(
            dim3(THREADS), b, barray, n, nrhs, ldb, upper, batch, groups, offset, width, backward, stage, activeRhs,
            work, check, reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(d.GetPhyAddr()));
    }
    else
    {
        asc_vf_call<NativeSolveVF<MAX_SOLVE_BLOCK>>(dim3(THREADS), b, barray, n, nrhs, ldb, upper, batch, groups,
                                                    offset, width, backward, stage, activeRhs, work, check,
                                                    reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(d.GetPhyAddr()));
    }
}
void ApplyNativeTail(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                     const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b, int ldb, int batch,
                     const int* check, int block, int matrices, int groups, int pass, int offset, int width,
                     int activeRhs, const uint64_t* pointersB, float* scratch)
{
    if (matrices == 0)
    {
        return;
    }
    int rows = pass ? offset : n - offset - width;
    if (!rows)
    {
        return;
    }
    int tailRhs = check && pass ? offset : activeRhs;
    SolveProduct(stream, upper, n, nrhs, a, aarray, lda, scratch, batch, offset, width, pass, check, false, tailRhs);
    int applyGroups = groups;
    if (nrhs <= SMALL_RHS_LIMIT)
    {
        int rowGroups = (rows + THREADS - 1) / THREADS;
        int available = VectorBlocks(MAX_BATCH) / matrices;
        applyGroups = available < rowGroups ? available : rowGroups;
    }
    NativeSolveKernel<<<matrices * applyGroups, NativeSolveBytes(block), stream>>>(
        b, pointersB, n, nrhs, ldb, upper, batch, applyGroups, offset, width, pass, NativeStage::ApplyTrailing, tailRhs,
        scratch, check);
}

void NativeSolvePass(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                     const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b, int ldb, int batch,
                     const int* check, int block, int matrices, int panels, int groups, int pass,
                     const uint64_t* pointersB, float* scratch)
{
    if (matrices == 0)
    {
        return;
    }

    for (int panel = 0; panel < panels; ++panel)
    {
        int offset = (pass ? panels - 1 - panel : panel) * block;
        int width = n - offset < block ? n - offset : block;
        int activeRhs = check ? offset + width : nrhs;
        if (nrhs > SMALL_RHS_LIMIT || panel == 0)
        {
            NativeSolveKernel<<<matrices * groups, NativeSolveBytes(block), stream>>>(
                b, pointersB, n, nrhs, ldb, upper, batch, groups, offset, width, pass,
                nrhs <= SMALL_RHS_LIMIT ? NativeStage::SolveDiagonal : NativeStage::PackRhs, activeRhs, scratch, check);
        }
        if (nrhs > SMALL_RHS_LIMIT)
        {
            SolveProduct(stream, upper, n, nrhs, a, aarray, lda, scratch, batch, offset, width, pass, check, true,
                         activeRhs);
            NativeSolveKernel<<<matrices * groups, NativeSolveBytes(block), stream>>>(
                b, pointersB, n, nrhs, ldb, upper, batch, groups, offset, width, pass, NativeStage::CopyDiagonal,
                activeRhs, scratch, check);
        }
        ApplyNativeTail(stream, upper, n, nrhs, a, aarray, lda, b, ldb, batch, check, block, matrices, groups, pass,
                        offset, width, activeRhs, pointersB, scratch);
    }
}

}  // namespace
void NativeSolve(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                 const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b,
                 aclsolverFloatComplex** barray, int ldb, int batch, aclsolverFloatComplex* work, const int* check)
{
    int block = SolveBlockSize(nrhs, batch);
    int matrices = VectorBlocks(batch);
    if (block == 0)
    {
        return;
    }
    if (matrices == 0)
    {
        return;
    }
    int panels = (n + block - 1) / block;
    int groups = VectorBlocks(MAX_BATCH) / matrices;
    int chunks = nrhs <= SMALL_RHS_LIMIT
                     ? nrhs
                     : ((nrhs + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT + RHS_WARPS - 1) / RHS_WARPS;
    groups = groups < chunks ? groups : chunks;
    auto pointersA = reinterpret_cast<const uint64_t*>(aarray), pointersB = reinterpret_cast<const uint64_t*>(barray);
    auto scratch = reinterpret_cast<float*>(work);
    int inverseMatrices = VectorBlocks(static_cast<int64_t>(batch) * panels);
    if (inverseMatrices == 0)
    {
        return;
    }
    int inverseGroups = VectorBlocks(MAX_BATCH) / inverseMatrices;
    inverseGroups =
        inverseGroups < block / (INVERSE_THREADS / WARP_SIZE) ? inverseGroups : block / (INVERSE_THREADS / WARP_SIZE);
    InvertTilesKernel<<<inverseMatrices * inverseGroups, block*(block + 1) * sizeof(aclsolverFloatComplex), stream>>>(
        a, pointersA, n, nrhs, lda, upper, batch, inverseGroups, scratch, check);
    for (int pass = 0; pass < 2; ++pass)
    {
        NativeSolvePass(stream, upper, n, nrhs, a, aarray, lda, b, ldb, batch, check, block, matrices, panels, groups,
                        pass, pointersB, scratch);
    }
}
}  // namespace Cholesky
