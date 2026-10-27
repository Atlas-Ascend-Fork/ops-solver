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
__simt_vf__ __aicore__ __launch_bounds__(THREADS) void InfoKernelVF(__gm__ int* info, int value)
{
    if (threadIdx.x == 0)
    {
        *info = value;
    }
}

template <int ROWS, bool UPPER, bool BACKWARD, int PART, int ROW, bool ASSESS = false>
__simt_callee__ inline void UpdateSolveRows(const __gm__ aclsolverFloatComplex* a, int n, int lda, int k, float pr,
                                            float pi, float (&re)[ROWS], float (&im)[ROWS], float& energy)
{
    if constexpr ((!BACKWARD && ROW >= PART) || (BACKWARD && ROW <= PART))
    {
        int lane = threadIdx.x % WARP_SIZE, r = lane + ROW * WARP_SIZE;
        bool selected = r < n;
        if constexpr (ROW == PART)
        {
            selected = selected && (BACKWARD ? r < k : r > k);
        }
        if (selected)
        {
            auto z = BACKWARD ? ReadLower(a, lda, k, r, UPPER, true) : ReadLower(a, lda, r, k, UPPER, true);
            if constexpr (ASSESS)
            {
                energy = fmaf(z.real, z.real, fmaf(z.imag, z.imag, energy));
            }
            if constexpr (BACKWARD)
            {
                z.imag = -z.imag;
            }
            auto result = SubProduct({re[ROW], im[ROW]}, z, {pr, pi});
            re[ROW] = result.real;
            im[ROW] = result.imag;
        }
    }
    if constexpr (ROW + 1 < ROWS)
    {
        UpdateSolveRows<ROWS, UPPER, BACKWARD, PART, ROW + 1, ASSESS>(a, n, lda, k, pr, pi, re, im, energy);
    }
}

template <int ROWS, bool UPPER, bool BACKWARD, int PART, int ROW, bool ASSESS>
__simt_callee__ inline void DotSolveRows(const __gm__ aclsolverFloatComplex* a, int n, int lda, int k,
                                         const float (&re)[ROWS], const float (&im)[ROWS], float* energy, float& real,
                                         float& imag, float& columnEnergy)
{
    if constexpr ((BACKWARD && ROW >= PART) || (!BACKWARD && ROW <= PART))
    {
        int r = threadIdx.x % WARP_SIZE + ROW * WARP_SIZE;
        bool selected = r < n;
        if constexpr (ROW == PART)
        {
            selected = selected && (BACKWARD ? r > k : r < k);
        }
        if (selected)
        {
            auto z = BACKWARD ? ReadLower(a, lda, r, k, UPPER) : ReadLower(a, lda, k, r, UPPER);
            if constexpr (ASSESS)
            {
                float norm = fmaf(z.real, z.real, z.imag * z.imag);
                if constexpr (BACKWARD)
                {
                    columnEnergy += norm;
                }
                else
                {
                    energy[ROW] += norm;
                }
            }
            if constexpr (BACKWARD)
            {
                z.imag = -z.imag;
            }
            real = fmaf(z.real, re[ROW], real);
            real = fmaf(-z.imag, im[ROW], real);
            imag = fmaf(z.real, im[ROW], imag);
            imag = fmaf(z.imag, re[ROW], imag);
        }
    }
    if constexpr (ROW + 1 < ROWS)
    {
        DotSolveRows<ROWS, UPPER, BACKWARD, PART, ROW + 1, ASSESS>(a, n, lda, k, re, im, energy, real, imag,
                                                                   columnEnergy);
    }
}

template <bool BACKWARD, bool ASSESS>
__simt_callee__ inline void ReduceSolveDot(float& real, float& imag, float& columnEnergy)
{
    for (int shift = WARP_SIZE / 2; shift; shift /= 2)
    {
        real += asc_shfl_down(real, shift);
        imag += asc_shfl_down(imag, shift);
        if constexpr (BACKWARD && ASSESS)
        {
            columnEnergy += asc_shfl_down(columnEnergy, shift);
        }
    }
    real = asc_shfl(real, 0);
    imag = asc_shfl(imag, 0);
    if constexpr (BACKWARD && ASSESS)
    {
        columnEnergy = asc_shfl(columnEnergy, 0);
    }
}

template <int ROWS, bool UPPER, bool BACKWARD, int PART, bool ASSESS, bool DIRECT_DIAGONAL>
__simt_callee__ inline void UpdateRegisterPivot(const __gm__ aclsolverFloatComplex* a, int n, int lda, int k,
                                                float (&re)[ROWS], float (&im)[ROWS],
                                                const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS], bool& accurate)
{
    constexpr int part = PART;
    int lane = threadIdx.x % WARP_SIZE;
    float inverse =
        DIRECT_DIAGONAL ? 1.0f / ReadLower(a, lda, k, k, UPPER, true).real : asc_shfl(inverseD[part], k % WARP_SIZE);
    float pr = asc_shfl(re[part], k % WARP_SIZE) * inverse, pi = asc_shfl(im[part], k % WARP_SIZE) * inverse;
    if (lane == k % WARP_SIZE)
    {
        re[part] = pr;
        im[part] = pi;
    }
    float energy = 0;
    UpdateSolveRows<ROWS, UPPER, BACKWARD, part, 0, ASSESS>(a, n, lda, k, pr, pi, re, im, energy);
    if constexpr (ASSESS)
    {
        // A lane sum times the warp width bounds the whole column's energy.
        // Conservatively fall back if any lane cannot certify a well-scaled column.
        accurate = accurate || energy * inverse * inverse * static_cast<float>(WARP_SIZE) > CONSERVATIVE_ENERGY_BOUND;
    }
}

template <int ROWS, bool UPPER, bool BACKWARD, int PART, bool ASSESS, bool DIRECT_DIAGONAL>
__simt_callee__ inline void DotRegisterPivot(const __gm__ aclsolverFloatComplex* a, int n, int lda, int k,
                                             float (&re)[ROWS], float (&im)[ROWS],
                                             const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS], float* energy,
                                             bool& accurate)
{
    constexpr int part = PART;
    int lane = threadIdx.x % WARP_SIZE;
    float real = 0, imag = 0, columnEnergy = 0;
    DotSolveRows<ROWS, UPPER, BACKWARD, part, 0, ASSESS>(a, n, lda, k, re, im, energy, real, imag, columnEnergy);
    ReduceSolveDot<BACKWARD, ASSESS>(real, imag, columnEnergy);
    if (lane == k % WARP_SIZE)
    {
        float inverse = DIRECT_DIAGONAL ? 1.0f / ReadLower(a, lda, k, k, UPPER, true).real : inverseD[part];
        re[part] = (re[part] - real) * inverse;
        im[part] = (im[part] - imag) * inverse;
    }
    if constexpr (BACKWARD && ASSESS)
    {
        float inverse = DIRECT_DIAGONAL ? 1.0f / ReadLower(a, lda, k, k, UPPER, true).real
                                        : asc_shfl(inverseD[part], k % WARP_SIZE);
        accurate = accurate || columnEnergy * inverse * inverse > COLUMN_ENERGY_BOUND;
    }
}

template <int ROWS, bool UPPER, bool BACKWARD, int GROUP, bool ASSESS, bool DIRECT_DIAGONAL, bool DOT_MODE>
__simt_callee__ inline bool RegisterSolvePhase(const __gm__ aclsolverFloatComplex* a, int n, int lda, float (&re)[ROWS],
                                               float (&im)[ROWS], const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS],
                                               float* energy)
{
    constexpr int part = BACKWARD ? ROWS - 1 - GROUP : GROUP;
    bool accurate = false;
    for (int step = 0; step < WARP_SIZE; ++step)
    {
        int k = part * WARP_SIZE + (BACKWARD ? WARP_SIZE - 1 - step : step);
        if (k >= n)
        {
            continue;
        }
        if constexpr (DOT_MODE)
        {
            DotRegisterPivot<ROWS, UPPER, BACKWARD, part, ASSESS, DIRECT_DIAGONAL>(a, n, lda, k, re, im, inverseD,
                                                                                   energy, accurate);
        }
        else
        {
            UpdateRegisterPivot<ROWS, UPPER, BACKWARD, part, ASSESS, DIRECT_DIAGONAL>(a, n, lda, k, re, im, inverseD,
                                                                                      accurate);
        }
    }
    if constexpr (GROUP + 1 < ROWS)
    {
        return RegisterSolvePhase<ROWS, UPPER, BACKWARD, GROUP + 1, ASSESS, DIRECT_DIAGONAL, DOT_MODE>(
                   a, n, lda, re, im, inverseD, energy) ||
               accurate;
    }
    return accurate;
}

template <int ROWS, bool UPPER, bool BACKWARD, int GROUP, bool ASSESS = false, bool DIRECT_DIAGONAL = false>
__simt_callee__ inline bool SolvePhase(const __gm__ aclsolverFloatComplex* a, int n, int lda, float (&re)[ROWS],
                                       float (&im)[ROWS], const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS])
{
    return RegisterSolvePhase<ROWS, UPPER, BACKWARD, GROUP, ASSESS, DIRECT_DIAGONAL, false>(a, n, lda, re, im, inverseD,
                                                                                            nullptr);
}

template <int ROWS, bool UPPER, bool BACKWARD, int GROUP, bool ASSESS = true, bool DIRECT_DIAGONAL = false>
__simt_callee__ inline bool SolveDotPhase(const __gm__ aclsolverFloatComplex* a, int n, int lda, float (&re)[ROWS],
                                          float (&im)[ROWS], const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS],
                                          float* energy)
{
    return RegisterSolvePhase<ROWS, UPPER, BACKWARD, GROUP, ASSESS, DIRECT_DIAGONAL, true>(a, n, lda, re, im, inverseD,
                                                                                           energy);
}

template <int ROWS, bool UPPER>
__simt_callee__ inline void SolveWarp(const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray,
                                      __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n,
                                      int nrhs, int lda, int ldb, int batch, const __gm__ int* flags)
{
    int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    for (int64_t task = static_cast<int64_t>(blockIdx.x) * (blockDim.x / WARP_SIZE) + warp;
         task < static_cast<int64_t>(batch) * nrhs; task += static_cast<int64_t>(gridDim.x) * (blockDim.x / WARP_SIZE))
    {
        int m = task / nrhs, rhs = task % nrhs;
        // The compensated pass replaces these jobs completely.
        int accurate = 0, groups = (n + WARP_SIZE - 1) / WARP_SIZE;
        for (int group = 0; group < groups && !accurate; ++group)
        {
            accurate = flags[static_cast<int64_t>(m) * groups + group];
        }
        if (accurate)
        {
            continue;
        }
        auto a = aarray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(aarray[m]) : singleA;
        auto b = (barray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(barray[m]) : singleB) +
                 static_cast<int64_t>(rhs) * ldb;
        float re[ROWS], im[ROWS], inverseD[ROWS];
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
        {
            int r = lane + j * WARP_SIZE;
            auto z = r < n ? ReadLower(b, ldb, r, 0, false) : aclsolverFloatComplex{0, 0};
            re[j] = z.real;
            im[j] = z.imag;
            inverseD[j] = r < n ? 1.0f / ReadLower(a, lda, r, r, UPPER, true).real : 0;
        }
        SolvePhase<ROWS, UPPER, false, 0>(a, n, lda, re, im, inverseD);
        SolvePhase<ROWS, UPPER, true, 0>(a, n, lda, re, im, inverseD);
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
        {
            int r = lane + j * WARP_SIZE;
            if (r < n)
            {
                WriteLower(b, ldb, r, 0, false, {re[j], im[j]});
            }
        }
    }
}

// Each VF uses only the register rows needed for this matrix tile.
template <int ROWS, int VF_THREADS>
__simt_vf__ __aicore__ __launch_bounds__(VF_THREADS) void SolveKernelVF(
    const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* singleB,
    __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb, bool upper, int batch, const __gm__ int* check,
    const __gm__ int* flags)
{
    if (check && *check != 0)
    {
        return;
    }
    if (upper)
    {
        SolveWarp<ROWS, true>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, batch, flags);
    }
    else
    {
        SolveWarp<ROWS, false>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, batch, flags);
    }
}

__simt_callee__ inline bool WarpSolveRisk(bool risk)
{
    int value = risk;
    for (int shift = WARP_SIZE / 2; shift; shift /= 2)
    {
        value |= asc_shfl_down(value, shift);
    }
    return asc_shfl(value, 0) != 0;
}

// Assess each coefficient while solving, retaining the original RHS only for difficult factors.
template <int ROWS, bool UPPER, bool DIRECT_DIAGONAL>
__simt_callee__ inline void SolveLongUpperRegister(const __gm__ aclsolverFloatComplex* a, int n, int lda,
                                                   float (&re)[ROWS], float (&im)[ROWS],
                                                   const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS], float* energy,
                                                   bool& accurate)
{
    int lane = threadIdx.x % WARP_SIZE;

    // Release each column norm before solving to keep the live register set bounded.
#pragma unroll
    for (int part = 0; part < ROWS; ++part)
    {
        int column = lane + part * WARP_SIZE;
        float norm = 0;
        for (int r = column + 1; r < n; ++r)
        {
            auto z = ReadLower(a, lda, r, column, UPPER, true);
            norm = fmaf(z.real, z.real, fmaf(z.imag, z.imag, norm));
        }
        accurate = accurate || norm * inverseD[part] * inverseD[part] > COLUMN_ENERGY_BOUND;
    }
    accurate = WarpSolveRisk(accurate);
    if (!accurate)
    {
        SolveDotPhase<ROWS, UPPER, false, 0, false, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD, energy);
    }
}

template <int ROWS, bool UPPER, bool DIRECT_DIAGONAL>
__simt_callee__ inline void SolveUpperRegister(const __gm__ aclsolverFloatComplex* a, int n, int lda, float (&re)[ROWS],
                                               float (&im)[ROWS], const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS],
                                               float* energy, bool& accurate)
{
    int lane = threadIdx.x % WARP_SIZE;

    if constexpr (ROWS >= WARP_SIZE && !DIRECT_DIAGONAL)
    {
        SolveLongUpperRegister<ROWS, UPPER, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD, energy, accurate);
    }
    else
    {
        SolveDotPhase<ROWS, UPPER, false, 0, true, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD, energy);
        int strong = 0;
#pragma unroll
        for (int row = 0; row < ROWS; ++row)
        {
            int r = lane + row * WARP_SIZE;
            float inverse =
                DIRECT_DIAGONAL ? (r < n ? 1.0f / ReadLower(a, lda, r, r, UPPER, true).real : 0) : inverseD[row];
            strong |= energy[row] * inverse * inverse > COLUMN_ENERGY_BOUND;
        }
        accurate = WarpSolveRisk(strong != 0);
    }
    if (!accurate)
    {
        SolvePhase<ROWS, UPPER, true, 0, false, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD);
    }
}

template <int ROWS, bool UPPER, bool DIRECT_DIAGONAL>
__simt_callee__ inline bool TryRegisterSolve(const __gm__ aclsolverFloatComplex* a, int n, int lda, float (&re)[ROWS],
                                             float (&im)[ROWS], const float (&inverseD)[DIRECT_DIAGONAL ? 1 : ROWS])
{
    bool accurate = false;
    if constexpr (ROWS == 1)
    {
        accurate = WarpSolveRisk(SolvePhase<ROWS, UPPER, false, 0, true, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD));
        if (!accurate)
        {
            SolvePhase<ROWS, UPPER, true, 0, false, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD);
        }
    }
    else
    {
        float energy[UPPER && (ROWS < WARP_SIZE || DIRECT_DIAGONAL) ? ROWS : 1]{};
        if constexpr (UPPER)
        {
            SolveUpperRegister<ROWS, UPPER, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD, energy, accurate);
        }
        else
        {
            accurate =
                WarpSolveRisk(SolvePhase<ROWS, UPPER, false, 0, true, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD));
            if (!accurate)
            {
                SolveDotPhase<ROWS, UPPER, true, 0, false, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD, energy);
            }
        }
    }
    return accurate;
}

template <int ROWS>
__simt_callee__ inline void StoreFusedSolve(bool accurate, __gm__ aclsolverFloatComplex* b, int ldb, int n,
                                            __gm__ aclsolverFloatComplex* saved, const float (&re)[ROWS],
                                            const float (&im)[ROWS])
{
    int lane = threadIdx.x % WARP_SIZE;
    if (accurate)
    {
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
        {
            int row = lane + j * WARP_SIZE;
            if (row < n)
            {
                WriteLower(saved, n, row, 0, false, ReadLower(b, ldb, row, 0, false));
            }
        }
    }
    else
    {
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
        {
            int row = lane + j * WARP_SIZE;
            if (row < n)
            {
                WriteLower(b, ldb, row, 0, false, {re[j], im[j]});
            }
        }
    }
}

template <int ROWS, bool UPPER, int ACTIVE_WARPS, bool DIRECT_DIAGONAL = false>
__simt_callee__ inline void FusedSolveWarp(const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray,
                                           __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n,
                                           int nrhs, int lda, int ldb, int batch, __gm__ aclsolverFloatComplex* backup,
                                           __gm__ int* flags)
{
    int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;
    if (warp >= ACTIVE_WARPS)
    {
        return;
    }
    for (int64_t task = static_cast<int64_t>(blockIdx.x) * ACTIVE_WARPS + warp;
         task < static_cast<int64_t>(batch) * nrhs; task += static_cast<int64_t>(gridDim.x) * ACTIVE_WARPS)
    {
        int m = task / nrhs, rhs = task % nrhs;
        auto a = aarray ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(aarray[m]) : singleA;
        auto b = (barray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(barray[m]) : singleB) +
                 static_cast<int64_t>(rhs) * ldb;
        float re[ROWS], im[ROWS], inverseD[DIRECT_DIAGONAL ? 1 : ROWS];
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
        {
            int row = lane + j * WARP_SIZE;
            auto z = row < n ? ReadLower(b, ldb, row, 0, false) : aclsolverFloatComplex{0, 0};
            re[j] = z.real;
            im[j] = z.imag;
            if constexpr (!DIRECT_DIAGONAL)
            {
                inverseD[j] = row < n ? 1.0f / ReadLower(a, lda, row, row, UPPER, true).real : 0;
            }
        }
        bool accurate = TryRegisterSolve<ROWS, UPPER, DIRECT_DIAGONAL>(a, n, lda, re, im, inverseD);
        if (lane == 0)
        {
            flags[task] = accurate;
        }
        StoreFusedSolve<ROWS>(accurate, b, ldb, n, backup + task * n, re, im);
    }
}
template <int ROWS, int VF_THREADS = SOLVE_THREADS, int ACTIVE_WARPS = VF_THREADS / WARP_SIZE,
          bool DIRECT_DIAGONAL = false>
__simt_vf__ __aicore__ __launch_bounds__(VF_THREADS) void FusedSolveVF(
    const __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* b,
    __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb, bool upper, int batch,
    __gm__ aclsolverFloatComplex* backup, __gm__ int* flags)
{
    if (upper)
    {
        FusedSolveWarp<ROWS, true, ACTIVE_WARPS, DIRECT_DIAGONAL>(a, aarray, b, barray, n, nrhs, lda, ldb, batch,
                                                                  backup, flags);
    }
    else
    {
        FusedSolveWarp<ROWS, false, ACTIVE_WARPS, DIRECT_DIAGONAL>(a, aarray, b, barray, n, nrhs, lda, ldb, batch,
                                                                   backup, flags);
    }
    asc_threadfence();
}

// Preparation checks singularity before any factor entry can be overwritten.
__simt_callee__ inline void InitializeInverseInfo(const __gm__ aclsolverFloatComplex* a, int n, int lda,
                                                  __gm__ int* info)
{
    int failure = 0;
    for (int k = 0; k < n; ++k)
    {
        if (a[static_cast<int64_t>(k) * lda + k].real == 0)
        {
            failure = k + 1;
            break;
        }
    }
    *info = failure;
}

template <bool COPY>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) void InverseKernelVF(__gm__ aclsolverFloatComplex* a,
                                                                       __gm__ aclsolverFloatComplex* work, int n,
                                                                       int lda, bool upper, __gm__ int* info)
{
    int64_t start = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if constexpr (!COPY)
    {
        if (start == 0)
        {
            InitializeInverseInfo(a, n, lda, info);
        }
        for (int64_t pos = start; pos < static_cast<int64_t>(n) * n; pos += gridDim.x * blockDim.x)
        {
            WriteLower(work, n, pos % n, pos / n, false, {pos / n == pos % n ? 1.0f : 0.0f, 0});
        }
    }
    else
    {
        if (*info != 0)
        {
            return;
        }
        for (int64_t pos = start; pos < static_cast<int64_t>(n) * n; pos += gridDim.x * blockDim.x)
        {
            int r = pos % n, c = pos / n;
            if ((upper && r <= c) || (!upper && r >= c))
            {
                auto z = ReadLower(work, n, upper ? c : r, upper ? r : c, false);
                if (upper)
                {
                    z.imag = -z.imag;
                }
                if (r == c)
                {
                    z.imag = 0;
                }
                WriteLower(a, lda, r, c, false, z);
            }
        }
    }
}
__global__ __aicore__ __vector__ void InfoKernel(__gm__ int* info, int value)
{
    asc_vf_call<InfoKernelVF>(dim3(THREADS), info, value);
}
// Keep a complete small matrix in UB; compensate the initial rank-one cancellation.

// Each warp owns its RHS in UB; shuffles exchange pivots without block-wide barriers.
template <bool UPPER>
__simt_callee__ inline void BufferedDotStep(const __gm__ aclsolverFloatComplex* a, int n, int lda, int k, int pass,
                                            __ubuf__ aclsolverFloatComplex* local, float maxInverse, bool& risky)
{
    int lane = threadIdx.x % WARP_SIZE;

    float real = 0, imag = 0, norm = 0;
    int firstRow = pass ? ((k + WARP_SIZE - lane) / WARP_SIZE * WARP_SIZE + lane) : lane, endRow = pass ? n : k;
#pragma unroll 2
    for (int row = firstRow; row < endRow; row += WARP_SIZE)
    {
        auto coefficient = pass ? ReadLower(a, lda, row, k, UPPER) : ReadLower(a, lda, k, row, UPPER);
        if constexpr (UPPER)
        {
            if (!pass)
            {
                norm = fmaf(coefficient.real, coefficient.real, fmaf(coefficient.imag, coefficient.imag, norm));
            }
        }
        if (pass)
        {
            coefficient.imag = -coefficient.imag;
        }
        auto value = ReadUB(local, row);
        real = fmaf(coefficient.real, value.real, real);
        real = fmaf(-coefficient.imag, value.imag, real);
        imag = fmaf(coefficient.real, value.imag, imag);
        imag = fmaf(coefficient.imag, value.real, imag);
    }
    // Each column energy is at most n times the greatest row energy.
    // A warp row sum is at most WARP_SIZE times the greatest lane partial sum.
    if constexpr (UPPER)
    {
        if (!pass)
        {
            risky = risky || norm * maxInverse * maxInverse * (static_cast<float>(WARP_SIZE) * n) > COLUMN_ENERGY_BOUND;
        }
    }
    for (int shift = WARP_SIZE / 2; shift; shift /= 2)
    {
        real += asc_shfl_down(real, shift);
        imag += asc_shfl_down(imag, shift);
    }
    real = asc_shfl(real, 0);
    imag = asc_shfl(imag, 0);
    if (lane == k % WARP_SIZE)
    {
        auto value = ReadUB(local, k);
        float inverse = 1.0f / ReadLower(a, lda, k, k, UPPER, true).real;
        WriteUB(local, k, {(value.real - real) * inverse, (value.imag - imag) * inverse});
    }
}

// The update-oriented passes are LOWER forward and UPPER backward.
// Reverse logical indices for the latter; both access coalesced columns in stored order.
template <bool BACKWARD>
__simt_callee__ inline int BufferedIndex(int n, int index)
{
    return BACKWARD ? n - 1 - index : index;
}

// Round each trailing RHS entry once per panel instead of once per coefficient.
template <bool BACKWARD>
__simt_callee__ inline aclsolverFloatComplex SolveBufferedPanel(const __gm__ aclsolverFloatComplex* a, int n, int lda,
                                                                int first, __ubuf__ aclsolverFloatComplex* local,
                                                                float inverse, float (&energy)[WARP_SIZE])
{
    int lane = threadIdx.x % WARP_SIZE, order = first + lane;
    int row = BufferedIndex<BACKWARD>(n, order);
    auto value = order < n ? ReadUB(local, row) : aclsolverFloatComplex{0, 0};
#pragma unroll
    for (int column = 0; column < WARP_SIZE; ++column)
    {
        if (first + column < n)
        {
            if (lane == column)
            {
                value.real *= inverse;
                value.imag *= inverse;
            }
            float real = asc_shfl(value.real, column), imag = asc_shfl(value.imag, column);
            if (lane > column && order < n)
            {
                auto coefficient = ReadLower(a, lda, row, BufferedIndex<BACKWARD>(n, first + column), false);
                if constexpr (!BACKWARD)
                {
                    energy[column] = fmaf(coefficient.real, coefficient.real, coefficient.imag * coefficient.imag);
                }
                value = SubProduct(value, coefficient, {real, imag});
            }
        }
    }
    if (order < n)
    {
        WriteUB(local, row, value);
    }
    return value;
}

template <bool BACKWARD>
__simt_callee__ inline void UpdateBufferedTrailing(const __gm__ aclsolverFloatComplex* a, int n, int lda, int first,
                                                   __ubuf__ aclsolverFloatComplex* local, aclsolverFloatComplex pivot,
                                                   float (&energy)[WARP_SIZE])
{
    int lane = threadIdx.x % WARP_SIZE;
    // Keep all lanes active for pivot shuffles in a partial trailing row tile.
    for (int group = first + WARP_SIZE; group < n; group += WARP_SIZE)
    {
        int order = group + lane, row = BufferedIndex<BACKWARD>(n, order);
        float real = 0, imag = 0;
#pragma unroll
        for (int column = 0; column < WARP_SIZE; ++column)
        {
            float pr = asc_shfl(pivot.real, column), pi = asc_shfl(pivot.imag, column);
            if (order < n)
            {
                auto coefficient = ReadLower(a, lda, row, BufferedIndex<BACKWARD>(n, first + column), false);
                real = fmaf(coefficient.real, pr, real);
                real = fmaf(-coefficient.imag, pi, real);
                imag = fmaf(coefficient.real, pi, imag);
                imag = fmaf(coefficient.imag, pr, imag);
                if constexpr (!BACKWARD)
                {
                    energy[column] = fmaf(coefficient.real, coefficient.real,
                                          fmaf(coefficient.imag, coefficient.imag, energy[column]));
                }
            }
        }
        if (order < n)
        {
            auto value = ReadUB(local, row);
            WriteUB(local, row, {value.real - real, value.imag - imag});
        }
    }
}

__simt_callee__ inline bool BufferedForwardRisk(int n, int first, float inverse, const float (&energy)[WARP_SIZE])
{
    bool risky = false;
#pragma unroll
    for (int column = 0; column < WARP_SIZE; ++column)
    {
        if (first + column < n)
        {
            float pivotInverse = asc_shfl(inverse, column);
            risky = risky || energy[column] * pivotInverse * pivotInverse * static_cast<float>(WARP_SIZE) >
                                 CONSERVATIVE_ENERGY_BOUND;
        }
    }
    return WarpSolveRisk(risky);
}

template <bool BACKWARD>
__simt_callee__ inline bool BufferedUpdatePass(const __gm__ aclsolverFloatComplex* a, int n, int lda,
                                               __ubuf__ aclsolverFloatComplex* local)
{
    int lane = threadIdx.x % WARP_SIZE;
    for (int first = 0; first < n; first += WARP_SIZE)
    {
        float energy[WARP_SIZE]{};
        int order = first + lane, row = BufferedIndex<BACKWARD>(n, order);
        float inverse = order < n ? 1.0f / ReadLower(a, lda, row, row, false, true).real : 0;
        auto pivot = SolveBufferedPanel<BACKWARD>(a, n, lda, first, local, inverse, energy);
        UpdateBufferedTrailing<BACKWARD>(a, n, lda, first, local, pivot, energy);
        if constexpr (!BACKWARD)
        {
            if (BufferedForwardRisk(n, first, inverse, energy))
            {
                return true;
            }
        }
    }
    return false;
}

template <bool UPPER>
__simt_callee__ inline void BufferedSolvePass(const __gm__ aclsolverFloatComplex* a, int n, int lda, int pass,
                                              __ubuf__ aclsolverFloatComplex* local, float maxInverse, bool& risky,
                                              int64_t task, __gm__ int* flags)
{
    int lane = threadIdx.x % WARP_SIZE;
    if (UPPER == static_cast<bool>(pass))
    {
        risky = BufferedUpdatePass<UPPER>(a, n, lda, local);
        if (!pass && !lane)
        {
            flags[task] = risky;
        }
        return;
    }
    for (int step = 0; step < n; ++step)
    {
        int k = pass ? n - 1 - step : step;
        BufferedDotStep<UPPER>(a, n, lda, k, pass, local, maxInverse, risky);
    }
    if (!pass)
    {
        risky = WarpSolveRisk(risky);
        if (!lane)
        {
            flags[task] = risky;
        }
    }
}

template <bool UPPER>
__simt_callee__ inline float LoadBufferedRhs(const __gm__ aclsolverFloatComplex* a,
                                             const __gm__ aclsolverFloatComplex* b, int n, int lda, int ldb,
                                             __ubuf__ aclsolverFloatComplex* local)
{
    int lane = threadIdx.x % WARP_SIZE;
    float maxInverse = 0;
    for (int row = lane; row < n; row += WARP_SIZE)
    {
        auto z = ReadLower(b, ldb, row, 0, false);
        WriteUB(local, row, z);
        if constexpr (UPPER)
        {
            maxInverse = fmaxf(maxInverse, fabsf(1.0f / ReadLower(a, lda, row, row, UPPER, true).real));
        }
    }
    if constexpr (UPPER)
    {
        for (int shift = WARP_SIZE / 2; shift; shift /= 2)
        {
            maxInverse = fmaxf(maxInverse, asc_shfl_down(maxInverse, shift));
        }
        maxInverse = asc_shfl(maxInverse, 0);
    }
    return maxInverse;
}

template <bool UPPER, int VF_THREADS>
__simt_vf__ __aicore__ __launch_bounds__(VF_THREADS) void WarpBufferedSolveVF(
    const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* singleB,
    __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb, int batch, __gm__ aclsolverFloatComplex* backup,
    __gm__ int* flags, __ubuf__ uint8_t* storage)
{
    constexpr int warps = VF_THREADS / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE,
        padded = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    auto local = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(storage + warp * padded * COMPLEX_BYTES);
    int64_t count = static_cast<int64_t>(batch) * nrhs;
    int64_t begin = count * blockIdx.x / gridDim.x, end = count * (blockIdx.x + 1) / gridDim.x;
    for (int64_t task = begin + warp; task < end; task += warps)
    {
        int m = task / nrhs, rhs = task % nrhs;
        auto a = aarray ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(aarray[m]) : singleA;
        auto b = (barray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(barray[m]) : singleB) +
                 static_cast<int64_t>(rhs) * ldb;
        float maxInverse = LoadBufferedRhs<UPPER>(a, b, n, lda, ldb, local);
        bool risky = false;
#pragma unroll
        for (int pass = 0; pass < 2; ++pass)
        {
            BufferedSolvePass<UPPER>(a, n, lda, pass, local, maxInverse, risky, task, flags);
            if (risky && !pass)
            {
                break;
            }
        }
        for (int row = lane; row < n; row += WARP_SIZE)
        {
            if (risky)
            {
                WriteLower(backup + task * n, n, row, 0, false, ReadLower(b, ldb, row, 0, false));
            }
            else
            {
                WriteLower(b, ldb, row, 0, false, ReadUB(local, row));
            }
        }
    }
    asc_threadfence();
}

template <int VF_THREADS>
__aicore__ inline void CallBufferedSolve(const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray,
                                         __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n,
                                         int nrhs, int lda, int ldb, bool upper, int batch,
                                         __gm__ aclsolverFloatComplex* backup, __gm__ int* flags,
                                         __ubuf__ uint8_t* storage)
{
    if (upper)
    {
        asc_vf_call<WarpBufferedSolveVF<true, VF_THREADS>>(dim3(VF_THREADS), singleA, aarray, singleB, barray, n, nrhs,
                                                           lda, ldb, batch, backup, flags, storage);
    }
    else
    {
        asc_vf_call<WarpBufferedSolveVF<false, VF_THREADS>>(dim3(VF_THREADS), singleA, aarray, singleB, barray, n, nrhs,
                                                            lda, ldb, batch, backup, flags, storage);
    }
}

__aicore__ inline void RunBufferedSolve(const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray,
                                        __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n,
                                        int nrhs, int lda, int ldb, bool upper, int batch,
                                        __gm__ aclsolverFloatComplex* backup, __gm__ int* flags)
{
    int padded = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    int warps = BufferedWarps(n);
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, warps * padded * COMPLEX_BYTES);
    auto storage = reinterpret_cast<__ubuf__ uint8_t*>(buffer.Get<uint8_t>().GetPhyAddr());
    if (warps == MAX_BUFFERED_WARPS)
    {
        CallBufferedSolve<MAX_BUFFERED_WARPS * WARP_SIZE>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, upper,
                                                          batch, backup, flags, storage);
    }
    else if (warps == MEDIUM_BUFFERED_WARPS)
    {
        CallBufferedSolve<MEDIUM_BUFFERED_WARPS * WARP_SIZE>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, upper,
                                                             batch, backup, flags, storage);
    }
    else if (warps == INITIAL_BUFFERED_WARPS)
    {
        CallBufferedSolve<INITIAL_BUFFERED_WARPS * WARP_SIZE>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb,
                                                              upper, batch, backup, flags, storage);
    }
    else if (warps == INITIAL_BUFFERED_WARPS / 2)
    {
        CallBufferedSolve<INITIAL_BUFFERED_WARPS / 2 * WARP_SIZE>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb,
                                                                  upper, batch, backup, flags, storage);
    }
    else if (warps == MIN_BUFFERED_WARPS)
    {
        CallBufferedSolve<MIN_BUFFERED_WARPS * WARP_SIZE>(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, upper,
                                                          batch, backup, flags, storage);
    }
    return;
}

__aicore__ inline void DispatchWideSolve(const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray,
                                         __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n,
                                         int nrhs, int lda, int ldb, bool upper, int batch, const __gm__ int* check,
                                         const __gm__ int* flags)
{
    if (n <= 5 * WARP_SIZE)
    {
        asc_vf_call<SolveKernelVF<5, LARGE_SOLVE_THREADS>>(dim3(LARGE_SOLVE_THREADS), singleA, aarray, singleB, barray,
                                                           n, nrhs, lda, ldb, upper, batch, check, flags);
    }
    else if (n <= 6 * WARP_SIZE)
    {
        asc_vf_call<SolveKernelVF<6, LARGE_SOLVE_THREADS>>(dim3(LARGE_SOLVE_THREADS), singleA, aarray, singleB, barray,
                                                           n, nrhs, lda, ldb, upper, batch, check, flags);
    }
    else if (n <= 7 * WARP_SIZE)
    {
        asc_vf_call<SolveKernelVF<7, LARGE_SOLVE_THREADS>>(dim3(LARGE_SOLVE_THREADS), singleA, aarray, singleB, barray,
                                                           n, nrhs, lda, ldb, upper, batch, check, flags);
    }
    else
    {
        asc_vf_call<SolveKernelVF<8, LARGE_SOLVE_THREADS>>(dim3(LARGE_SOLVE_THREADS), singleA, aarray, singleB, barray,
                                                           n, nrhs, lda, ldb, upper, batch, check, flags);
    }
}

template <bool ADAPTIVE = false>
__global__ __aicore__ __vector__ void SolveKernel(const __gm__ aclsolverFloatComplex* singleA,
                                                  __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* singleB,
                                                  __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb,
                                                  bool upper, int batch, const __gm__ int* check, __gm__ int* flags,
                                                  __gm__ aclsolverFloatComplex* backup)
{
    if (ADAPTIVE && n > FUSED_REGISTER_ORDER)
    {
        RunBufferedSolve(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
        return;
    }
    if constexpr (ADAPTIVE)
    {
        return;
    }
    else
    {
        if (n <= WARP_SIZE)
        {
            asc_vf_call<SolveKernelVF<1, SOLVE_THREADS>>(dim3(SOLVE_THREADS), singleA, aarray, singleB, barray, n, nrhs,
                                                         lda, ldb, upper, batch, check, flags);
        }
        else if (n <= 2 * WARP_SIZE)
        {
            asc_vf_call<SolveKernelVF<2, SOLVE_THREADS>>(dim3(SOLVE_THREADS), singleA, aarray, singleB, barray, n, nrhs,
                                                         lda, ldb, upper, batch, check, flags);
        }
        else if (n <= 3 * WARP_SIZE)
        {
            asc_vf_call<SolveKernelVF<3, SOLVE_THREADS>>(dim3(SOLVE_THREADS), singleA, aarray, singleB, barray, n, nrhs,
                                                         lda, ldb, upper, batch, check, flags);
        }
        else if (n <= REGISTER_SOLVE_ORDER)
        {
            asc_vf_call<SolveKernelVF<4, SOLVE_THREADS>>(dim3(SOLVE_THREADS), singleA, aarray, singleB, barray, n, nrhs,
                                                         lda, ldb, upper, batch, check, flags);
        }
        else
        {
            DispatchWideSolve(singleA, aarray, singleB, barray, n, nrhs, lda, ldb, upper, batch, check, flags);
        }
    }
}
template <bool COPY>
__global__ __aicore__ __vector__ void InverseKernel(__gm__ aclsolverFloatComplex* a, __gm__ aclsolverFloatComplex* work,
                                                    int n, int lda, bool upper, __gm__ int* info)
{
    asc_vf_call<InverseKernelVF<COPY>>(dim3(THREADS), a, work, n, lda, upper, info);
}
// Large column energy identifies factors whose cancellation needs more than FP32 accumulation.
template <int LANES, int ROWS>
__simt_callee__ inline void AssessSmallSolve(const __gm__ aclsolverFloatComplex* a, __gm__ aclsolverFloatComplex* b,
                                             __gm__ aclsolverFloatComplex* saved, int n, int nrhs, int lda, int ldb,
                                             bool upper, int flag, __gm__ int* flags)
{
    int lane = threadIdx.x % LANES, shuffleBase = (threadIdx.x % WARP_SIZE) / LANES * LANES;

    int accurate = 0;
    for (int col = 0; col < n && !accurate; ++col)
    {
        float energy = 0;
#pragma unroll
        for (int part = 0; part < ROWS; ++part)
        {
            int row = lane + part * LANES;
            auto z = row > col && row < n ? ReadLower(a, lda, row, col, upper, true) : aclsolverFloatComplex{0, 0};
            energy += fmaf(z.real, z.real, z.imag * z.imag);
        }
        for (int shift = LANES / 2; shift; shift /= 2)
        {
            float next = asc_shfl(energy, shuffleBase + (lane + shift) % LANES);
            if (lane < shift)
            {
                energy += next;
            }
        }
        float diagonal = ReadLower(a, lda, col, col, upper, true).real;
        accurate = asc_shfl(energy, shuffleBase) > COLUMN_ENERGY_BOUND * diagonal * diagonal;
    }
    if (lane == 0)
    {
        for (int group = 0; group < (n + WARP_SIZE - 1) / WARP_SIZE; ++group)
        {
            flags[flag + group] = accurate;
        }
    }
    if (accurate)
    {
        for (int64_t pos = lane; pos < static_cast<int64_t>(n) * nrhs; pos += LANES)
        {
            WriteLower(saved, n, pos % n, pos / n, false, ReadLower(b, ldb, pos % n, pos / n, false));
        }
    }
}

template <int LANES, int ROWS>
__simt_callee__ inline void AccurateRegisterStep(const __gm__ aclsolverFloatComplex* a, int n, int lda, bool upper,
                                                 int pass, int step, aclsolverFloatComplex (&value)[ROWS],
                                                 aclsolverFloatComplex (&low)[ROWS])
{
    int lane = threadIdx.x % LANES, shuffleBase = (threadIdx.x % WARP_SIZE) / LANES * LANES;

    int k = pass ? n - 1 - step : step;
    aclsolverFloatComplex pivot{0, 0}, pivotLow{0, 0};
#pragma unroll
    for (int part = 0; part < ROWS; ++part)
    {
        if (k / LANES == part)
        {
            int owner = shuffleBase + k % LANES;
            pivot = {asc_shfl(value[part].real, owner), asc_shfl(value[part].imag, owner)};
            pivotLow = {asc_shfl(low[part].real, owner), asc_shfl(low[part].imag, owner)};
        }
    }
    pivot = DividePair(pivot, pivotLow, ReadLower(a, lda, k, k, upper, true).real);
#pragma unroll
    for (int part = 0; part < ROWS; ++part)
    {
        int row = lane + part * LANES;
        if (row == k)
        {
            value[part] = pivot;
            low[part] = pivotLow;
        }
        else if (row < n && (pass ? row < k : row > k))
        {
            auto coefficient = pass ? ReadLower(a, lda, k, row, upper, true) : ReadLower(a, lda, row, k, upper, true);
            if (pass)
            {
                coefficient.imag = -coefficient.imag;
            }
            value[part] = SubPairProduct(value[part], low[part], coefficient, pivot, pivotLow);
        }
    }
}

template <int LANES, int ROWS>
__simt_callee__ inline void SolveCompensatedRegisters(const __gm__ aclsolverFloatComplex* a,
                                                      __gm__ aclsolverFloatComplex* b,
                                                      const __gm__ aclsolverFloatComplex* saved, int n, int lda,
                                                      int ldb, bool upper, int rhs)
{
    int lane = threadIdx.x % LANES;
    aclsolverFloatComplex value[ROWS], low[ROWS];
#pragma unroll
    for (int part = 0; part < ROWS; ++part)
    {
        int row = lane + part * LANES;
        value[part] = row < n ? ReadLower(saved, n, row, rhs, false) : aclsolverFloatComplex{0, 0};
        low[part] = {0, 0};
    }
    for (int pass = 0; pass < 2; ++pass)
    {
        for (int step = 0; step < n; ++step)
        {
            AccurateRegisterStep<LANES, ROWS>(a, n, lda, upper, pass, step, value, low);
        }
    }
#pragma unroll
    for (int part = 0; part < ROWS; ++part)
    {
        int row = lane + part * LANES;
        if (row < n)
        {
            WriteLower(b, ldb, row, rhs, false, value[part]);
        }
    }
}

template <int LANES, int ROWS = 1, int VF_THREADS = THREADS, bool PER_RHS = false, int GROUPS = 0>
__simt_vf__ __aicore__ __launch_bounds__(VF_THREADS) void SmallAccurateSolveVF(
    const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* singleB,
    __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb, bool upper, int batch,
    __gm__ aclsolverFloatComplex* backup, __gm__ int* flags, bool assess)
{
    int group = threadIdx.x / LANES;
    constexpr int groups = GROUPS ? GROUPS : VF_THREADS / LANES;
    if (group >= groups)
    {
        return;
    }
    int64_t jobs = static_cast<int64_t>(batch) * (assess ? 1 : nrhs);
    for (int64_t job = static_cast<int64_t>(blockIdx.x) * groups + group; job < jobs;
         job += static_cast<int64_t>(gridDim.x) * groups)
    {
        int m = assess ? job : job / nrhs, rhs = assess ? 0 : job % nrhs;
        auto a = aarray ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(aarray[m]) : singleA;
        auto b = barray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(barray[m]) : singleB;
        auto saved = backup + static_cast<int64_t>(m) * n * nrhs;
        int flag = PER_RHS ? m * nrhs + rhs : m * ((n + WARP_SIZE - 1) / WARP_SIZE);
        if (assess)
        {
            AssessSmallSolve<LANES, ROWS>(a, b, saved, n, nrhs, lda, ldb, upper, flag, flags);
        }
        else if (flags[flag])
        {
            SolveCompensatedRegisters<LANES, ROWS>(a, b, saved, n, lda, ldb, upper, rhs);
        }
    }
}
template <int ROWS, int VF_THREADS, int ACTIVE_WARPS = VF_THREADS / WARP_SIZE>
__aicore__ inline void RunFusedSolve(const __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* aarray,
                                     __gm__ aclsolverFloatComplex* b, __gm__ const uint64_t* barray, int n, int nrhs,
                                     int lda, int ldb, bool upper, int batch, __gm__ aclsolverFloatComplex* backup,
                                     __gm__ int* flags)
{
    asc_vf_call<FusedSolveVF<ROWS, VF_THREADS, ACTIVE_WARPS>>(dim3(VF_THREADS), a, aarray, b, barray, n, nrhs, lda, ldb,
                                                              upper, batch, backup, flags);
    AscendC::PipeBarrier<PIPE_ALL>();
    // The same warp handles the same RHS in both VFs; flags and backup need no cross-core barrier.
    asc_vf_call<SmallAccurateSolveVF<WARP_SIZE, ROWS, VF_THREADS, true, ACTIVE_WARPS>>(
        dim3(VF_THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags, false);
}

__simt_callee__ inline void AssessUpperColumns(const __gm__ aclsolverFloatComplex* a, int n, int lda, int group,
                                               int64_t job, __gm__ int* flags, __ubuf__ aclsolverFloatComplex* shared)
{
    int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;

    int col = group * WARP_SIZE + lane;
    float energy = 0;
    for (int row = group * WARP_SIZE + 1 + warp; row < n; row += THREADS / WARP_SIZE)
    {
        if (col < n && row > col)
        {
            auto z = ReadLower(a, lda, row, col, true, true);
            energy = fmaf(z.real, z.real, fmaf(z.imag, z.imag, energy));
        }
    }
    WriteUB(shared, threadIdx.x, {energy, 0});
    asc_syncthreads();
    if (threadIdx.x < WARP_SIZE)
    {
        float sum = 0;
        for (int w = 0; w < THREADS / WARP_SIZE; ++w)
        {
            sum += ReadUB(shared, w * WARP_SIZE + lane).real;
        }
        float diagonal = col < n ? ReadLower(a, lda, col, col, true, true).real : 0;
        WriteUB(shared, lane, {static_cast<float>(col < n && sum > COLUMN_ENERGY_BOUND * diagonal * diagonal), 0});
    }
    asc_syncthreads();
    if (threadIdx.x == 0)
    {
        int any = 0;
        for (int c = 0; c < WARP_SIZE; ++c)
        {
            any |= ReadUB(shared, c).real != 0;
        }
        flags[job] = any;
    }
}

__simt_callee__ inline void AssessLowerColumns(const __gm__ aclsolverFloatComplex* a, int n, int lda, int group,
                                               bool upper, int64_t job, __gm__ int* flags,
                                               __ubuf__ aclsolverFloatComplex* shared)
{
    int lane = threadIdx.x % WARP_SIZE, warp = threadIdx.x / WARP_SIZE;

    int accurate = 0;
    for (int col = group * WARP_SIZE + warp; col < n && col < (group + 1) * WARP_SIZE && !accurate;
         col += THREADS / WARP_SIZE)
    {
        float energy = 0;
        for (int row = col + 1 + lane; row < n; row += WARP_SIZE)
        {
            auto z = ReadLower(a, lda, row, col, upper, true);
            energy = fmaf(z.real, z.real, fmaf(z.imag, z.imag, energy));
        }
        for (int shift = WARP_SIZE / 2; shift; shift /= 2)
        {
            energy += asc_shfl_down(energy, shift);
        }
        float diagonal = ReadLower(a, lda, col, col, upper, true).real;
        accurate = asc_shfl(energy, 0) > COLUMN_ENERGY_BOUND * diagonal * diagonal;
    }
    if (lane == 0)
    {
        WriteUB(shared, warp, {static_cast<float>(accurate), 0});
    }
    asc_syncthreads();
    if (threadIdx.x == 0)
    {
        int any = 0;
        for (int w = 0; w < THREADS / WARP_SIZE; ++w)
        {
            any |= ReadUB(shared, w).real != 0;
        }
        flags[job] = any;
    }
}

__simt_vf__ __aicore__ __launch_bounds__(THREADS) void AssessSolveVF(
    const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray,
    const __gm__ aclsolverFloatComplex* singleB, __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb,
    bool upper, int batch, __gm__ aclsolverFloatComplex* backup, __gm__ int* flags, const __gm__ int* check,
    __ubuf__ aclsolverFloatComplex* shared)
{
    if (check && *check != 0)
    {
        return;
    }
    int groups = (n + WARP_SIZE - 1) / WARP_SIZE;
    for (int64_t job = blockIdx.x; job < static_cast<int64_t>(batch) * groups; job += gridDim.x)
    {
        int m = job / groups, group = job % groups;
        auto a = aarray ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(aarray[m]) : singleA;
        if (upper)
        {
            AssessUpperColumns(a, n, lda, group, job, flags, shared);
        }
        else
        {
            AssessLowerColumns(a, n, lda, group, upper, job, flags, shared);
        }
        auto b = barray ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(barray[m]) : singleB;
        auto saved = backup + static_cast<int64_t>(m) * n * nrhs;
        int first = group * WARP_SIZE, rows = n - first < WARP_SIZE ? n - first : WARP_SIZE;
        for (int64_t pos = threadIdx.x; pos < static_cast<int64_t>(rows) * nrhs; pos += THREADS)
        {
            WriteLower(saved, n, first + pos % rows, pos / rows, false,
                       ReadLower(b, ldb, first + pos % rows, pos / rows, false));
        }
        asc_syncthreads();
    }
}
__simt_callee__ inline void AccurateGlobalStep(const __gm__ aclsolverFloatComplex* a, __gm__ aclsolverFloatComplex* b,
                                               __gm__ aclsolverFloatComplex* tail, int n, int lda, bool upper, int pass,
                                               int step, __ubuf__ aclsolverFloatComplex* pivot)
{
    int k = pass ? n - 1 - step : step;
    if (threadIdx.x == 0)
    {
        auto remainder = ReadLower(tail, n, k, 0, false);
        auto value = DividePair(ReadLower(b, n, k, 0, false), remainder, ReadLower(a, lda, k, k, upper, true).real);
        WriteLower(b, n, k, 0, false, value);
        WriteLower(tail, n, k, 0, false, remainder);
        WriteUB(pivot, 0, value);
        WriteUB(pivot, 1, remainder);
    }
    asc_syncthreads();
    auto value = ReadUB(pivot, 0), remainder = ReadUB(pivot, 1);
    for (int row = threadIdx.x; row < n; row += THREADS)
    {
        if (pass ? row < k : row > k)
        {
            auto coefficient = pass ? ReadLower(a, lda, k, row, upper, true) : ReadLower(a, lda, row, k, upper, true);
            if (pass)
            {
                coefficient.imag = -coefficient.imag;
            }
            auto tailValue = ReadLower(tail, n, row, 0, false);
            auto result = SubPairProduct(ReadLower(b, n, row, 0, false), tailValue, coefficient, value, remainder);
            WriteLower(b, n, row, 0, false, result);
            WriteLower(tail, n, row, 0, false, tailValue);
        }
    }
    asc_syncthreads();
}

__simt_callee__ inline void RestoreAccurateRhs(__gm__ aclsolverFloatComplex* b,
                                               const __gm__ aclsolverFloatComplex* saved,
                                               __gm__ aclsolverFloatComplex* tail, int n)
{
    for (int row = threadIdx.x; row < n; row += THREADS)
    {
        WriteLower(b, n, row, 0, false, ReadLower(saved, n, row, 0, false));
        WriteLower(tail, n, row, 0, false, {0, 0});
    }
    asc_syncthreads();
}

template <bool PER_RHS = false>
__simt_vf__ __aicore__ __launch_bounds__(THREADS) void AccurateSolveVF(
    const __gm__ aclsolverFloatComplex* singleA, __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* singleB,
    __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb, bool upper, int batch,
    const __gm__ aclsolverFloatComplex* backup, __gm__ aclsolverFloatComplex* low, const __gm__ int* flags,
    const __gm__ int* check, __ubuf__ aclsolverFloatComplex* pivot)
{
    if (check && *check != 0)
    {
        return;
    }
    for (int64_t task = blockIdx.x; task < static_cast<int64_t>(batch) * nrhs; task += gridDim.x)
    {
        int m = task / nrhs, rhs = task % nrhs;
        int accurate = 0;
        if constexpr (PER_RHS)
        {
            accurate = flags[task];
        }
        else
        {
            int groups = (n + WARP_SIZE - 1) / WARP_SIZE;
            for (int group = 0; group < groups && !accurate; ++group)
            {
                accurate = flags[static_cast<int64_t>(m) * groups + group];
            }
        }
        if (!accurate)
        {
            continue;
        }
        auto a = aarray ? reinterpret_cast<const __gm__ aclsolverFloatComplex*>(aarray[m]) : singleA;
        auto b = (barray ? reinterpret_cast<__gm__ aclsolverFloatComplex*>(barray[m]) : singleB) +
                 static_cast<int64_t>(rhs) * ldb;
        auto saved = backup + task * n;
        auto tail = low + task * n;
        RestoreAccurateRhs(b, saved, tail, n);
        for (int pass = 0; pass < 2; ++pass)
        {
            for (int step = 0; step < n; ++step)
            {
                AccurateGlobalStep(a, b, tail, n, lda, upper, pass, step, pivot);
            }
        }
        asc_syncthreads();
    }
}
__aicore__ inline void DispatchFusedSolve(const __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* aarray,
                                          __gm__ aclsolverFloatComplex* b, __gm__ const uint64_t* barray, int n,
                                          int nrhs, int lda, int ldb, bool upper, int batch,
                                          __gm__ aclsolverFloatComplex* backup, __gm__ int* flags)
{
    if (n <= WARP_SIZE)
    {
        RunFusedSolve<1, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 2 * WARP_SIZE)
    {
        RunFusedSolve<2, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 3 * WARP_SIZE)
    {
        RunFusedSolve<3, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= REGISTER_SOLVE_ORDER)
    {
        RunFusedSolve<4, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 6 * WARP_SIZE)
    {
        RunFusedSolve<6, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 8 * WARP_SIZE)
    {
        RunFusedSolve<8, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 10 * WARP_SIZE)
    {
        RunFusedSolve<10, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 12 * WARP_SIZE)
    {
        RunFusedSolve<12, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else if (n <= 14 * WARP_SIZE)
    {
        RunFusedSolve<14, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
    }
    else
    {
        RunFusedSolve<FUSED_REGISTER_ORDER / WARP_SIZE, SOLVE_THREADS>(a, aarray, b, barray, n, nrhs, lda, ldb, upper,
                                                                       batch, backup, flags);
    }
}

__aicore__ inline void DispatchSmallAccurateSolve(const __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* aarray,
                                                  __gm__ aclsolverFloatComplex* b, __gm__ const uint64_t* barray, int n,
                                                  int nrhs, int lda, int ldb, bool upper, int batch,
                                                  __gm__ aclsolverFloatComplex* backup, __gm__ int* flags, bool assess)
{
    if (n <= MIN_SUBGROUP_LANES)
    {
        asc_vf_call<SmallAccurateSolveVF<MIN_SUBGROUP_LANES>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb,
                                                              upper, batch, backup, flags, assess);
    }
    else if (n <= MEDIUM_SUBGROUP_LANES)
    {
        asc_vf_call<SmallAccurateSolveVF<MEDIUM_SUBGROUP_LANES>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb,
                                                                 upper, batch, backup, flags, assess);
    }
    else if (n <= HALF_WARP_LANES)
    {
        asc_vf_call<SmallAccurateSolveVF<HALF_WARP_LANES>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb,
                                                           upper, batch, backup, flags, assess);
    }
    else if (n <= WARP_SIZE)
    {
        asc_vf_call<SmallAccurateSolveVF<WARP_SIZE>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper,
                                                     batch, backup, flags, assess);
    }
    else if (n <= 2 * WARP_SIZE)
    {
        asc_vf_call<SmallAccurateSolveVF<WARP_SIZE, 2>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper,
                                                        batch, backup, flags, assess);
    }
    else
    {
        asc_vf_call<SmallAccurateSolveVF<WARP_SIZE, REGISTER_SOLVE_ORDER / WARP_SIZE>>(
            dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags, assess);
    }
    return;
}

__global__ __aicore__ __vector__ void AccurateSolveKernel(
    const __gm__ aclsolverFloatComplex* a, __gm__ const uint64_t* aarray, __gm__ aclsolverFloatComplex* b,
    __gm__ const uint64_t* barray, int n, int nrhs, int lda, int ldb, bool upper, int batch,
    __gm__ aclsolverFloatComplex* backup, __gm__ aclsolverFloatComplex* low, __gm__ int* flags, const __gm__ int* check,
    bool assess, bool direct)
{
    if (direct)
    {
        if (n > FUSED_REGISTER_ORDER)
        {
            AscendC::TPipe pipe;
            AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
            pipe.InitBuffer(buffer, 2 * DMA_ALIGNMENT_BYTES);
            auto pivot = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(buffer.Get<uint8_t>().GetPhyAddr());
            asc_vf_call<AccurateSolveVF<true>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch,
                                               backup, low, flags, check, pivot);
            return;
        }
        DispatchFusedSolve(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags);
        return;
    }
    if (n <= REGISTER_SOLVE_ORDER && batch >= BATCH_THRESHOLD)
    {
        DispatchSmallAccurateSolve(a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags, assess);
        return;
    }
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, THREADS * sizeof(aclsolverFloatComplex));
    auto shared = reinterpret_cast<__ubuf__ aclsolverFloatComplex*>(buffer.Get<uint8_t>().GetPhyAddr());
    if (assess)
    {
        asc_vf_call<AssessSolveVF>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch, backup, flags,
                                   check, shared);
    }
    else
    {
        asc_vf_call<AccurateSolveVF<false>>(dim3(THREADS), a, aarray, b, barray, n, nrhs, lda, ldb, upper, batch,
                                            backup, low, flags, check, shared);
    }
}

void FactorDiagonalTile(aclrtStream stream, aclsolverFloatComplex* a, aclsolverFloatComplex** array, int width, int lda,
                        bool upper, int* info, int batch, int offset, aclsolverFloatComplex* firstColumn, int capacity,
                        float* originalDiagonal, int fullOrder)
{
    if (width <= REGISTER_FACTOR_ORDER)
    {
        LaunchSmallFactor(stream, false, VectorBlocks((batch + THREADS / WARP_SIZE - 1) / (THREADS / WARP_SIZE)), a,
                          reinterpret_cast<const uint64_t*>(array), width, lda, upper, info, batch, offset, firstColumn,
                          capacity, originalDiagonal, fullOrder);
    }
    else
    {
        LaunchSmallFactor(stream, true, VectorBlocks((batch + WHOLE_MATRICES_PER_AIV - 1) / WHOLE_MATRICES_PER_AIV), a,
                          reinterpret_cast<const uint64_t*>(array), width, lda, upper, info, batch, offset, firstColumn,
                          capacity, originalDiagonal, fullOrder);
    }
}

void BlockedFactor(aclrtStream stream, bool upper, int n, aclsolverFloatComplex* a, aclsolverFloatComplex** array,
                   int lda, int* info, int batch, aclsolverFloatComplex* work)
{
    int capacity = (n + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT;
    int matrices = VectorBlocks(batch);
    if (matrices == 0)
    {
        return;
    }
    int64_t packedStride = static_cast<int64_t>(capacity) * GRAM_COMPONENT_ROWS * batch;
    float *real = reinterpret_cast<float*>(work), *imag = real + packedStride;
    auto firstColumn = reinterpret_cast<aclsolverFloatComplex*>(imag + packedStride);
    auto originalDiagonal = reinterpret_cast<float*>(firstColumn + static_cast<int64_t>(capacity) * batch);
    for (int offset = 0; offset < n; offset += FACTOR_BLOCK)
    {
        int width = n - offset < FACTOR_BLOCK ? n - offset : FACTOR_BLOCK;
        FactorDiagonalTile(stream, a, array, width, lda, upper, info, batch, offset, firstColumn, capacity,
                           originalDiagonal, n);
        int rows = n - offset - width;
        if (!rows)
        {
            continue;
        }
        int groups = VectorBlocks(MAX_BATCH) / matrices;
        int panelTiles = (rows + FACTOR_BLOCK - 1) / FACTOR_BLOCK;
        groups = groups < panelTiles ? groups : panelTiles;
        LaunchPanel(stream, false, matrices * groups,
                    (batch >= BATCH_THRESHOLD && rows <= PACKED_TAIL_ROWS ? PACKED_PANEL_BYTES : SIMD_PANEL_BYTES), a,
                    reinterpret_cast<const uint64_t*>(array), real, n, lda, upper, offset, width, info, batch, groups);
        bool fused = batch >= BATCH_THRESHOLD && rows <= PACKED_TAIL_ROWS;
        if (!offset && upper && !fused)
        {
            int rowTiles = (rows + WARP_SIZE - 1) / WARP_SIZE, tiles = rowTiles * (rowTiles + 1) / 2;
            int available = VectorBlocks(MAX_BATCH) / matrices;
            int applyGroups = available < tiles ? available : tiles;
            LaunchPanel(stream, true, matrices * applyGroups, 0, a, reinterpret_cast<const uint64_t*>(array), real, n,
                        lda, upper, offset, width, info, batch, applyGroups);
        }
        if (!fused)
        {
            Schur(stream, (rows + CUBE_ALIGNMENT - 1) / CUBE_ALIGNMENT * CUBE_ALIGNMENT, real, imag, capacity, batch,
                  upper, n, a, array, lda, offset + width, info);
        }
    }
}
void CheckedSolve(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                  const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b,
                  aclsolverFloatComplex** barray, int ldb, int batch, aclsolverFloatComplex* work, const int* check)
{
    auto backup = work + static_cast<int64_t>(CoreSolveWorkspace(n, nrhs, batch)) * batch;
    auto low = backup + static_cast<int64_t>(n) * nrhs * batch;
    auto flags = reinterpret_cast<int*>(low + static_cast<int64_t>(n) * nrhs * batch);
    auto pointersA = reinterpret_cast<const uint64_t*>(aarray), pointersB = reinterpret_cast<const uint64_t*>(barray);
    // Inspect coefficients during the ordinary solve; only difficult factors need a compensated pass.
    if (UseBufferedSolve(n, nrhs, batch))
    {
        int padded = (n + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
        int warps = BufferedWarps(n);
        int threads = n <= FUSED_REGISTER_ORDER ? SOLVE_THREADS : warps * WARP_SIZE;
        if (n > FUSED_REGISTER_ORDER)
        {
            SolveKernel<true>
                <<<VectorBlocks(static_cast<int64_t>(batch) * nrhs), warps * padded * COMPLEX_BYTES, stream>>>(
                    a, pointersA, b, pointersB, n, nrhs, lda, ldb, upper, batch, check, flags, backup);
        }
        AccurateSolveKernel<<<VectorBlocks(n <= FUSED_REGISTER_ORDER
                                               ? (static_cast<int64_t>(batch) * nrhs + threads / WARP_SIZE - 1) /
                                                     (threads / WARP_SIZE)
                                               : static_cast<int64_t>(batch) * nrhs),
                              nullptr, stream>>>(a, pointersA, b, pointersB, n, nrhs, lda, ldb, upper, batch, backup,
                                                 low, flags, check, false, true);
        return;
    }
    AccurateSolveKernel<<<VectorBlocks(static_cast<int64_t>(batch) * ((n + WARP_SIZE - 1) / WARP_SIZE)), nullptr,
                          stream>>>(a, pointersA, b, pointersB, n, nrhs, lda, ldb, upper, batch, backup, low, flags,
                                    check, true, false);
    if (UseNativeSolve(n, nrhs, batch))
    {
        NativeSolve(stream, upper, n, nrhs, a, aarray, lda, b, barray, ldb, batch, work, check);
    }
    else
    {
        int threads = n <= REGISTER_SOLVE_ORDER ? SOLVE_THREADS : LARGE_SOLVE_THREADS;
        SolveKernel<false>
            <<<VectorBlocks((static_cast<int64_t>(batch) * nrhs + threads / WARP_SIZE - 1) / (threads / WARP_SIZE)),
               nullptr, stream>>>(a, pointersA, b, pointersB, n, nrhs, lda, ldb, upper, batch, check, flags, nullptr);
    }
    AccurateSolveKernel<<<VectorBlocks(static_cast<int64_t>(batch) * nrhs), nullptr, stream>>>(
        a, pointersA, b, pointersB, n, nrhs, lda, ldb, upper, batch, backup, low, flags, check, false, false);
}
}  // namespace

bool UseBufferedSolve(int n, int nrhs, int batch)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    int vectorCores = platform ? platform->GetCoreNumAiv() : 0;
    return BufferedSolvePolicy(n, nrhs, batch, vectorCores);
}

aclsolverStatus_t Finish(aclsolverHandle_t handle, int* info, int argument)
{
    if (!handle)
    {
        return ReportError(ACLSOLVER_STATUS_NOT_INITIALIZED, "Cholesky handle is null");
    }
    if (!info)
    {
        return ReportError(ACLSOLVER_STATUS_INVALID_VALUE, "Cholesky info is null");
    }
    aclrtStream stream = nullptr;
    auto status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return ReportError(status, "cannot obtain Cholesky stream");
    }
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint64_t ubBytes = 0;
    if (platform)
    {
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    }
    if (!platform || platform->GetCoreNumAiv() <= 0 || platform->GetCoreNumAic() <= 0 ||
        ubBytes < VECTOR_KERNEL_UB_BYTES)
    {
        return ReportError(ACLSOLVER_STATUS_ARCH_MISMATCH, "Cholesky requires available Ascend950 AIV/AIC and UB");
    }
    InfoKernel<<<1, nullptr, stream>>>(info, -argument);
    return argument ? ReportError(ACLSOLVER_STATUS_INVALID_VALUE, "invalid Cholesky argument; inspect device info")
                    : ACLSOLVER_STATUS_SUCCESS;
}
void Factor(aclrtStream stream, bool upper, int n, aclsolverFloatComplex* a, aclsolverFloatComplex** array, int lda,
            int* info, int batch, aclsolverFloatComplex* work)
{
    if (batch >= BATCH_THRESHOLD && n > REGISTER_FACTOR_ORDER && n <= FACTOR_BLOCK)
    {
        LaunchSmallFactor(stream, true, VectorBlocks((batch + WHOLE_MATRICES_PER_AIV - 1) / WHOLE_MATRICES_PER_AIV), a,
                          reinterpret_cast<const uint64_t*>(array), n, lda, upper, info, batch, 0, nullptr, 0);
        return;
    }
    if (n > REGISTER_FACTOR_ORDER)
    {
        BlockedFactor(stream, upper, n, a, array, lda, info, batch, work);
        return;
    }
    LaunchSmallFactor(stream, false,
                      VectorBlocks((batch + BATCH_FACTOR_THREADS / WARP_SIZE - 1) / (BATCH_FACTOR_THREADS / WARP_SIZE)),
                      a, reinterpret_cast<const uint64_t*>(array), n, lda, upper, info, batch, 0, nullptr, 0);
}
void Solve(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
           const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b,
           aclsolverFloatComplex** barray, int ldb, int batch, aclsolverFloatComplex* work)
{
    CheckedSolve(stream, upper, n, nrhs, a, aarray, lda, b, barray, ldb, batch, work, nullptr);
}
void Inverse(aclrtStream stream, bool upper, int n, aclsolverFloatComplex* a, int lda, aclsolverFloatComplex* work,
             int* info)
{
    InverseKernel<false>
        <<<VectorBlocks(static_cast<int64_t>(n) * n / THREADS + 1), nullptr, stream>>>(a, work, n, lda, upper, info);
    CheckedSolve(stream, upper, n, n, a, nullptr, lda, work, nullptr, n, 1, work + static_cast<int64_t>(n) * n, info);
    InverseKernel<true>
        <<<VectorBlocks(static_cast<int64_t>(n) * n / THREADS + 1), nullptr, stream>>>(a, work, n, lda, upper, info);
}

}  // namespace Cholesky
