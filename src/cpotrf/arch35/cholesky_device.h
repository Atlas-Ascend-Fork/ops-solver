/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once
#include "../../utils/cholesky/cholesky.h"
#include "../../utils/cholesky/cholesky_update.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_functions.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/vector_functions.h"
#include "tiling/platform/platform_ascendc.h"

namespace Cholesky
{
namespace
{
constexpr int THREADS = 256;
constexpr int PANEL_THREADS = 1024;
constexpr int SOLVE_THREADS = 512;
constexpr int LARGE_SOLVE_THREADS = 256;
constexpr int INVERSE_THREADS = 512;
constexpr int BATCH_FACTOR_THREADS = 512;
constexpr int PANEL_IO_THREADS = 512;
constexpr int RHS_WARPS = THREADS / WARP_SIZE;
int VectorBlocks(int64_t tasks)
{
    int cores = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    return tasks < cores ? static_cast<int>(tasks) : cores;
}

// Reuse the distributed panel data when only a few matrices are assigned to each AIV.
inline bool UseFactorDiagonalState(int batch) { return batch <= MIN_BATCHES_PER_CORE * VectorBlocks(MAX_BATCH); }

// Both storage modes are viewed as a lower factor; upper entries are conjugated.
__simt_callee__ inline aclsolverFloatComplex ReadLower(const __gm__ aclsolverFloatComplex* a, int lda, int row, int col,
                                                       bool upper, bool cached = false)
{
    int64_t index = upper ? static_cast<int64_t>(row) * lda + col : static_cast<int64_t>(col) * lda + row;
    auto p = reinterpret_cast<__gm__ float2*>(const_cast<__gm__ aclsolverFloatComplex*>(a) + index);
    auto v = cached ? asc_ldca(p) : asc_ldcg(p);
    aclsolverFloatComplex z{v.x, v.y};
    if (upper)
    {
        z.imag = -z.imag;
    }
    return z;
}
__simt_callee__ inline void WriteLower(__gm__ aclsolverFloatComplex* a, int lda, int row, int col, bool upper,
                                       aclsolverFloatComplex z)
{
    if (upper)
    {
        z.imag = -z.imag;
    }
    int64_t index = upper ? static_cast<int64_t>(row) * lda + col : static_cast<int64_t>(col) * lda + row;
    asc_stcg(reinterpret_cast<__gm__ float2*>(a + index), make_float2(z.real, z.imag));
}
__simt_callee__ inline aclsolverFloatComplex ReadUB(const __ubuf__ aclsolverFloatComplex* p, int index)
{
    return {p[index].real, p[index].imag};
}
__simt_callee__ inline void WriteUB(__ubuf__ aclsolverFloatComplex* p, int index, aclsolverFloatComplex z)
{
    p[index].real = z.real;
    p[index].imag = z.imag;
}

__simt_callee__ inline void AddDiagonalProduct(FloatPair& diagonal, float left, float right)
{
    float product = left * right;
    auto sum = PairSum(diagonal.high, product);
    diagonal.low += sum.low + fmaf(left, right, -product);
    diagonal.high = sum.high;
}

__simt_callee__ inline void UpdateScalarDiagonal(FloatPair& diagonal, aclsolverFloatComplex value,
                                                 aclsolverFloatComplex low, bool firstColumn)
{
    if (!firstColumn)
    {
        float energy = fmaf(value.imag, value.imag, value.real * value.real);
        float change = diagonal.low - energy, sum = diagonal.high + change;
        diagonal.low = change - (sum - diagonal.high);
        diagonal.high = sum;
        return;
    }
    AddDiagonalProduct(diagonal, -value.real, value.real);
    AddDiagonalProduct(diagonal, -value.imag, value.imag);
    AddDiagonalProduct(diagonal, -(value.real + value.real), low.real);
    AddDiagonalProduct(diagonal, -(value.imag + value.imag), low.imag);
    AddDiagonalProduct(diagonal, -low.real, low.real);
    AddDiagonalProduct(diagonal, -low.imag, low.imag);
}

}  // namespace
void NativeSolve(aclrtStream stream, bool upper, int n, int nrhs, const aclsolverFloatComplex* a,
                 const aclsolverFloatComplex* const* aarray, int lda, aclsolverFloatComplex* b,
                 aclsolverFloatComplex** barray, int ldb, int batch, aclsolverFloatComplex* work, const int* check);
void LaunchSmallFactor(aclrtStream stream, bool whole, int blocks, aclsolverFloatComplex* a, const uint64_t* array,
                       int n, int lda, bool upper, int* info, int batch, int offset, aclsolverFloatComplex* firstColumn,
                       int firstStride, float* originalDiagonal = nullptr, int fullOrder = 0);
void LaunchPanel(aclrtStream stream, bool first, int blocks, size_t sharedBytes, aclsolverFloatComplex* a,
                 const uint64_t* array, float* work, int n, int lda, bool upper, int offset, int width, int* info,
                 int batch, int groups);
}  // namespace Cholesky
