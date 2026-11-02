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

#include "cann_ops_solver_common.h"
#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"
#include "simt_api/device_warp_functions.h"
#include "simt_api/math_functions.h"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
constexpr uint32_t THREADS = 256;
constexpr int PANEL = 32;

template <typename Pointer>
__simt_callee__ __aicore__ inline aclFloatComplex Load(Pointer pointer, uint64_t index)
{
    return {pointer[index].real, pointer[index].imag};
}

template <typename Pointer>
__simt_callee__ __aicore__ inline void Store(Pointer pointer, uint64_t index, aclFloatComplex value)
{
    pointer[index].real = value.real;
    pointer[index].imag = value.imag;
}

__simt_callee__ __aicore__ inline float Abs(float x) { return x < 0.0f ? -x : x; }

__simt_callee__ __aicore__ inline aclFloatComplex SubMul(aclFloatComplex x, aclFloatComplex a, aclFloatComplex b)
{
    return {fmaf(a.imag, b.imag, fmaf(-a.real, b.real, x.real)), fmaf(-a.imag, b.real, fmaf(-a.real, b.imag, x.imag))};
}

__simt_callee__ __aicore__ inline float Quotient(float ar, float br, float ai, float bi, float denominator,
                                                 float denominatorError)
{
    float product = ar * br;
    float numerator = fmaf(ai, bi, product);
    float error = fmaf(ar, br, -product) + fmaf(ai, bi, product - numerator);
    float reciprocal = 1.0f / denominator;
    float result = numerator * reciprocal;
    float remainder = fmaf(-result, denominator, numerator) + error - result * denominatorError;
    return fmaf(remainder, reciprocal, result);
}

__simt_callee__ __aicore__ inline aclFloatComplex Divide(aclFloatComplex a, aclFloatComplex b)
{
    union
    {
        float value;
        uint32_t bits;
    } scale;
    scale.value = Abs(b.real) > Abs(b.imag) ? Abs(b.real) : Abs(b.imag);
    uint32_t exponent = scale.bits & 0x7f800000U;
    if (exponent != 0) scale.bits = exponent;
    float br = b.real / scale.value;
    float bi = b.imag / scale.value;
    float ar = a.real / scale.value;
    float ai = a.imag / scale.value;
    float first = Abs(br) > Abs(bi) ? br : bi;
    float second = Abs(br) > Abs(bi) ? bi : br;
    float square = first * first;
    float denominator = fmaf(second, second, square);
    float error = fmaf(first, first, -square) + fmaf(second, second, square - denominator);
    return {Quotient(ar, br, ai, bi, denominator, error), Quotient(ai, br, -ar, bi, denominator, error)};
}

__simt_callee__ __aicore__ inline bool Zero(aclFloatComplex a) { return a.real == 0.0f && a.imag == 0.0f; }

__simt_callee__ __aicore__ inline __gm__ aclFloatComplex* MatrixPointer(__gm__ uint8_t* array, uint32_t batch)
{
    uint64_t address = reinterpret_cast<__gm__ const uint64_t*>(array)[batch];
    return reinterpret_cast<__gm__ aclFloatComplex*>(address);
}

__simt_callee__ __aicore__ inline void ReducePivot(float& score, int& row)
{
    for (int offset = 16; offset > 0; offset /= 2)
    {
        float other = asc_shfl_down(score, offset);
        int otherRow = asc_shfl_down(row, offset);
        if ((threadIdx.x % 32) + offset < 32 && (other > score || (other == score && otherRow < row)))
        {
            score = other;
            row = otherRow;
        }
    }
}

template <typename MatrixPointer>
__simt_callee__ __aicore__ inline int FindPivot(MatrixPointer a, int m, int lda, int col, __ubuf__ float* scores,
                                                __ubuf__ int* rows)
{
    float best = -1.0f;
    int row = m;
    for (int r = col + threadIdx.x; r < m; r += blockDim.x)
    {
        auto value = Load(a, static_cast<uint64_t>(col) * lda + r);
        float score = Abs(value.real) + Abs(value.imag);
        if (score > best || (score == best && r < row))
        {
            best = score;
            row = r;
        }
    }
    ReducePivot(best, row);
    if (threadIdx.x % 32 == 0)
    {
        scores[threadIdx.x / 32] = best;
        rows[threadIdx.x / 32] = row;
    }
    asc_syncthreads();
    if (threadIdx.x < 32)
    {
        best = threadIdx.x < blockDim.x / 32 ? scores[threadIdx.x] : -1.0f;
        row = threadIdx.x < blockDim.x / 32 ? rows[threadIdx.x] : m;
        ReducePivot(best, row);
        if (threadIdx.x == 0)
        {
            scores[0] = best;
            rows[0] = row;
        }
    }
    asc_syncthreads();
    return rows[0];
}

template <typename MatrixPointer>
__simt_callee__ __aicore__ inline void SwapRows(MatrixPointer a, int n, int lda, int left, int right)
{
    if (left != right)
    {
        for (int col = threadIdx.x; col < n; col += blockDim.x)
        {
            uint64_t base = static_cast<uint64_t>(col) * lda;
            aclFloatComplex value = Load(a, base + left);
            Store(a, base + left, Load(a, base + right));
            Store(a, base + right, value);
        }
    }
    asc_syncthreads();
}

template <typename MatrixPointer>
__simt_callee__ __aicore__ inline void DivideColumn(MatrixPointer a, int m, int lda, int col)
{
    aclFloatComplex diagonal = Load(a, static_cast<uint64_t>(col) * lda + col);
    if (!Zero(diagonal))
    {
        for (int row = col + 1 + threadIdx.x; row < m; row += blockDim.x)
        {
            uint64_t index = static_cast<uint64_t>(col) * lda + row;
            Store(a, index, Divide(Load(a, index), diagonal));
        }
    }
    asc_syncthreads();
}

template <typename MatrixPointer>
__simt_callee__ __aicore__ inline void UpdatePanel(MatrixPointer a, int m, int lda, int col, int end)
{
    int rows = m - col - 1;
    int count = rows * (end - col - 1);
    for (int index = threadIdx.x; index < count; index += blockDim.x)
    {
        int row = col + 1 + index % rows;
        int column = col + 1 + index / rows;
        uint64_t offset = static_cast<uint64_t>(column) * lda + row;
        Store(a, offset,
              SubMul(Load(a, offset), Load(a, static_cast<uint64_t>(col) * lda + row),
                     Load(a, static_cast<uint64_t>(column) * lda + col)));
    }
    asc_syncthreads();
}

template <typename MatrixPointer, typename PivotPointer, typename InfoPointer>
__simt_callee__ __aicore__ inline void FactorColumn(MatrixPointer a, int m, int n, int lda, int col, int end,
                                                    PivotPointer pivots, InfoPointer info, __ubuf__ float* scores,
                                                    __ubuf__ int* rows)
{
    int pivot = pivots == nullptr ? col : FindPivot(a, m, lda, col, scores, rows);
    if (threadIdx.x == 0)
    {
        if (pivots != nullptr) pivots[col] = pivot + 1;
        if (*info == 0 && Zero(Load(a, static_cast<uint64_t>(col) * lda + pivot))) *info = col + 1;
    }
    SwapRows(a, n, lda, col, pivot);
    DivideColumn(a, m, lda, col);
    UpdatePanel(a, m, lda, col, end);
}

template <typename MatrixPointer, typename PivotPointer, typename InfoPointer>
__simt_callee__ __aicore__ inline void FactorPanel(MatrixPointer a, int m, int n, int lda, int begin, int end,
                                                   PivotPointer pivots, InfoPointer info, __ubuf__ float* scores,
                                                   __ubuf__ int* rows)
{
    for (int col = begin; col < end; ++col) FactorColumn(a, m, n, lda, col, end, pivots, info, scores, rows);
}

template <typename MatrixPointer>
__simt_callee__ __aicore__ inline int CheckLu(MatrixPointer a, int n, int lda, __gm__ const int* pivots)
{
    if (pivots != nullptr)
        for (int row = 0; row < n; ++row)
            if (pivots[row] < row + 1 || pivots[row] > n) return -4;
    for (int row = 0; row < n; ++row)
        if (Zero(Load(a, static_cast<uint64_t>(row) * lda + row))) return row + 1;
    return 0;
}

template <typename MatrixPointer>
__simt_callee__ __aicore__ inline int ParallelCheckLu(MatrixPointer a, int n, int lda, __gm__ const int* pivots,
                                                      __ubuf__ int* groups)
{
    int result = 0x7fffffff;
    for (int row = threadIdx.x; row < n; row += blockDim.x)
    {
        if (pivots != nullptr && (pivots[row] < row + 1 || pivots[row] > n)) result = -4;
        if (Zero(Load(a, static_cast<uint64_t>(row) * lda + row)) && result > row + 1) result = row + 1;
    }
    for (int offset = 16; offset > 0; offset /= 2)
    {
        int other = asc_shfl_down(result, offset);
        if (threadIdx.x % 32 + offset < 32 && other < result) result = other;
    }
    if (threadIdx.x % 32 == 0) groups[threadIdx.x / 32] = result;
    asc_syncthreads();
    if (threadIdx.x < 32)
    {
        result = threadIdx.x < blockDim.x / 32 ? groups[threadIdx.x] : 0x7fffffff;
        for (int offset = 16; offset > 0; offset /= 2)
        {
            int other = asc_shfl_down(result, offset);
            if (threadIdx.x + offset < 32 && other < result) result = other;
        }
        if (threadIdx.x == 0) groups[0] = result == 0x7fffffff ? 0 : result;
    }
    asc_syncthreads();
    return groups[0];
}

template <typename PivotPointer>
__simt_callee__ __aicore__ inline void IdentityRhs(__ubuf__ aclFloatComplex* rhs, int n, int firstColumn, int columns,
                                                   PivotPointer pivots)
{
    for (int index = threadIdx.x; index < n * columns; index += blockDim.x)
        Store(rhs, index, {index % n == firstColumn + index / n ? 1.0f : 0.0f, 0.0f});
    asc_syncthreads();
    // One thread owns each RHS column throughout the permutation sequence.
    if (pivots != nullptr && threadIdx.x < columns)
    {
        int base = threadIdx.x * n;
        for (int row = 0; row < n; ++row)
        {
            int pivot = pivots[row] - 1;
            aclFloatComplex value = Load(rhs, base + row);
            Store(rhs, base + row, Load(rhs, base + pivot));
            Store(rhs, base + pivot, value);
        }
    }
    asc_syncthreads();
}

__simt_callee__ __aicore__ inline void StoreRhs(__ubuf__ const aclFloatComplex* rhs, int n, int firstColumn,
                                                int columns, __gm__ aclFloatComplex* c, int ldc)
{
    for (int index = threadIdx.x; index < n * columns; index += blockDim.x)
    {
        int row = index % n;
        int column = firstColumn + index / n;
        Store(c, static_cast<uint64_t>(column) * ldc + row, Load(rhs, index));
    }
    asc_syncthreads();
}
}  // namespace ComplexSolverKernel
#endif
