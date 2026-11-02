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
#include "complex_solver_resident.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int MaxOrder>
__simt_callee__ __aicore__ inline void CooperativeUpper(__ubuf__ aclFloatComplex* matrix, int n,
                                                        __ubuf__ aclFloatComplex* factors,
                                                        __ubuf__ aclFloatComplex* upper)
{
    constexpr int PITCH = MaxOrder + 1;
    for (int pivot = 0; pivot < n; ++pivot)
    {
        auto divisor = PrepareDivisor(Load(matrix, pivot * PITCH + pivot));
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            Store(factors, i, i < pivot ? Load(matrix, pivot * PITCH + i) : aclFloatComplex{0, 0});
            if (i > pivot) Store(upper, i, PreparedDivide(Load(matrix, i * PITCH + pivot), divisor));
            if (i == pivot) Store(upper, i, PreparedDivide({1, 0}, divisor));
        }
        asc_syncthreads();
        for (int index = threadIdx.x; index < pivot * (n - pivot - 1); index += blockDim.x)
        {
            int row = index % pivot;
            int column = pivot + 1 + index / pivot;
            Store(matrix, column * PITCH + row,
                  SubMul(Load(matrix, column * PITCH + row), Load(factors, row), Load(upper, column)));
        }
        for (int i = threadIdx.x; i < n; i += blockDim.x)
        {
            if (i >= pivot) Store(matrix, i * PITCH + pivot, Load(upper, i));
            if (i < pivot) Store(matrix, pivot * PITCH + i, SubMul({0, 0}, Load(factors, i), Load(upper, pivot)));
        }
        asc_syncthreads();
    }
}

template <int MaxOrder>
__simt_callee__ __aicore__ inline void CooperativeLower(__ubuf__ aclFloatComplex* matrix, int n)
{
    constexpr int PITCH = MaxOrder + 1;
    for (int column = n - 1; column >= 0; --column)
    {
        int row = threadIdx.x;
        auto value = row <= column ? Load(matrix, column * PITCH + row) : aclFloatComplex{0, 0};
        if (row < n)
            for (int k = column + 1; k < n; ++k)
                value = SubMul(value, Load(matrix, k * PITCH + row), Load(matrix, column * PITCH + k));
        asc_syncthreads();
        if (row < n) Store(matrix, column * PITCH + row, value);
        asc_syncthreads();
    }
}

template <int MaxOrder>
__simt_callee__ __aicore__ inline void StoreCooperativeInverse(ComplexSolverShape shape, __gm__ aclFloatComplex* output,
                                                               __gm__ const int* pivot,
                                                               __ubuf__ const aclFloatComplex* matrix,
                                                               __ubuf__ int* targets)
{
    for (int column = threadIdx.x; column < shape.n; column += blockDim.x)
    {
        int target = column;
        if (pivot != nullptr)
            for (int row = shape.n - 1; row >= 0; --row)
            {
                int swapped = pivot[row] - 1;
                target = target == row ? swapped : target == swapped ? row : target;
            }
        targets[column] = target;
    }
    asc_syncthreads();
    for (int index = threadIdx.x; index < shape.n * shape.n; index += blockDim.x)
    {
        int row = index % shape.n;
        int column = index / shape.n;
        Store(output, static_cast<uint64_t>(targets[column]) * shape.ldc + row,
              Load(matrix, column * (MaxOrder + 1) + row));
    }
    asc_syncthreads();
}

template <int MaxOrder>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void CooperativeInverseVf(ComplexSolverShape shape,
                                                                              __gm__ uint8_t* a,
                                                                              __gm__ const int* pivots,
                                                                              __gm__ uint8_t* c, __gm__ int* info)
{
    __ubuf__ float storage[2 * MaxOrder * (MaxOrder + 1)];
    __ubuf__ float rowStorage[4 * MaxOrder];
    __ubuf__ int targets[MaxOrder];
    __ubuf__ int groups[THREADS / 32];
    auto* matrix = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    auto* factors = reinterpret_cast<__ubuf__ aclFloatComplex*>(rowStorage);
    for (int batch = blockIdx.x; batch < shape.batch; batch += gridDim.x)
    {
        auto* input = MatrixPointer(a, batch);
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        for (int index = threadIdx.x; index < shape.n * shape.n; index += blockDim.x)
            Store(matrix, (index / shape.n) * (MaxOrder + 1) + index % shape.n,
                  Load(input, static_cast<uint64_t>(index / shape.n) * shape.lda + index % shape.n));
        int code = ParallelCheckLu(input, shape.n, shape.lda, pivot, groups);
        if (threadIdx.x == 0) info[batch] = code;
        if (code == 0)
        {
            CooperativeUpper<MaxOrder>(matrix, shape.n, factors, factors + MaxOrder);
            CooperativeLower<MaxOrder>(matrix, shape.n);
            StoreCooperativeInverse<MaxOrder>(shape, MatrixPointer(c, batch), pivot, matrix, targets);
        }
        asc_syncthreads();
    }
}

template <bool Forward>
__simt_callee__ __aicore__ inline void InverseRowTilePhase(ComplexSolverShape shape,
                                                           __gm__ const aclFloatComplex* input,
                                                           __ubuf__ aclFloatComplex* matrix,
                                                           __ubuf__ aclFloatComplex* factors, int first)
{
    constexpr int ROWS = 64;
    int row = first + threadIdx.x;
    for (int step = 0; step < shape.n; ++step)
    {
        int column = Forward ? step : shape.n - 1 - step;
        for (int k = threadIdx.x; k < shape.n; k += blockDim.x)
            Store(factors, k, Load(input, static_cast<uint64_t>(column) * shape.lda + k));
        asc_syncthreads();
        if (row < shape.n)
        {
            aclFloatComplex value{0, 0};
            if constexpr (Forward)
            {
                for (int k = row; k < column; ++k)
                    value = SubMul(value, Load(matrix, k * (ROWS + 1) + threadIdx.x), Load(factors, k));
                value = row <= column ? PreparedDivide(row == column ? aclFloatComplex{1, 0} : value,
                                                       PrepareDivisor(Load(factors, column)))
                                      : value;
            }
            else
            {
                value = Load(matrix, column * (ROWS + 1) + threadIdx.x);
                for (int k = column + 1; k < shape.n; ++k)
                    value = SubMul(value, Load(matrix, k * (ROWS + 1) + threadIdx.x), Load(factors, k));
            }
            Store(matrix, column * (ROWS + 1) + threadIdx.x, value);
        }
        asc_syncthreads();
    }
}

__simt_callee__ __aicore__ inline int RowTileColumnTarget(__gm__ const int* pivot, int n, int column)
{
    if (pivot == nullptr) return column;
    int target = column;
    for (int k = n - 1; k >= 0; --k)
    {
        int swapped = pivot[k] - 1;
        target = target == k ? swapped : target == swapped ? k : target;
    }
    return target;
}

__simt_callee__ __aicore__ inline void StoreInverseRowTile(ComplexSolverShape shape, __gm__ uint8_t* c,
                                                           __gm__ const int* pivot, int batch, int first,
                                                           __ubuf__ const aclFloatComplex* matrix)
{
    auto* output = MatrixPointer(c, batch);
    for (int column = 0; column < shape.n; ++column)
    {
        int target = RowTileColumnTarget(pivot, shape.n, column);
        if (first + threadIdx.x < shape.n)
            Store(output, static_cast<uint64_t>(target) * shape.ldc + first + threadIdx.x,
                  Load(matrix, column * 65 + threadIdx.x));
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(64) inline void RowTileInverseVf(ComplexSolverShape shape, __gm__ uint8_t* a,
                                                                     __gm__ const int* pivots, __gm__ uint8_t* c,
                                                                     __gm__ int* info)
{
    __ubuf__ float storage[2 * 256 * 65];
    __ubuf__ float columnStorage[2 * 256];
    __ubuf__ int groups[2];
    auto* matrix = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    auto* factors = reinterpret_cast<__ubuf__ aclFloatComplex*>(columnStorage);
    int tiles = (shape.n + 63) / 64;
    for (uint64_t task = blockIdx.x; task < static_cast<uint64_t>(shape.batch) * tiles; task += gridDim.x)
    {
        int batch = task / tiles;
        int first = (task % tiles) * 64;
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        auto* input = MatrixPointer(a, batch);
        int code = ParallelCheckLu(input, shape.n, shape.lda, pivot, groups);
        if (threadIdx.x == 0 && first == 0) info[batch] = code;
        if (code == 0)
        {
            InverseRowTilePhase<true>(shape, input, matrix, factors, first);
            InverseRowTilePhase<false>(shape, input, matrix, factors, first);
            StoreInverseRowTile(shape, c, pivot, batch, first, matrix);
        }
        asc_syncthreads();
    }
}
}  // namespace ComplexSolverKernel
#endif
