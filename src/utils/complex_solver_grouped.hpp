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
#include "complex_solver_panel.hpp"
#include "complex_solver_resident.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int Segments>
__simt_callee__ __aicore__ inline aclFloatComplex GroupedBroadcast(const RegisterMatrix<Segments>& values, int row)
{
    return Broadcast<32>(values.Get(row / 32), row % 32);
}

template <int MaxOrder>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void GroupedUpper(__ubuf__ aclFloatComplex* matrix,
                                                                                   int n)
{
    constexpr int PITCH = MaxOrder + 1;
    int lane = threadIdx.x % 32;
    for (int column = 0; column < n; ++column)
    {
        RegisterMatrix<MaxOrder / 32> sum;
        auto inverse = Load(matrix, column * PITCH + column);
#pragma unroll 8
        for (int k = 0; k < column; ++k)
        {
            auto factor = Load(matrix, column * PITCH + k);
#pragma unroll
            for (int segment = 0; segment < MaxOrder / 32; ++segment)
            {
                auto masked = segment * 32 + lane <= k ? factor : aclFloatComplex{0, 0};
                sum.Set(segment, SubMul(sum.Get(segment), Load(matrix, k * PITCH + segment * 32 + lane), masked));
            }
        }
#pragma unroll
        for (int segment = 0; segment < MaxOrder / 32; ++segment)
        {
            int row = segment * 32 + lane;
            auto high = sum.Get(segment);
            auto value = row == column ? aclFloatComplex{1, 0} : high;
            if (row < column)
                Store(matrix, column * PITCH + row,
                      {fmaf(value.real, inverse.real, -value.imag * inverse.imag),
                       fmaf(value.real, inverse.imag, value.imag * inverse.real)});
        }
    }
}

template <int MaxOrder>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void GroupedLower(__ubuf__ aclFloatComplex* matrix,
                                                                                   int n)
{
    constexpr int PITCH = MaxOrder + 1;
    int lane = threadIdx.x % 32;
    for (int column = n - 1; column >= 0; --column)
    {
        RegisterMatrix<MaxOrder / 32> sum;
#pragma unroll
        for (int segment = 0; segment < MaxOrder / 32; ++segment)
        {
            int row = segment * 32 + lane;
            auto value = Load(matrix, column * PITCH + row);
            sum.Set(segment, row <= column ? value : aclFloatComplex{0, 0});
        }
#pragma unroll 8
        for (int k = column + 1; k < n; ++k)
        {
            auto factor = Load(matrix, column * PITCH + k);
#pragma unroll
            for (int segment = 0; segment < MaxOrder / 32; ++segment)
            {
                sum.Set(segment, SubMul(sum.Get(segment), Load(matrix, k * PITCH + segment * 32 + lane), factor));
            }
        }
#pragma unroll
        for (int segment = 0; segment < MaxOrder / 32; ++segment)
        {
            auto high = sum.Get(segment);
            Store(matrix, column * PITCH + segment * 32 + lane, high);
        }
    }
}

template <int MaxOrder>
__simt_callee__ __aicore__ inline int LoadGroupedLu(ComplexSolverShape shape, __gm__ const aclFloatComplex* input,
                                                    __gm__ const int* pivot, __ubuf__ aclFloatComplex* matrix)
{
    int lane = threadIdx.x % 32;
    for (int index = lane; index < shape.n * MaxOrder; index += 32)
    {
        int row = index % MaxOrder;
        int column = index / MaxOrder;
        Store(matrix, column * (MaxOrder + 1) + row,
              row < shape.n ? Load(input, static_cast<uint64_t>(column) * shape.lda + row) : aclFloatComplex{0, 0});
    }
    int code = 0x7fffffff;
#pragma unroll
    for (int segment = 0; segment < MaxOrder / 32; ++segment)
    {
        int row = segment * 32 + lane;
        if (row < shape.n)
        {
            if (pivot != nullptr && (pivot[row] < row + 1 || pivot[row] > shape.n)) code = -4;
            if (Zero(Load(matrix, row * (MaxOrder + 1) + row)) && code > row + 1) code = row + 1;
        }
    }
    code = asc_reduce_min(code);
    return code == 0x7fffffff ? 0 : code;
}

__simt_callee__ __aicore__ inline void StoreWide(__gm__ aclFloatComplex* output, uint64_t index, aclFloatComplex value)
{
    union Bits
    {
        float real;
        uint32_t bits;
    } real, imag;
    real.real = value.real;
    imag.real = value.imag;
    uint64_t packed = static_cast<uint64_t>(real.bits) | (static_cast<uint64_t>(imag.bits) << 32);
    reinterpret_cast<__gm__ uint64_t*>(output)[index] = packed;
}

template <int MaxOrder>
__simt_callee__ __aicore__ inline void StoreGroupedInverse(ComplexSolverShape shape, __gm__ aclFloatComplex* output,
                                                           __gm__ const int* pivot,
                                                           __ubuf__ const aclFloatComplex* matrix)
{
    int lane = threadIdx.x % 32;
    int targets[MaxOrder / 32];
#pragma unroll
    for (int segment = 0; segment < MaxOrder / 32; ++segment) targets[segment] = segment * 32 + lane;
    if (pivot != nullptr)
        for (int row = shape.n - 1; row >= 0; --row)
        {
            int swapped = pivot[row] - 1;
#pragma unroll
            for (int segment = 0; segment < MaxOrder / 32; ++segment)
                targets[segment] = targets[segment] == row       ? swapped
                                   : targets[segment] == swapped ? row
                                                                 : targets[segment];
        }
    for (int column = 0; column < shape.n; ++column)
    {
        int own = 0;
#pragma unroll
        for (int segment = 0; segment < MaxOrder / 32; ++segment)
            if (column / 32 == segment) own = targets[segment];
        int target = asc_shfl(own, column % 32);
#pragma unroll
        for (int segment = 0; segment < MaxOrder / 32; ++segment)
        {
            int row = segment * 32 + lane;
            if (row < shape.n)
                StoreWide(output, static_cast<uint64_t>(target) * shape.ldc + row,
                          Load(matrix, column * (MaxOrder + 1) + row));
        }
    }
}

template <int MaxOrder, int Groups, int Columns = MaxOrder, int PanelWidth = 1>
__simt_vf__ __aicore__ LAUNCH_BOUND(Groups * 32) inline void GroupedLuInverseVf(ComplexSolverShape shape,
                                                                                __gm__ uint8_t* a,
                                                                                __gm__ const int* pivots,
                                                                                __gm__ uint8_t* c, __gm__ int* info)
{
    __ubuf__ float storage[2 * Groups * Columns * (MaxOrder + 1)];
    auto* matrix = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage) + (threadIdx.x / 32) * Columns * (MaxOrder + 1);
    for (int batch = blockIdx.x * Groups + threadIdx.x / 32; batch < shape.batch; batch += gridDim.x * Groups)
    {
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        int code = LoadGroupedLu<MaxOrder>(shape, MatrixPointer(a, batch), pivot, matrix);
        if (code == 0)
        {
#pragma unroll
            for (int segment = 0; segment < MaxOrder / 32; ++segment)
            {
                int row = segment * 32 + threadIdx.x % 32;
                if (row < shape.n)
                    Store(matrix, row * (MaxOrder + 1) + row,
                          PreparedDivide({1, 0}, PrepareDivisor(Load(matrix, row * (MaxOrder + 1) + row))));
            }
            if constexpr (PanelWidth > 1)
            {
                if (shape.n % PanelWidth == 0)
                {
                    GroupedPanelInverse<MaxOrder, PanelWidth, true>(matrix, shape.n);
                    GroupedPanelInverse<MaxOrder, PanelWidth, false>(matrix, shape.n);
                }
                else
                {
                    GroupedUpper<MaxOrder>(matrix, shape.n);
                    GroupedLower<MaxOrder>(matrix, shape.n);
                }
            }
            else
            {
                GroupedUpper<MaxOrder>(matrix, shape.n);
                GroupedLower<MaxOrder>(matrix, shape.n);
            }
            StoreGroupedInverse<MaxOrder>(shape, MatrixPointer(c, batch), pivot, matrix);
        }
        if (threadIdx.x % 32 == 0) info[batch] = code;
    }
}
}  // namespace ComplexSolverKernel
#endif
