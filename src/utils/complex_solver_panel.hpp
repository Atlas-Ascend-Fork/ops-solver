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

#include "complex_solver_small.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int MaxOrder, int Width, bool Upper>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void AccumulateGroupedPanel(
    __ubuf__ const aclFloatComplex* matrix, RegisterMatrix<Width*(MaxOrder / 32)>& sum, int begin, int end, int first,
    int last)
{
    constexpr int Segments = MaxOrder / 32;
    constexpr int Pitch = MaxOrder + 1;
    int lane = threadIdx.x % 32;
#pragma unroll 4
    for (int k = first; k < last; ++k)
    {
        RegisterMatrix<Segments> left;
#pragma unroll
        for (int segment = 0; segment < Segments; ++segment)
            left.Set(segment, Load(matrix, k * Pitch + segment * 32 + lane));
#pragma unroll
        for (int column = 0; column < Width; ++column)
        {
            if (begin + column >= end) continue;
            auto factor = Load(matrix, (begin + column) * Pitch + k);
#pragma unroll
            for (int segment = 0; segment < Segments; ++segment)
            {
                auto masked = Upper && segment * 32 + lane > k ? aclFloatComplex{0, 0} : factor;
                int index = column * Segments + segment;
                sum.Set(index, SubMul(sum.Get(index), left.Get(segment), masked));
            }
        }
    }
}

template <int MaxOrder, int Width, bool Upper>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void SolveGroupedPanel(
    __ubuf__ aclFloatComplex* matrix, RegisterMatrix<Width*(MaxOrder / 32)>& sum, int begin, int end)
{
    constexpr int Segments = MaxOrder / 32;
    constexpr int Pitch = MaxOrder + 1;
    int lane = threadIdx.x % 32;
#pragma unroll
    for (int step = 0; step < Width; ++step)
    {
        int local = Upper ? step : Width - 1 - step;
        int column = begin + local;
        if (column >= end) continue;
        auto inverse = Load(matrix, column * Pitch + column);
        int first = Upper ? begin : column + 1;
        int last = Upper ? column : end;
#pragma unroll 4
        for (int k = first; k < last; ++k)
        {
            auto factor = Load(matrix, column * Pitch + k);
#pragma unroll
            for (int segment = 0; segment < Segments; ++segment)
            {
                int index = local * Segments + segment;
                auto masked = Upper && segment * 32 + lane > k ? aclFloatComplex{0, 0} : factor;
                sum.Set(index, SubMul(sum.Get(index), Load(matrix, k * Pitch + segment * 32 + lane), masked));
            }
        }
#pragma unroll
        for (int segment = 0; segment < Segments; ++segment)
        {
            int row = segment * 32 + lane;
            auto value = sum.Get(local * Segments + segment);
            if constexpr (Upper)
            {
                if (row < column)
                    Store(matrix, column * Pitch + row,
                          {fmaf(value.real, inverse.real, -value.imag * inverse.imag),
                           fmaf(value.real, inverse.imag, value.imag * inverse.real)});
            }
            else
                Store(matrix, column * Pitch + row, value);
        }
    }
}

template <int MaxOrder, int Width>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void LoadGroupedLowerPanel(
    __ubuf__ const aclFloatComplex* matrix, RegisterMatrix<Width*(MaxOrder / 32)>& sum, int begin)
{
    constexpr int Segments = MaxOrder / 32;
#pragma unroll
    for (int column = 0; column < Width; ++column)
    {
#pragma unroll
        for (int segment = 0; segment < Segments; ++segment)
        {
            int row = segment * 32 + threadIdx.x % 32;
            auto value = Load(matrix, (begin + column) * (MaxOrder + 1) + row);
            sum.Set(column * Segments + segment, row <= begin + column ? value : aclFloatComplex{0, 0});
        }
    }
}

template <int MaxOrder, int Width, bool Upper>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void GroupedPanelInverse(
    __ubuf__ aclFloatComplex* matrix, int n)
{
    constexpr int Segments = MaxOrder / 32;
    int panels = n / Width;
    for (int step = 0; step < panels; ++step)
    {
        int begin = (Upper ? step : panels - 1 - step) * Width;
        int end = begin + Width;
        RegisterMatrix<Width * Segments> sum;
        if constexpr (!Upper) LoadGroupedLowerPanel<MaxOrder, Width>(matrix, sum, begin);
        AccumulateGroupedPanel<MaxOrder, Width, Upper>(matrix, sum, begin, end, Upper ? 0 : end, Upper ? begin : n);
        SolveGroupedPanel<MaxOrder, Width, Upper>(matrix, sum, begin, end);
    }
}
}  // namespace ComplexSolverKernel
#endif
