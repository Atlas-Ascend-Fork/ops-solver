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
#include "complex_solver_kernel.hpp"
#include "complex_solver_launch.h"
#include "complex_solver_small.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
static_assert((THREADS / 32) * 2 == COMPLEX_SOLVER_SMALL_RESIDENT_COLUMNS,
              "Small resident host and kernel column counts must match");

// Keep the roundoff from each real product and subtraction until the pivot row is solved.
__simt_callee__ __aicore__ inline float CompensatedSubtract(float value, float& correction, float left, float right)
{
    float product = left * right;
    float productError = fmaf(left, right, -product);
    float result = value - product;
    float virtualProduct = result - value;
    float sumError = (value - (result - virtualProduct)) - (product + virtualProduct);
    correction += sumError - productError;
    return result;
}

__simt_callee__ __aicore__ inline aclFloatComplex CompensatedSubMul(aclFloatComplex value, aclFloatComplex factor,
                                                                    aclFloatComplex solution,
                                                                    aclFloatComplex& correction)
{
    value.real = CompensatedSubtract(value.real, correction.real, factor.real, solution.real);
    value.real = CompensatedSubtract(value.real, correction.real, -factor.imag, solution.imag);
    value.imag = CompensatedSubtract(value.imag, correction.imag, factor.imag, solution.real);
    value.imag = CompensatedSubtract(value.imag, correction.imag, factor.real, solution.imag);
    return value;
}

template <int Segments, bool Cached, bool Forward>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void WarpResidentStep(
    RegisterMatrix<Segments>& rhs, RegisterMatrix<Segments>& correction, __gm__ const aclFloatComplex* input,
    __ubuf__ const aclFloatComplex* lu, int n, int lda, int pivot)
{
    int lane = threadIdx.x % 32;
    aclFloatComplex value{0, 0};
#pragma unroll
    for (int segment = 0; segment < Segments; ++segment)
        if (pivot / 32 == segment)
        {
            auto low = correction.Get(segment);
            auto high = rhs.Get(segment);
            value = {high.real + low.real, high.imag + low.imag};
        }
    if constexpr (!Forward)
        if (lane == pivot % 32) value = Divide(value, Load(lu, (Cached ? pivot : pivot % PANEL) * n + pivot));
    value = Broadcast<32>(value, pivot % 32);
#pragma unroll
    for (int segment = 0; segment < Segments; ++segment)
    {
        int row = segment * 32 + lane;
        if constexpr (!Forward)
            if (row == pivot)
            {
                rhs.Set(segment, value);
                correction.Set(segment, {0, 0});
            }
        bool update = Forward ? row > pivot && row < n : row < pivot;
        if (update)
        {
            auto factor = Load(lu, (Cached ? pivot : pivot % PANEL) * n + row);
            auto low = correction.Get(segment);
            rhs.Set(segment, CompensatedSubMul(rhs.Get(segment), factor, value, low));
            correction.Set(segment, low);
        }
    }
}

__simt_callee__ __aicore__ inline void LoadResidentPanel(ComplexSolverShape shape, __gm__ const aclFloatComplex* input,
                                                         __ubuf__ aclFloatComplex* lu, int begin)
{
    int columns = shape.n - begin < PANEL ? shape.n - begin : PANEL;
    for (int index = threadIdx.x; index < columns * shape.n; index += blockDim.x)
        Store(lu, index, Load(input, static_cast<uint64_t>(begin + index / shape.n) * shape.lda + index % shape.n));
    asc_syncthreads();
}

template <int Segments, bool Cached>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void WarpResidentSolve(
    ComplexSolverShape shape, __gm__ const aclFloatComplex* input, __gm__ aclFloatComplex* output,
    __ubuf__ aclFloatComplex* lu, __ubuf__ const int* permutation, int column)
{
    RegisterMatrix<Segments> rhs;
    RegisterMatrix<Segments> correction;
    int lane = threadIdx.x % 32;
#pragma unroll
    for (int segment = 0; segment < Segments; ++segment)
    {
        int row = segment * 32 + lane;
        rhs.Set(segment, {row < shape.n && permutation[row] == column ? 1.0f : 0.0f, 0.0f});
        correction.Set(segment, {0, 0});
    }
    for (int pivot = 0; pivot < shape.n; ++pivot)
    {
        if constexpr (!Cached)
            if (pivot % PANEL == 0)
            {
                asc_syncthreads();
                LoadResidentPanel(shape, input, lu, pivot);
            }
        WarpResidentStep<Segments, Cached, true>(rhs, correction, input, lu, shape.n, shape.lda, pivot);
    }
    for (int pivot = shape.n - 1; pivot >= 0; --pivot)
    {
        if constexpr (!Cached)
            if (pivot == shape.n - 1 || pivot % PANEL == PANEL - 1)
            {
                asc_syncthreads();
                LoadResidentPanel(shape, input, lu, (pivot / PANEL) * PANEL);
            }
        WarpResidentStep<Segments, Cached, false>(rhs, correction, input, lu, shape.n, shape.lda, pivot);
    }
#pragma unroll
    for (int segment = 0; segment < Segments; ++segment)
    {
        int row = segment * 32 + lane;
        if (row < shape.n && column < shape.n)
            Store(output, static_cast<uint64_t>(column) * shape.ldc + row, rhs.Get(segment));
    }
}

template <bool Forward>
__simt_callee__ __aicore__ inline void WarpPairStep(aclFloatComplex& rhs0, aclFloatComplex& rhs1, aclFloatComplex& low0,
                                                    aclFloatComplex& low1, __ubuf__ const aclFloatComplex* lu, int n,
                                                    int pivot)
{
    int lane = threadIdx.x % 32;
    aclFloatComplex value0{rhs0.real + low0.real, rhs0.imag + low0.imag};
    aclFloatComplex value1{rhs1.real + low1.real, rhs1.imag + low1.imag};
    if constexpr (!Forward)
        if (lane == pivot)
        {
            auto divisor = PrepareDivisor(Load(lu, pivot * n + pivot));
            value0 = PreparedDivide(value0, divisor);
            value1 = PreparedDivide(value1, divisor);
        }
    value0 = Broadcast<32>(value0, pivot);
    value1 = Broadcast<32>(value1, pivot);
    if constexpr (!Forward)
        if (lane == pivot)
        {
            rhs0 = value0;
            rhs1 = value1;
            low0 = {0, 0};
            low1 = {0, 0};
        }
    if (Forward ? lane > pivot && lane < n : lane < pivot)
    {
        auto coefficient = Load(lu, pivot * n + lane);
        rhs0 = CompensatedSubMul(rhs0, coefficient, value0, low0);
        rhs1 = CompensatedSubMul(rhs1, coefficient, value1, low1);
    }
}

__simt_callee__ __aicore__ inline void WarpPairSolve(ComplexSolverShape shape, __gm__ aclFloatComplex* output,
                                                     __ubuf__ const aclFloatComplex* lu,
                                                     __ubuf__ const int* permutation, int column)
{
    int lane = threadIdx.x % 32;
    int inputRow = lane < shape.n ? permutation[lane] : -1;
    aclFloatComplex rhs0{inputRow == column ? 1.0f : 0.0f, 0};
    aclFloatComplex rhs1{inputRow == column + 1 ? 1.0f : 0.0f, 0};
    aclFloatComplex low0{0, 0}, low1{0, 0};
    for (int pivot = 0; pivot < shape.n; ++pivot) WarpPairStep<true>(rhs0, rhs1, low0, low1, lu, shape.n, pivot);
    for (int pivot = shape.n - 1; pivot >= 0; --pivot) WarpPairStep<false>(rhs0, rhs1, low0, low1, lu, shape.n, pivot);
    if (lane < shape.n && column < shape.n) Store(output, static_cast<uint64_t>(column) * shape.ldc + lane, rhs0);
    if (lane < shape.n && column + 1 < shape.n)
        Store(output, static_cast<uint64_t>(column + 1) * shape.ldc + lane, rhs1);
}

template <int MaxOrder, bool Cached, bool Batched, int RhsColumns = 1>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREADS) inline void WarpResidentInverseVf(ComplexSolverShape shape,
                                                                               __gm__ uint8_t* a,
                                                                               __gm__ const int* pivots,
                                                                               __gm__ uint8_t* c, __gm__ int* info)
{
    static_assert(RhsColumns == 1 || (RhsColumns == 2 && MaxOrder == 32 && Cached),
                  "Paired right-hand sides require the small cached LU path");
    __ubuf__ float luStorage[2 * (Cached ? MaxOrder * MaxOrder : MaxOrder * PANEL)];
    __ubuf__ int permutation[MaxOrder];
    __ubuf__ int status[1];
    __ubuf__ int checkGroups[THREADS / 32];
    auto* lu = reinterpret_cast<__ubuf__ aclFloatComplex*>(luStorage);
    int columnsPerTask = (THREADS / 32) * RhsColumns;
    int tiles = (shape.n + columnsPerTask - 1) / columnsPerTask;
    for (int task = blockIdx.x; task < shape.batch * tiles; task += gridDim.x)
    {
        int batch = task / tiles;
        int column = (task % tiles) * columnsPerTask + (threadIdx.x / 32) * RhsColumns;
        auto* input = Batched ? MatrixPointer(a, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(a);
        auto* output = Batched ? MatrixPointer(c, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(c);
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        if constexpr (Cached)
            for (int i = threadIdx.x; i < shape.n * shape.n; i += blockDim.x)
                Store(lu, i, Load(input, static_cast<uint64_t>(i / shape.n) * shape.lda + i % shape.n));
        for (int row = threadIdx.x; row < shape.n; row += blockDim.x) permutation[row] = row;
        int code = ParallelCheckLu(input, shape.n, shape.lda, pivot, checkGroups);
        if (threadIdx.x == 0)
        {
            status[0] = code;
            if (task % tiles == 0) info[batch] = status[0];
        }
        asc_syncthreads();
        if (threadIdx.x == 0 && status[0] == 0 && pivot != nullptr)
            for (int row = 0; row < shape.n; ++row)
            {
                int target = pivot[row] - 1;
                int value = permutation[row];
                permutation[row] = permutation[target];
                permutation[target] = value;
            }
        asc_syncthreads();
        if (status[0] == 0)
        {
            if constexpr (RhsColumns == 2)
                WarpPairSolve(shape, output, lu, permutation, column);
            else
                WarpResidentSolve<MaxOrder / 32, Cached>(shape, input, output, lu, permutation, column);
        }
        asc_syncthreads();
    }
}

template <bool Batched>
__aicore__ inline void CallResidentInverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* pivot,
                                           __gm__ uint8_t* c, __gm__ uint8_t* info)
{
    auto* pivots = reinterpret_cast<__gm__ const int*>(pivot);
    auto* status = reinterpret_cast<__gm__ int*>(info);
    if (shape.n <= 32)
        asc_vf_call<WarpResidentInverseVf<32, true, Batched, 2>>(dim3{256, 1, 1}, shape, a, pivots, c, status);
    else if (shape.n <= 64)
        asc_vf_call<WarpResidentInverseVf<64, true, Batched>>(dim3{256, 1, 1}, shape, a, pivots, c, status);
    else if (shape.n <= 128)
        asc_vf_call<WarpResidentInverseVf<128, true, Batched>>(dim3{256, 1, 1}, shape, a, pivots, c, status);
    else
        asc_vf_call<WarpResidentInverseVf<256, false, Batched>>(dim3{256, 1, 1}, shape, a, pivots, c, status);
}
}  // namespace ComplexSolverKernel
#endif
