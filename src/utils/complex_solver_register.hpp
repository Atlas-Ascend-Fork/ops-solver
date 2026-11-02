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

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int Order>
struct RegisterMatrix
{
    float real[Order]{};
    float imag[Order]{};
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline aclFloatComplex Get(int row) const
    {
        aclFloatComplex value{0, 0};
#pragma unroll
        for (int index = 0; index < Order; ++index)
            if (row == index) value = {real[index], imag[index]};
        return value;
    }
    __simt_callee__ __aicore__ __attribute__((always_inline)) inline void Set(int row, aclFloatComplex value)
    {
#pragma unroll
        for (int index = 0; index < Order; ++index)
            if (row == index)
            {
                real[index] = value.real;
                imag[index] = value.imag;
            }
    }
};

template <int Width, bool Shared = (Width >= 4)>
struct LaneMatrix : RegisterMatrix<Width>
{
};

template <int Width>
struct LaneMatrix<Width, true>
{
    static constexpr int PITCH = Width + 1;
    __ubuf__ aclFloatComplex* column;
    __simt_callee__ __aicore__ aclFloatComplex Get(int row) const { return Load(column, row * PITCH); }
    __simt_callee__ __aicore__ void Set(int row, aclFloatComplex value) { Store(column, row * PITCH, value); }
};

template <int Width>
__simt_callee__ __aicore__ inline aclFloatComplex Broadcast(aclFloatComplex value, int lane)
{
    return {asc_shfl(value.real, lane, Width), asc_shfl(value.imag, lane, Width)};
}

template <int Width>
__simt_callee__ __aicore__ inline void LoadRegisterLu(__gm__ const aclFloatComplex* input, int n, int lda, int lane,
                                                      __ubuf__ aclFloatComplex* shared, LaneMatrix<Width>& lu)
{
    if constexpr (Width >= 4)
    {
        int group = threadIdx.x / Width;
        lu.column = shared + group * Width * (Width + 1) + lane;
        for (int column = 0; column < Width; ++column)
            Store(shared, group * Width * (Width + 1) + lane * (Width + 1) + column,
                  column < n && lane < n ? Load(input, static_cast<uint64_t>(column) * lda + lane)
                                         : aclFloatComplex{0, 0});
    }
    else
        for (int row = 0; row < n; ++row)
            lu.Set(row, lane < n ? Load(input, static_cast<uint64_t>(lane) * lda + row) : aclFloatComplex{0, 0});
}

template <int Width>
__simt_callee__ __aicore__ inline int RegisterPivot(const LaneMatrix<Width>& lu, int n, int column, int lane)
{
    if constexpr (Width >= 4)
    {
        auto value = Load(lu.column - lane, lane * (Width + 1) + column);
        float score = lane >= column && lane < n ? Abs(value.real) + Abs(value.imag) : -1.0f;
        int row = lane;
        for (int offset = Width / 2; offset > 0; offset /= 2)
        {
            float other = asc_shfl_down(score, offset, Width);
            int otherRow = asc_shfl_down(row, offset, Width);
            if (lane + offset < Width && (other > score || (other == score && otherRow < row)))
            {
                score = other;
                row = otherRow;
            }
        }
        return asc_shfl(row, 0, Width);
    }
    int row = column;
    if (lane == column)
    {
        float best = -1;
        for (int i = column; i < n; ++i)
        {
            auto value = lu.Get(i);
            float score = Abs(value.real) + Abs(value.imag);
            if (score > best)
            {
                best = score;
                row = i;
            }
        }
    }
    return asc_shfl(row, column, Width);
}

template <int Width>
__simt_callee__ __aicore__ inline void RegisterFactor(LaneMatrix<Width>& lu, int n, int lane, int& ownPivot,
                                                      int& status)
{
    for (int column = 0; column < n; ++column)
    {
        int row = RegisterPivot(lu, n, column, lane);
        if (lane == column) ownPivot = row;
        auto old = lu.Get(column);
        lu.Set(column, lu.Get(row));
        lu.Set(row, old);
        auto diagonal = Broadcast<Width>(lu.Get(column), column);
        if (Zero(diagonal) && status == 0) status = column + 1;
        if (lane == column && !Zero(diagonal))
            for (int i = column + 1; i < n; ++i) lu.Set(i, Divide(lu.Get(i), diagonal));
        auto upper = lu.Get(column);
        for (int i = column + 1; i < n; ++i)
        {
            auto lower = Broadcast<Width>(lu.Get(i), column);
            if (lane > column && lane < n) lu.Set(i, SubMul(lu.Get(i), lower, upper));
        }
    }
}

template <int Width>
__simt_callee__ __aicore__ inline int RegisterCheck(const LaneMatrix<Width>& lu, int n, int lane,
                                                    __gm__ const int* pivot, int& ownPivot)
{
    ownPivot = lane < n && pivot != nullptr ? pivot[lane] - 1 : lane;
    int status = 0;
    for (int row = 0; row < n; ++row)
    {
        int value = asc_shfl(ownPivot, row, Width);
        if (value < row || value >= n) status = -4;
    }
    for (int row = 0; row < n; ++row)
    {
        auto diagonal = Broadcast<Width>(lu.Get(row), row);
        if (Zero(diagonal) && status == 0) status = row + 1;
    }
    return status;
}

template <int Width>
__simt_callee__ __aicore__ inline aclFloatComplex SolveCoefficient(const LaneMatrix<Width>& lu, int row, int column,
                                                                   int lane)
{
    if constexpr (Width >= 4)
        return Load(lu.column - lane, row * (Width + 1) + column);
    else
        return Broadcast<Width>(lu.Get(row), column);
}

struct PreparedDivisor
{
    float scale;
    float real;
    float imag;
    float denominator;
    float error;
    float reciprocal;
};

__simt_callee__ __aicore__ inline PreparedDivisor PrepareDivisor(aclFloatComplex value)
{
    union
    {
        float value;
        uint32_t bits;
    } scale;
    scale.value = Abs(value.real) > Abs(value.imag) ? Abs(value.real) : Abs(value.imag);
    uint32_t exponent = scale.bits & 0x7f800000U;
    if (exponent != 0) scale.bits = exponent;
    float real = value.real / scale.value;
    float imag = value.imag / scale.value;
    float first = Abs(real) > Abs(imag) ? real : imag;
    float second = Abs(real) > Abs(imag) ? imag : real;
    float square = first * first;
    float denominator = fmaf(second, second, square);
    float error = fmaf(first, first, -square) + fmaf(second, second, square - denominator);
    return {scale.value, real, imag, denominator, error, 1.0f / denominator};
}

__simt_callee__ __aicore__ inline float PreparedQuotient(float ar, float br, float ai, float bi,
                                                         const PreparedDivisor& divisor)
{
    float product = ar * br;
    float numerator = fmaf(ai, bi, product);
    float error = fmaf(ar, br, -product) + fmaf(ai, bi, product - numerator);
    float result = numerator * divisor.reciprocal;
    float remainder = fmaf(-result, divisor.denominator, numerator) + error - result * divisor.error;
    return fmaf(remainder, divisor.reciprocal, result);
}

__simt_callee__ __aicore__ inline aclFloatComplex PreparedDivide(aclFloatComplex value, const PreparedDivisor& divisor)
{
    float real = value.real / divisor.scale;
    float imag = value.imag / divisor.scale;
    return {PreparedQuotient(real, divisor.real, imag, divisor.imag, divisor),
            PreparedQuotient(imag, divisor.real, -real, divisor.imag, divisor)};
}

template <int Width>
__simt_callee__ __aicore__ inline PreparedDivisor BroadcastDivisor(PreparedDivisor divisor, int column)
{
    return {asc_shfl(divisor.scale, column, Width), asc_shfl(divisor.real, column, Width),
            asc_shfl(divisor.imag, column, Width),  asc_shfl(divisor.denominator, column, Width),
            asc_shfl(divisor.error, column, Width), asc_shfl(divisor.reciprocal, column, Width)};
}

template <int Width>
__simt_callee__ __aicore__ inline void RegisterSolve(const LaneMatrix<Width>& lu, LaneMatrix<Width>& rhs, int n,
                                                     int lane, int ownPivot)
{
    auto divisor = PrepareDivisor(lu.Get(lane < n ? lane : 0));
    for (int row = 0; row < n; ++row) rhs.Set(row, {row == lane ? 1.0f : 0.0f, 0.0f});
    for (int row = 0; row < n; ++row)
    {
        int pivot = asc_shfl(ownPivot, row, Width);
        auto old = rhs.Get(row);
        rhs.Set(row, rhs.Get(pivot));
        rhs.Set(pivot, old);
    }
    for (int column = 0; column < n; ++column)
    {
        auto value = rhs.Get(column);
        for (int row = column + 1; row < n; ++row)
            rhs.Set(row, SubMul(rhs.Get(row), SolveCoefficient(lu, row, column, lane), value));
    }
    for (int column = n - 1; column >= 0; --column)
    {
        auto value = PreparedDivide(rhs.Get(column), BroadcastDivisor<Width>(divisor, column));
        rhs.Set(column, value);
        for (int row = 0; row < column; ++row)
            rhs.Set(row, SubMul(rhs.Get(row), SolveCoefficient(lu, row, column, lane), value));
    }
}

template <int Width>
__simt_callee__ __aicore__ inline int InvertSmallInPlace(LaneMatrix<Width>& matrix, int n, int lane, int& ownPivot)
{
    for (int column = 0; column < n; ++column)
    {
        int pivot = RegisterPivot(matrix, n, column, lane);
        if (lane == column) ownPivot = pivot;
        auto old = matrix.Get(column);
        matrix.Set(column, matrix.Get(pivot));
        matrix.Set(pivot, old);
        auto diagonal = SolveCoefficient(matrix, column, column, lane);
        if (Zero(diagonal)) return column + 1;
        auto divisor = PrepareDivisor(diagonal);
        auto inverse = PreparedDivide({1, 0}, divisor);
        auto upper = lane == column ? inverse : PreparedDivide(matrix.Get(column), divisor);
        matrix.Set(column, upper);
        for (int row = 0; row < n; ++row)
            if (row != column)
            {
                auto factor = SolveCoefficient(matrix, row, column, lane);
                auto value = lane == column
                                 ? aclFloatComplex{fmaf(factor.imag, inverse.imag, -factor.real * inverse.real),
                                                   fmaf(-factor.real, inverse.imag, -factor.imag * inverse.real)}
                                 : SubMul(matrix.Get(row), factor, upper);
                matrix.Set(row, value);
            }
    }
    return 0;
}

template <int Width>
__simt_callee__ __aicore__ inline void SolveDirectInPlace(ComplexSolverShape shape, __gm__ uint8_t* c, __gm__ int* info,
                                                          int batch, int lane, LaneMatrix<Width>& lu)
{
    int ownPivot = lane;
    int status = InvertSmallInPlace(lu, shape.n, lane, ownPivot);
    int target = lane;
    for (int row = shape.n - 1; row >= 0; --row)
    {
        int pivot = asc_shfl(ownPivot, row, Width);
        target = target == row ? pivot : target == pivot ? row : target;
    }
    if (status == 0)
    {
        auto* output = MatrixPointer(c, batch);
        for (int column = 0; column < shape.n; ++column)
        {
            int outputColumn = asc_shfl(target, column, Width);
            if (lane < shape.n)
                Store(output, static_cast<uint64_t>(outputColumn) * shape.ldc + lane,
                      SolveCoefficient(lu, lane, column, lane));
        }
    }
    if (lane == 0) info[batch] = status;
}

template <int Width, bool Direct>
__simt_callee__ __aicore__ inline void SolveRegisterBatch(ComplexSolverShape shape, __gm__ uint8_t* c,
                                                          __gm__ const int* pivot, __gm__ int* info, int batch,
                                                          int lane, LaneMatrix<Width>& lu,
                                                          __ubuf__ aclFloatComplex* sharedRhs)
{
    int ownPivot = lane;
    int status = 0;
    if constexpr (Direct)
        RegisterFactor(lu, shape.n, lane, ownPivot, status);
    else
        status = RegisterCheck(lu, shape.n, lane,
                               pivot == nullptr ? nullptr : pivot + static_cast<uint64_t>(batch) * shape.n, ownPivot);
    if (status == 0)
    {
        LaneMatrix<Width> rhs;
        if constexpr (Width >= 4) rhs.column = sharedRhs + (threadIdx.x / Width) * Width * (Width + 1) + lane;
        RegisterSolve(lu, rhs, shape.n, lane, ownPivot);
        auto* output = MatrixPointer(c, batch);
        if (lane < shape.n)
            for (int row = 0; row < shape.n; ++row)
                Store(output, static_cast<uint64_t>(lane) * shape.ldc + row, rhs.Get(row));
    }
    if (lane == 0) info[batch] = status;
}

template <int Width, typename Matrix>
__simt_callee__ __aicore__ inline void InvertLuUpper(Matrix& matrix, int n, int lane)
{
    auto inverse = PreparedDivide({1, 0}, PrepareDivisor(matrix.Get(lane < n ? lane : 0)));
    for (int column = 0; column < n; ++column)
    {
        auto old = matrix.Get(column);
        auto reciprocal = Broadcast<Width>(inverse, column);
        aclFloatComplex sum{0, 0};
        for (int k = 0; k < column; ++k)
        {
            auto value = Broadcast<Width>(old, k);
            if (lane <= k) sum = SubMul(sum, matrix.Get(k), value);
        }
        auto value = lane == column ? aclFloatComplex{1, 0} : sum;
        aclFloatComplex result{fmaf(value.real, reciprocal.real, -value.imag * reciprocal.imag),
                               fmaf(value.real, reciprocal.imag, value.imag * reciprocal.real)};
        if (lane <= column) matrix.Set(column, result);
    }
}

template <int Width, typename Matrix>
__simt_callee__ __aicore__ inline void MultiplyInverseLower(Matrix& matrix, int n, int lane)
{
    for (int column = n - 1; column >= 0; --column)
    {
        auto old = matrix.Get(column);
        auto value = lane <= column ? old : aclFloatComplex{0, 0};
        for (int k = column + 1; k < n; ++k) value = SubMul(value, matrix.Get(k), Broadcast<Width>(old, k));
        matrix.Set(column, value);
    }
}

template <int Width, int Columns = Width>
__simt_callee__ __aicore__ inline void SolveLuInPlace(ComplexSolverShape shape, __gm__ uint8_t* a,
                                                      __gm__ const int* pivot, __gm__ uint8_t* c, __gm__ int* info,
                                                      int batch, int lane, __ubuf__ aclFloatComplex* shared)
{
    LaneMatrix<Width> matrix;
    matrix.column = shared + (threadIdx.x / Width) * Columns * (Width + 1) + lane;
    // Each lane owns a row; the shared matrix keeps the public column order.
    auto* input = MatrixPointer(a, batch);
    for (int column = 0; column < Columns; ++column)
        matrix.Set(column, lane < shape.n && column < shape.n
                               ? Load(input, static_cast<uint64_t>(column) * shape.lda + lane)
                               : aclFloatComplex{0, 0});
    int ownPivot =
        pivot == nullptr ? lane : pivot[static_cast<uint64_t>(batch) * shape.n + (lane < shape.n ? lane : 0)] - 1;
    int status = 0x7fffffff;
    if (lane < shape.n)
    {
        if (ownPivot < lane || ownPivot >= shape.n) status = -4;
        if (Zero(matrix.Get(lane)) && status > lane + 1) status = lane + 1;
    }
    for (int offset = 1; offset < Width; offset *= 2)
    {
        int other = asc_shfl_xor(status, offset, Width);
        status = other < status ? other : status;
    }
    if (status == 0x7fffffff) status = 0;
    if (status == 0)
    {
        InvertLuUpper<Width>(matrix, shape.n, lane);
        MultiplyInverseLower<Width>(matrix, shape.n, lane);
        auto* output = MatrixPointer(c, batch);
        int ownTarget = lane;
        for (int row = shape.n - 1; row >= 0; --row)
        {
            int swapped = asc_shfl(ownPivot, row, Width);
            ownTarget = ownTarget == row ? swapped : ownTarget == swapped ? row : ownTarget;
        }
        for (int column = 0; column < shape.n; ++column)
        {
            int target = asc_shfl(ownTarget, column, Width);
            if (lane < shape.n) Store(output, static_cast<uint64_t>(target) * shape.ldc + lane, matrix.Get(column));
        }
    }
    if (lane == 0) info[batch] = status;
}

template <int Width, bool Direct, bool Full = false, int BatchThreads = COMPLEX_SOLVER_BATCH_THREADS<Width, Direct>,
          int Columns = Width>
__simt_vf__ __aicore__ LAUNCH_BOUND(BatchThreads) inline void RegisterInverseVf(ComplexSolverShape shape,
                                                                                __gm__ uint8_t* a,
                                                                                __gm__ const int* pivot,
                                                                                __gm__ uint8_t* c, __gm__ int* info)
{
    if constexpr (Full) shape.n = Width;
    constexpr int SHARED_VALUES = Width >= 4 ? (BatchThreads / Width) * Columns * (Width + 1) : 1;
    __ubuf__ float storage[(Width >= 4 ? 2 : 4) * SHARED_VALUES];
    auto* shared = reinterpret_cast<__ubuf__ aclFloatComplex*>(storage);
    int lane = threadIdx.x % Width;
    int groups = blockDim.x / Width;
    for (int base = blockIdx.x * groups; base < shape.batch; base += gridDim.x * groups)
    {
        int batch = base + threadIdx.x / Width;
        LaneMatrix<Width> lu;
        if (batch < shape.batch && (Direct || Width < 4))
            LoadRegisterLu(MatrixPointer(a, batch), shape.n, shape.lda, lane, shared, lu);
        if constexpr (Width >= 4) asc_syncthreads();
        if (batch < shape.batch)
        {
            if constexpr (!Direct && Width >= 4)
                SolveLuInPlace<Width, Columns>(shape, a, pivot, c, info, batch, lane, shared);
            else if constexpr (Direct && Width >= 4)
                SolveDirectInPlace(shape, c, info, batch, lane, lu);
            else
                SolveRegisterBatch<Width, Direct>(shape, c, pivot, info, batch, lane, lu, shared + SHARED_VALUES);
        }
        if constexpr (Width >= 4) asc_syncthreads();
    }
}

__simt_callee__ __aicore__ inline aclFloatComplex LoadRowRegisterWide(__gm__ const aclFloatComplex* input,
                                                                      uint64_t index)
{
    union Value
    {
        uint64_t bits;
        aclFloatComplex value;
    } packed;
    packed.bits = reinterpret_cast<__gm__ const uint64_t*>(input)[index];
    return packed.value;
}
__simt_callee__ __aicore__ inline void StoreRowRegisterWide(__gm__ aclFloatComplex* output, uint64_t index,
                                                            aclFloatComplex value)
{
    union Value
    {
        uint64_t bits;
        aclFloatComplex value;
    } packed;
    packed.value = value;
    reinterpret_cast<__gm__ uint64_t*>(output)[index] = packed.bits;
}

}  // namespace ComplexSolverKernel
#endif
