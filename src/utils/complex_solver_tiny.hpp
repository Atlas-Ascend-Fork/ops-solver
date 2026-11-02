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
#include "complex_solver_register.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int Width>
__simt_callee__ __aicore__ inline int LoadTinyLu(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivot,
                                                 int batch, RegisterMatrix<Width * Width>& matrix, int& packed)
{
    auto* input = MatrixPointer(a, batch);
    bool wide = Width == 2 && (reinterpret_cast<uintptr_t>(input) & 7U) == 0;
    int status = 0;
    packed = 0;
#pragma unroll
    for (int column = 0; column < Width; ++column)
    {
        if (column < shape.n)
        {
            int row = pivot == nullptr ? column : pivot[static_cast<uint64_t>(batch) * shape.n + column] - 1;
            if (row < column || row >= shape.n) status = -4;
            packed |= (row & 255) << (column * 8);
        }
        if (Width == 2 && shape.n == 2 && (reinterpret_cast<uintptr_t>(input) & 15U) == 0 && shape.lda % 2 == 0)
        {
            auto values = *reinterpret_cast<__gm__ const float4*>(input + static_cast<uint64_t>(column) * shape.lda);
            matrix.Set(column * Width, {values.x, values.y});
            matrix.Set(column * Width + 1, {values.z, values.w});
            if (Zero(matrix.Get(column * Width + column)) && status == 0) status = column + 1;
        }
        else
        {
#pragma unroll
            for (int row = 0; row < Width; ++row)
            {
                auto value = row < shape.n && column < shape.n
                                 ? (wide ? LoadRowRegisterWide(input, static_cast<uint64_t>(column) * shape.lda + row)
                                         : Load(input, static_cast<uint64_t>(column) * shape.lda + row))
                                 : aclFloatComplex{0, 0};
                matrix.Set(column * Width + row, value);
                if (row == column && row < shape.n && Zero(value) && status == 0) status = row + 1;
            }
        }
    }
    return status;
}

template <int Width>
__simt_callee__ __aicore__ inline void TinyUpper(RegisterMatrix<Width * Width>& matrix, int n)
{
#pragma unroll
    for (int column = 0; column < Width; ++column)
    {
        if (column >= n) continue;
        auto divisor = PrepareDivisor(matrix.Get(column * Width + column));
#pragma unroll
        for (int row = 0; row < Width; ++row)
        {
            if (row > column) continue;
            aclFloatComplex sum{row == column ? 1.0f : 0.0f, 0};
#pragma unroll
            for (int k = 0; k < Width; ++k)
                if (row <= k && k < column)
                    sum = SubMul(sum, matrix.Get(k * Width + row), matrix.Get(column * Width + k));
            matrix.Set(column * Width + row, PreparedDivide(sum, divisor));
        }
    }
}

template <int Width>
__simt_callee__ __aicore__ inline void TinyLower(RegisterMatrix<Width * Width>& matrix, int n)
{
#pragma unroll
    for (int column = Width - 1; column >= 0; --column)
    {
        if (column >= n) continue;
        RegisterMatrix<Width> old;
#pragma unroll
        for (int row = 0; row < Width; ++row) old.Set(row, matrix.Get(column * Width + row));
#pragma unroll
        for (int row = 0; row < Width; ++row)
        {
            auto value = row <= column ? old.Get(row) : aclFloatComplex{0, 0};
#pragma unroll
            for (int k = 0; k < Width; ++k)
                if (k > column && k < n) value = SubMul(value, matrix.Get(k * Width + row), old.Get(k));
            matrix.Set(column * Width + row, value);
        }
    }
}

template <int Width>
__simt_callee__ __aicore__ inline int TinyColumnTarget(int column, int n, int packed)
{
    int target = column;
#pragma unroll
    for (int row = Width - 1; row >= 0; --row)
    {
        if (row >= n) continue;
        int swapped = (static_cast<unsigned>(packed) >> (row * 8)) & 255;
        target = target == row ? swapped : target == swapped ? row : target;
    }
    return target;
}

template <int Width>
__simt_callee__ __aicore__ inline void StoreTinyColumn(ComplexSolverShape shape, __gm__ aclFloatComplex* output,
                                                       const RegisterMatrix<Width * Width>& matrix, int column,
                                                       int target)
{
    if (Width == 2 && shape.n == 2 && (reinterpret_cast<uintptr_t>(output) & 15U) == 0 && shape.ldc % 2 == 0)
    {
        auto first = matrix.Get(column * Width);
        auto second = matrix.Get(column * Width + 1);
        *reinterpret_cast<__gm__ float4*>(output + static_cast<uint64_t>(target) * shape.ldc) =
            float4{first.real, first.imag, second.real, second.imag};
        return;
    }
    bool wide = Width == 2 && (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
#pragma unroll
    for (int row = 0; row < Width; ++row)
    {
        if (row >= shape.n) continue;
        auto value = matrix.Get(column * Width + row);
        if (wide)
            StoreRowRegisterWide(output, static_cast<uint64_t>(target) * shape.ldc + row, value);
        else
            Store(output, static_cast<uint64_t>(target) * shape.ldc + row, value);
    }
}

template <int Width>
__simt_callee__ __aicore__ inline void StoreTinyInverse(ComplexSolverShape shape, __gm__ uint8_t* c, int batch,
                                                        const RegisterMatrix<Width * Width>& matrix, int packed)
{
    auto* output = MatrixPointer(c, batch);
#pragma unroll
    for (int column = 0; column < Width; ++column)
    {
        if (column >= shape.n) continue;
        int target = TinyColumnTarget<Width>(column, shape.n, packed);
        StoreTinyColumn<Width>(shape, output, matrix, column, target);
    }
}

__simt_callee__ __aicore__ inline void TinyInverseTwo(RegisterMatrix<4>& matrix)
{
    auto lower = matrix.Get(1);
    auto upper = matrix.Get(2);
    auto first = PreparedDivide({1, 0}, PrepareDivisor(matrix.Get(0)));
    auto last = PreparedDivide({1, 0}, PrepareDivisor(matrix.Get(3)));
    upper = SubMul({0, 0}, first, upper);
    upper = {fmaf(upper.real, last.real, -upper.imag * last.imag), fmaf(upper.real, last.imag, upper.imag * last.real)};
    matrix.Set(0, SubMul(first, upper, lower));
    matrix.Set(1, SubMul({0, 0}, last, lower));
    matrix.Set(2, upper);
    matrix.Set(3, last);
}

template <int Width, bool Full, int TinyThreads = COMPLEX_SOLVER_TINY_THREADS>
__simt_vf__ __aicore__ LAUNCH_BOUND(TinyThreads) inline void TinyLuInverseVf(ComplexSolverShape shape,
                                                                             __gm__ uint8_t* a, __gm__ const int* pivot,
                                                                             __gm__ uint8_t* c, __gm__ int* info)
{
    if constexpr (Full) shape.n = Width;
    for (int batch = blockIdx.x * blockDim.x + threadIdx.x; batch < shape.batch; batch += gridDim.x * blockDim.x)
    {
        RegisterMatrix<Width * Width> matrix;
        int packed = 0;
        int status = LoadTinyLu<Width>(shape, a, pivot, batch, matrix, packed);
        if (status == 0)
        {
            if constexpr (Width == 2 && Full)
                TinyInverseTwo(matrix);
            else
            {
                TinyUpper<Width>(matrix, shape.n);
                TinyLower<Width>(matrix, shape.n);
            }
            StoreTinyInverse<Width>(shape, c, batch, matrix, packed);
        }
        info[batch] = status;
    }
}

}  // namespace ComplexSolverKernel
#endif
