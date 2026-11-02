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
__simt_callee__ __aicore__ inline void RowRegisterUpper(RegisterMatrix<Width>& matrix, int n, int lane)
{
    auto inverse = PreparedDivide({1, 0}, PrepareDivisor(matrix.Get(lane < n ? lane : 0)));
#pragma unroll
    for (int column = 0; column < Width; ++column)
    {
        if (column >= n) continue;
        auto old = matrix.Get(column);
        aclFloatComplex sum{0, 0};
#pragma unroll
        for (int k = 0; k < Width; ++k)
        {
            if (k >= column) continue;
            auto factor = Broadcast<Width>(old, k);
            sum = SubMul(sum, matrix.Get(k), lane <= k ? factor : aclFloatComplex{0, 0});
        }
        auto value = lane == column ? aclFloatComplex{1, 0} : sum;
        auto reciprocal = Broadcast<Width>(inverse, column);
        auto result = aclFloatComplex{fmaf(value.real, reciprocal.real, -value.imag * reciprocal.imag),
                                      fmaf(value.real, reciprocal.imag, value.imag * reciprocal.real)};
        matrix.Set(column, lane <= column ? result : old);
    }
}

template <int Width>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline void RowRegisterLower(RegisterMatrix<Width>& matrix,
                                                                                       int n, int lane)
{
#pragma unroll
    for (int column = Width - 1; column >= 0; --column)
        if (column < n)
        {
            auto old = matrix.Get(column);
            auto value = lane <= column ? old : aclFloatComplex{0, 0};
#pragma unroll
            for (int k = 0; k < Width; ++k)
                if (k > column && k < n) value = SubMul(value, matrix.Get(k), Broadcast<Width>(old, k));
            matrix.Set(column, value);
        }
}

template <bool Wide>
__simt_callee__ __aicore__ inline aclFloatComplex LoadRowRegister(__gm__ const aclFloatComplex* input, uint64_t index)
{
    if constexpr (Wide)
        return LoadRowRegisterWide(input, index);
    else
        return Load(input, index);
}
template <bool Wide>
__simt_callee__ __aicore__ inline void StoreRowRegister(__gm__ aclFloatComplex* output, uint64_t index,
                                                        aclFloatComplex value)
{
    if constexpr (Wide)
        StoreRowRegisterWide(output, index, value);
    else
        Store(output, index, value);
}
template <int Width>
__simt_callee__ __aicore__ inline int RowRegisterLuStatus(const RegisterMatrix<Width>& matrix, int n, int lane,
                                                          int ownPivot)
{
    int status = 0x7fffffff;
    if (lane < n)
    {
        if (ownPivot < lane || ownPivot >= n) status = -4;
        if (Zero(matrix.Get(lane)) && status > lane + 1) status = lane + 1;
    }
#pragma unroll
    for (int offset = 1; offset < Width; offset *= 2)
    {
        int other = asc_shfl_xor(status, offset, Width);
        status = other < status ? other : status;
    }
    return status == 0x7fffffff ? 0 : status;
}

template <int Width, bool Wide>
__simt_callee__ __aicore__ inline void StoreRowRegisterInverse(ComplexSolverShape shape, __gm__ uint8_t* c,
                                                               const RegisterMatrix<Width>& matrix, int batch, int lane,
                                                               int ownPivot)
{
    int target = lane;
#pragma unroll
    for (int row = Width - 1; row >= 0; --row)
    {
        if (row >= shape.n) continue;
        int swapped = asc_shfl(ownPivot, row, Width);
        target = target == row ? swapped : target == swapped ? row : target;
    }
    auto* output = MatrixPointer(c, batch);
#pragma unroll
    for (int column = 0; column < Width; ++column)
        if (column < shape.n && lane < shape.n)
            StoreRowRegister<Wide>(output, static_cast<uint64_t>(asc_shfl(target, column, Width)) * shape.ldc + lane,
                                   matrix.Get(column));
}

template <int Width, int Threads, bool Full, bool Wide>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void RowRegisterLuInverseVf(ComplexSolverShape shape,
                                                                                __gm__ uint8_t* a,
                                                                                __gm__ const int* pivot,
                                                                                __gm__ uint8_t* c, __gm__ int* info)
{
    if constexpr (Full) shape.n = Width;
    int lane = threadIdx.x % Width;
    for (int batch = blockIdx.x * (Threads / Width) + threadIdx.x / Width; batch < shape.batch;
         batch += gridDim.x * (Threads / Width))
    {
        RegisterMatrix<Width> matrix;
        auto* input = MatrixPointer(a, batch);
#pragma unroll
        for (int column = 0; column < Width; ++column)
            matrix.Set(column, lane < shape.n && column < shape.n
                                   ? LoadRowRegister<Wide>(input, static_cast<uint64_t>(column) * shape.lda + lane)
                                   : aclFloatComplex{0, 0});
        int ownPivot =
            pivot == nullptr || lane >= shape.n ? lane : pivot[static_cast<uint64_t>(batch) * shape.n + lane] - 1;
        int status = RowRegisterLuStatus(matrix, shape.n, lane, ownPivot);
        if (status == 0)
        {
            RowRegisterUpper<Width>(matrix, shape.n, lane);
            RowRegisterLower<Width>(matrix, shape.n, lane);
            StoreRowRegisterInverse<Width, Wide>(shape, c, matrix, batch, lane, ownPivot);
        }
        if (lane == 0) info[batch] = status;
    }
}
template <int Groups>
__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void CheckRowRegisterAlignment(__gm__ uint8_t* a, __gm__ uint8_t* c,
                                                                               int batchCount, __ubuf__ int* flags)
{
    int bits = 0;
    for (int batch = blockIdx.x * Groups + threadIdx.x; threadIdx.x < Groups && batch < batchCount;
         batch += gridDim.x * Groups)
        bits |= (reinterpret_cast<uintptr_t>(MatrixPointer(a, batch)) |
                 reinterpret_cast<uintptr_t>(MatrixPointer(c, batch))) &
                7U;
    for (int offset = 16; offset > 0; offset /= 2) bits |= asc_shfl_xor(bits, offset);
    if (threadIdx.x % 32 == 0) flags[threadIdx.x / 32] = bits;
    asc_syncthreads();
    if (threadIdx.x == 0) flags[0] = flags[0] | flags[1] | flags[2] | flags[3];
}
template <int Width, int Threads, bool Full>
__aicore__ inline void CallRowRegisterInverse(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivots,
                                              __gm__ uint8_t* c, __gm__ int* status)
{
    __ubuf__ int flags[8];
    asc_vf_call<CheckRowRegisterAlignment<Threads / Width>>(dim3{128, 1, 1}, a, c, shape.batch, flags);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
    AscendC::LocalTensor<int> value(AscendC::TPosition::VECCALC,
                                    static_cast<uint32_t>(reinterpret_cast<uintptr_t>(flags)), 8);
    if (value.GetValue(0) == 0)
        asc_vf_call<RowRegisterLuInverseVf<Width, Threads, Full, true>>(dim3{Threads, 1, 1}, shape, a, pivots, c,
                                                                        status);
    else
        asc_vf_call<RowRegisterLuInverseVf<Width, Threads, Full, false>>(dim3{Threads, 1, 1}, shape, a, pivots, c,
                                                                         status);
}

template <int Width>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline int DirectRowPivot(const RegisterMatrix<Width>& matrix,
                                                                                    int column, int n, int lane)
{
    auto value = matrix.Get(column);
    float score = lane >= column && lane < n ? Abs(value.real) + Abs(value.imag) : -1.0f;
    int row = lane;
    for (int offset = Width / 2; offset > 0; offset /= 2)
    {
        float other = asc_shfl_down(score, offset, Width);
        int target = asc_shfl_down(row, offset, Width);
        if (lane + offset < Width && (other > score || (other == score && target < row)))
        {
            score = other;
            row = target;
        }
    }
    return asc_shfl(row, 0, Width);
}

template <int Width>
__simt_callee__ __aicore__ __attribute__((always_inline)) inline int InvertDirectRows(RegisterMatrix<Width>& matrix,
                                                                                      int n, int lane, int& ownPivot)
{
#pragma unroll
    for (int column = 0; column < Width; ++column)
    {
        if (column >= n) continue;
        int pivot = DirectRowPivot(matrix, column, n, lane);
        if (lane == column) ownPivot = pivot;
#pragma unroll
        for (int j = 0; j < Width; ++j)
        {
            aclFloatComplex value{matrix.real[j], matrix.imag[j]};
            int source = lane == column ? pivot : lane == pivot ? column : lane;
            value = Broadcast<Width>(value, source);
            matrix.real[j] = value.real;
            matrix.imag[j] = value.imag;
        }
        auto diagonal = Broadcast<Width>(matrix.Get(column), column);
        if (Zero(diagonal)) return column + 1;
        auto divisor = PrepareDivisor(diagonal);
        auto inverse = PreparedDivide({1, 0}, divisor);
        auto factor = matrix.Get(column);
#pragma unroll
        for (int j = 0; j < Width; ++j)
        {
            aclFloatComplex old{matrix.real[j], matrix.imag[j]};
            auto upper = Broadcast<Width>(old, column);
            upper = j == column ? inverse : SubMul({0, 0}, upper, {-inverse.real, -inverse.imag});
            auto value = j == column ? SubMul({0, 0}, factor, inverse) : SubMul(old, factor, upper);
            value = lane == column ? upper : value;
            matrix.real[j] = value.real;
            matrix.imag[j] = value.imag;
        }
    }
    return 0;
}

template <int Width, int Threads>
__simt_vf__ __aicore__ LAUNCH_BOUND(Threads) inline void DirectRowInverseVf(ComplexSolverShape shape, __gm__ uint8_t* a,
                                                                            __gm__ uint8_t* c, __gm__ int* info)
{
    int lane = threadIdx.x % Width;
    constexpr int GROUPS = Threads / Width;
    for (int batch = blockIdx.x * GROUPS + threadIdx.x / Width; batch < shape.batch; batch += gridDim.x * GROUPS)
    {
        RegisterMatrix<Width> matrix;
        auto* input = MatrixPointer(a, batch);
#pragma unroll
        for (int column = 0; column < Width; ++column)
            matrix.Set(column, lane < shape.n && column < shape.n
                                   ? Load(input, static_cast<uint64_t>(column) * shape.lda + lane)
                                   : aclFloatComplex{0, 0});
        int ownPivot = lane;
        int status = InvertDirectRows(matrix, shape.n, lane, ownPivot);
        int target = lane;
        for (int row = shape.n - 1; row >= 0; --row)
        {
            int pivot = asc_shfl(ownPivot, row, Width);
            target = target == row ? pivot : target == pivot ? row : target;
        }
        auto* output = MatrixPointer(c, batch);
#pragma unroll
        for (int column = 0; column < Width; ++column)
        {
            int outputColumn = asc_shfl(target, column, Width);
            if (status == 0 && lane < shape.n && column < shape.n)
                Store(output, static_cast<uint64_t>(outputColumn) * shape.ldc + lane, matrix.Get(column));
        }
        if (lane == 0) info[batch] = status;
    }
}

}  // namespace ComplexSolverKernel
#endif
