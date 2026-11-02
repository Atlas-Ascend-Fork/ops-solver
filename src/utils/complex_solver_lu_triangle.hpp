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
#include "complex_solver_lu_vector.hpp"

#if __NPU_ARCH__ == 3510
namespace ComplexLuTriangle
{
namespace A = AscendC;
using ComplexLuVector::Prepare;
using ComplexLuVector::Ptr;
using ComplexLuVector::ScalePivot;
using ComplexLuVector::StoreInverseTile;
using ComplexLuVector::UpdateColumn;
template <bool Upper>
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Unpack(__ubuf__ const float* stage, __ubuf__ float* ar,
                                                            __ubuf__ float* ai, int n, int pitch, int first,
                                                            int columns)
{
    for (int index = threadIdx.x; index < pitch * columns; index += blockDim.x)
    {
        int row = index % pitch, column = first + index / pitch;
        if (row < n && (Upper ? row < column : row > column))
        {
            int target = Upper ? row * (2 * n - row - 1) / 2 + column - row - 1 : row * (row - 1) / 2 + column;
            ar[target] = -stage[index * 2];
            ai[target] = -stage[index * 2 + 1];
        }
    }
}
template <int Rows, bool Upper>
__simd_vf__ inline void Solve(__ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi,
                              __ubuf__ float* inverse, int n, int pitch, int first)
{
    uint32_t count = n - first < Rows ? n - first : Rows;
    auto mask = A::Reg::UpdateMask<float>(count);
    A::Reg::RegTensor<float> pr, pi, dr, di, negativePi;
    if constexpr (Upper)
    {
        for (int p = first; p < n; ++p)
        {
            A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
            A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(dr, inverse + p);
            A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(di, inverse + pitch + p);
            ScalePivot(xr, xi, p * Rows, dr, di, pr, pi, mask);
            A::Reg::Muls(negativePi, pi, -1.0f, mask);
            int base = p * (2 * n - p - 1) / 2 - p - 1;
            for (int column = p + 1; column < n; ++column)
                UpdateColumn(xr, xi, ar, ai, pr, pi, negativePi, column * Rows, base + column, mask);
        }
    }
    else
    {
        for (int p = n - 1; p > 0; --p)
        {
            A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
            A::Reg::LoadAlign(pr, xr + p * Rows);
            A::Reg::LoadAlign(pi, xi + p * Rows);
            A::Reg::Muls(negativePi, pi, -1.0f, mask);
            int base = p * (p - 1) / 2;
            for (int column = 0; column < p; ++column)
                UpdateColumn(xr, xi, ar, ai, pr, pi, negativePi, column * Rows, base + column, mask);
        }
    }
}
template <int Rows>
class Inverse
{
    template <bool Upper>
    __aicore__ inline void Load(ComplexSolverShape shape, __gm__ uint8_t* a, int batch, A::LocalTensor<float> stage,
                                A::LocalTensor<float> ar, A::LocalTensor<float> ai, int pitch)
    {
        A::GlobalTensor<float> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ComplexSolverKernel::MatrixPointer(a, batch)));
        for (int column = 0; column < shape.n; column += 32)
        {
            int columns = shape.n - column < 32 ? shape.n - column : 32;
            A::DataCopyExtParams cp{static_cast<uint16_t>(columns), static_cast<uint32_t>(shape.n * 8),
                                    static_cast<uint32_t>((shape.lda - shape.n) * 8),
                                    static_cast<uint32_t>((pitch - (shape.n + 3) / 4 * 4) / 4), 0};
            A::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            A::SetFlag<A::HardEvent::V_MTE2>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::V_MTE2>(EVENT_ID0);
            A::DataCopyPad(stage, input[static_cast<uint64_t>(column) * shape.lda * 2], cp, pad);
            A::SetFlag<A::HardEvent::MTE2_V>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<Unpack<Upper>>(dim3{256, 1, 1}, Ptr(stage), Ptr(ar), Ptr(ai), shape.n, pitch, column, columns);
        }
    }
    __aicore__ inline void Matrix(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivots,
                                  __gm__ uint8_t* c, __gm__ int* info, A::LocalTensor<float> arena, uint64_t task,
                                  int pitch)
    {
        int tiles = (shape.n + Rows - 1) / Rows;
        int batch = task / tiles, first = (task % tiles) * Rows;
        int plane = (shape.n * (shape.n - 1) / 2 + 7) / 8 * 8, outputPlane = shape.n * Rows;
        int sharedPlane = plane > outputPlane ? plane : outputPlane;
        auto ar = arena, ai = arena[plane], xr = arena[2 * sharedPlane], xi = xr[outputPlane];
        auto stage = xi[outputPlane], inverse = stage[64 * pitch + 64];
        auto targets = inverse[2 * pitch].template ReinterpretCast<int>(), status = targets[2 * pitch];
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        asc_vf_call<Prepare>(dim3{128, 1, 1}, shape, ComplexSolverKernel::MatrixPointer(a, batch), pivot, Ptr(ar),
                             Ptr(ai), Ptr(xr), Ptr(xi), Ptr(inverse),
                             reinterpret_cast<__ubuf__ int*>(targets.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ int*>(status.GetPhyAddr()), pitch, Rows, first);
        A::SetFlag<A::HardEvent::V_S>(EVENT_ID0);
        A::WaitFlag<A::HardEvent::V_S>(EVENT_ID0);
        int code = status.GetValue(0);
        if (first == 0) info[batch] = code;
        if (code != 0) return;
        Load<true>(shape, a, batch, stage, ar, ai, pitch);
        asc_vf_call<Solve<Rows, true>>(Ptr(ar), Ptr(ai), Ptr(xr), Ptr(xi), Ptr(inverse), shape.n, pitch, first);
        Load<false>(shape, a, batch, stage, ar, ai, pitch);
        asc_vf_call<Solve<Rows, false>>(Ptr(ar), Ptr(ai), Ptr(xr), Ptr(xi), Ptr(inverse), shape.n, pitch, first);
        StoreInverseTile<Rows>(shape, c, batch, first, ar, xr, xi, targets);
    }

   public:
    __aicore__ inline void Run(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivots, __gm__ uint8_t* c,
                               __gm__ int* info, __ubuf__ float* memory)
    {
        A::LocalTensor<float> arena(A::TPosition::VECCALC, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)),
                                    COMPLEX_SOLVER_LU_VECTOR_BYTES / sizeof(float));
        int pitch = (shape.n + 7) / 8 * 8;
        uint64_t tasks = static_cast<uint64_t>(shape.batch) * ((shape.n + Rows - 1) / Rows);
        for (uint64_t task = A::GetBlockIdx(); task < tasks; task += A::GetBlockNum())
            Matrix(shape, a, pivots, c, info, arena, task, pitch);
    }
};
}  // namespace ComplexLuTriangle
#endif
