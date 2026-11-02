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
namespace ComplexLuVector
{
namespace A = AscendC;
__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void Prepare(
    ComplexSolverShape shape, __gm__ const aclFloatComplex* input, __gm__ const int* pivot, __ubuf__ float* ar,
    __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ float* inverse, __ubuf__ int* targets,
    __ubuf__ int* status, __gm__ int* info, int batch, int pitch, int rows, int first)
{
    __ubuf__ int checks[4];
    int code = ComplexSolverKernel::ParallelCheckLu(input, shape.n, shape.lda, pivot, checks);
    if (threadIdx.x == 0)
    {
        status[0] = code;
        if (first == 0) info[batch] = code;
    }
    if (code != 0) return;
    __ubuf__ int* cachedPivots = targets + pitch;
    if (pivot != nullptr)
    {
        for (int row = threadIdx.x; row < shape.n; row += blockDim.x) cachedPivots[row] = pivot[row];
        asc_syncthreads();
    }
    for (int index = threadIdx.x; index < shape.n * rows; index += blockDim.x)
    {
        xr[index] = index / rows == first + index % rows ? 1.0f : 0.0f;
        xi[index] = 0.0f;
    }
    for (int column = threadIdx.x; column < shape.n; column += blockDim.x)
    {
        auto value = ComplexSolverKernel::PreparedDivide(
            {1, 0}, ComplexSolverKernel::PrepareDivisor(
                        ComplexSolverKernel::Load(input, static_cast<uint64_t>(column) * shape.lda + column)));
        inverse[column] = value.real;
        inverse[pitch + column] = value.imag;
        targets[column] = column;
    }
    asc_syncthreads();
    // Apply each interchange once. This is the source-to-destination column map.
    if (pivot != nullptr && threadIdx.x == 0)
        for (int k = 0; k < shape.n; ++k)
        {
            int swapped = cachedPivots[k] - 1;
            int value = targets[k];
            targets[k] = targets[swapped];
            targets[swapped] = value;
        }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void Unpack(__ubuf__ const float* stage, __ubuf__ float* ar,
                                                            __ubuf__ float* ai, int n, int pitch, int rows, int first,
                                                            int live)
{
    for (int index = threadIdx.x; index < n * rows; index += blockDim.x)
    {
        int row = index % rows;
        if (row < live)
        {
            int target = index / rows * pitch + first + row;
            ar[target] = -stage[index * 2];
            ai[target] = -stage[index * 2 + 1];
        }
    }
}
__simd_callee__ inline void UpdateColumn(__ubuf__ float* xr, __ubuf__ float* xi, __ubuf__ float* ar, __ubuf__ float* ai,
                                         A::Reg::RegTensor<float>& pr, A::Reg::RegTensor<float>& pi,
                                         A::Reg::RegTensor<float>& negativePi, int position, int coefficient,
                                         A::Reg::MaskReg& mask)
{
    A::Reg::RegTensor<float> cr, ci, vr, vi, negative;
    A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(cr, ar + coefficient);
    A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(ci, ai + coefficient);
    A::Reg::LoadAlign(vr, xr + position);
    A::Reg::LoadAlign(vi, xi + position);
    A::Reg::MulAddDst(vr, pr, cr, mask);
    A::Reg::MulAddDst(vi, pr, ci, mask);
    A::Reg::MulAddDst(vi, pi, cr, mask);
    A::Reg::MulAddDst(vr, negativePi, ci, mask);
    A::Reg::StoreAlign(xr + position, vr, mask);
    A::Reg::StoreAlign(xi + position, vi, mask);
}
__simd_callee__ inline void ScalePivot(__ubuf__ float* xr, __ubuf__ float* xi, int offset, A::Reg::RegTensor<float>& dr,
                                       A::Reg::RegTensor<float>& di, A::Reg::RegTensor<float>& pr,
                                       A::Reg::RegTensor<float>& pi, A::Reg::MaskReg& mask)
{
    A::Reg::RegTensor<float> vr, vi, negative;
    A::Reg::LoadAlign(vr, xr + offset);
    A::Reg::LoadAlign(vi, xi + offset);
    A::Reg::Mul(pr, vr, dr, mask);
    A::Reg::Muls(negative, vi, -1.0f, mask);
    A::Reg::MulAddDst(pr, negative, di, mask);
    A::Reg::Mul(pi, vi, dr, mask);
    A::Reg::MulAddDst(pi, vr, di, mask);
    A::Reg::StoreAlign(xr + offset, pr, mask);
    A::Reg::StoreAlign(xi + offset, pi, mask);
}
template <int Rows, bool Tail>
__simd_callee__ inline void Upper(__ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi,
                                  __ubuf__ float* inverse, uint16_t n, uint16_t pitch, uint16_t first, uint16_t begin,
                                  uint16_t end, A::Reg::MaskReg& mask, A::Reg::MaskReg& tailMask)
{
    A::Reg::RegTensor<float> pr, pi, dr, di, qr, qi, negativePi, negativeQi;
    for (uint16_t p = begin; p < end; ++p)
    {
        A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
        A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(dr, inverse + p);
        A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(di, inverse + pitch + p);
        ScalePivot(xr, xi, p * Rows, dr, di, pr, pi, mask);
        A::Reg::Muls(negativePi, pi, -1.0f, mask);
        if constexpr (Tail)
        {
            ScalePivot(xr, xi, p * Rows + 64, dr, di, qr, qi, tailMask);
            A::Reg::Muls(negativeQi, qi, -1.0f, tailMask);
        }
        for (uint16_t delta = 1; delta < n - p; ++delta)
        {
            UpdateColumn(xr, xi, ar, ai, pr, pi, negativePi, (p * Rows + delta * Rows), (p * pitch + delta * pitch + p),
                         mask);
            if constexpr (Tail)
                UpdateColumn(xr, xi, ar, ai, qr, qi, negativeQi, (p * Rows + delta * Rows) + 64,
                             (p * pitch + delta * pitch + p), tailMask);
        }
    }
}
template <int Rows>
__simd_vf__ inline void Solve(__ubuf__ float* ar, __ubuf__ float* ai, __ubuf__ float* xr, __ubuf__ float* xi,
                              __ubuf__ float* inverse, uint16_t n, uint16_t pitch, uint16_t first)
{
    uint32_t count = n - first < 64 ? n - first : 64;
    auto mask = A::Reg::UpdateMask<float>(count);
    uint32_t tail = n - first > 64 ? n - first - 64 : 0;
    auto tailMask = A::Reg::UpdateMask<float>(tail);
    uint16_t split = first + 64 < n ? first + 64 : n;
    Upper<Rows, false>(ar, ai, xr, xi, inverse, n, pitch, first, first, split, mask, tailMask);
    if constexpr (Rows > 64)
        Upper<Rows, true>(ar, ai, xr, xi, inverse, n, pitch, first, split, n, mask, tailMask);
    else
        Upper<Rows, false>(ar, ai, xr, xi, inverse, n, pitch, first, split, n, mask, tailMask);
    A::Reg::RegTensor<float> pr, pi, qr, qi, negativePi, negativeQi;
    for (uint16_t step = 1; step < n; ++step)
    {
        uint16_t p = n - step;
        A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
        A::Reg::LoadAlign(pr, xr + p * Rows);
        A::Reg::LoadAlign(pi, xi + p * Rows);
        A::Reg::Muls(negativePi, pi, -1.0f, mask);
        if constexpr (Rows > 64)
        {
            A::Reg::LoadAlign(qr, xr + p * Rows + 64);
            A::Reg::LoadAlign(qi, xi + p * Rows + 64);
            A::Reg::Muls(negativeQi, qi, -1.0f, tailMask);
        }
        for (uint16_t column = 0; column < p; ++column)
        {
            UpdateColumn(xr, xi, ar, ai, pr, pi, negativePi, column * Rows, column * pitch + p, mask);
            if constexpr (Rows > 64)
                UpdateColumn(xr, xi, ar, ai, qr, qi, negativeQi, column * Rows + 64, column * pitch + p, tailMask);
        }
    }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void Pack(__ubuf__ float* stage, __ubuf__ const float* xr,
                                                          __ubuf__ const float* xi, __ubuf__ const int* targets, int n,
                                                          int rows)
{
    for (int index = threadIdx.x; index < n * rows; index += blockDim.x)
    {
        int target = targets[index / rows] * rows + index % rows;
        stage[2 * target] = xr[index];
        stage[2 * target + 1] = xi[index];
    }
}
__aicore__ inline __ubuf__ float* Ptr(A::LocalTensor<float> value)
{
    return reinterpret_cast<__ubuf__ float*>(value.GetPhyAddr());
}

template <int Rows>
__aicore__ inline void StoreInverseTile(ComplexSolverShape shape, __gm__ uint8_t* c, int batch, int first,
                                        A::LocalTensor<float> ar, A::LocalTensor<float> xr, A::LocalTensor<float> xi,
                                        A::LocalTensor<int> targets)
{
    asc_vf_call<Pack>(dim3{128, 1, 1}, Ptr(ar), Ptr(xr), Ptr(xi), reinterpret_cast<__ubuf__ int*>(targets.GetPhyAddr()),
                      shape.n, Rows);
    A::GlobalTensor<float> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ComplexSolverKernel::MatrixPointer(c, batch)));
    int live = shape.n - first < Rows ? shape.n - first : Rows;
    A::DataCopyExtParams cp{static_cast<uint16_t>(shape.n), static_cast<uint32_t>(live * 8),
                            static_cast<uint32_t>((Rows - (live + 3) / 4 * 4) / 4),
                            static_cast<uint32_t>((shape.ldc - live) * 8), 0};
    A::SetFlag<A::HardEvent::V_MTE3>(EVENT_ID0);
    A::WaitFlag<A::HardEvent::V_MTE3>(EVENT_ID0);
    A::DataCopyPad(output[first * 2], ar, cp);
    A::SetFlag<A::HardEvent::MTE3_V>(EVENT_ID0);
    A::WaitFlag<A::HardEvent::MTE3_V>(EVENT_ID0);
}

template <int Rows>
class Inverse
{
    __aicore__ inline void Load(ComplexSolverShape shape, __gm__ uint8_t* a, int batch, A::LocalTensor<float> stage,
                                A::LocalTensor<float> ar, A::LocalTensor<float> ai, int pitch)
    {
        A::GlobalTensor<float> input;
        input.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ComplexSolverKernel::MatrixPointer(a, batch)));
        for (int row = 0; row < shape.n; row += Rows)
        {
            int live = shape.n - row < Rows ? shape.n - row : Rows;
            A::DataCopyExtParams cp{static_cast<uint16_t>(shape.n), static_cast<uint32_t>(live * 8),
                                    static_cast<uint32_t>((shape.lda - live) * 8),
                                    static_cast<uint32_t>((Rows - (live + 3) / 4 * 4) / 4), 0};
            A::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
            A::SetFlag<A::HardEvent::V_MTE2>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::V_MTE2>(EVENT_ID0);
            A::DataCopyPad(stage, input[row * 2], cp, pad);
            A::SetFlag<A::HardEvent::MTE2_V>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<Unpack>(dim3{256, 1, 1}, Ptr(stage), Ptr(ar), Ptr(ai), shape.n, pitch, Rows, row, live);
        }
    }
    __aicore__ inline void Matrix(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivots,
                                  __gm__ uint8_t* c, __gm__ int* info, A::LocalTensor<float> arena, uint64_t task,
                                  int pitch)
    {
        int tiles = (shape.n + Rows - 1) / Rows;
        int batch = task / tiles, first = (task % tiles) * Rows;
        int plane = shape.n * pitch, outputPlane = shape.n * Rows;
        int sharedPlane = plane > outputPlane ? plane : outputPlane;
        auto ar = arena, ai = arena[plane], xr = arena[2 * sharedPlane], xi = arena[2 * sharedPlane + outputPlane];
        auto inverse = arena[2 * sharedPlane + 2 * outputPlane + 64];
        auto targets = inverse[2 * pitch].template ReinterpretCast<int>();
        auto status = targets[2 * pitch];
        auto* pivot = pivots == nullptr ? nullptr : pivots + static_cast<uint64_t>(batch) * shape.n;
        Load(shape, a, batch, xr, ar, ai, pitch);
        asc_vf_call<Prepare>(dim3{128, 1, 1}, shape, ComplexSolverKernel::MatrixPointer(a, batch), pivot, Ptr(ar),
                             Ptr(ai), Ptr(xr), Ptr(xi), Ptr(inverse),
                             reinterpret_cast<__ubuf__ int*>(targets.GetPhyAddr()),
                             reinterpret_cast<__ubuf__ int*>(status.GetPhyAddr()), info, batch, pitch, Rows, first);
        A::SetFlag<A::HardEvent::V_S>(EVENT_ID0);
        A::WaitFlag<A::HardEvent::V_S>(EVENT_ID0);
        int code = status.GetValue(0);
        if (code != 0) return;
        asc_vf_call<Solve<Rows>>(Ptr(ar), Ptr(ai), Ptr(xr), Ptr(xi), Ptr(inverse), shape.n, pitch, first);
        StoreInverseTile<Rows>(shape, c, batch, first, ar, xr, xi, targets);
    }

   public:
    __aicore__ inline void Run(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ const int* pivots, __gm__ uint8_t* c,
                               __gm__ int* info, __ubuf__ float* memory)
    {
        A::LocalTensor<float> arena(A::TPosition::VECCALC, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)),
                                    208 * 1024 / sizeof(float));
        int pitch = (shape.n + 7) / 8 * 8;
        uint64_t tasks = static_cast<uint64_t>(shape.batch) * ((shape.n + Rows - 1) / Rows);
        for (uint64_t task = A::GetBlockIdx(); task < tasks; task += A::GetBlockNum())
            Matrix(shape, a, pivots, c, info, arena, task, pitch);
    }
};
}  // namespace ComplexLuVector
#endif
