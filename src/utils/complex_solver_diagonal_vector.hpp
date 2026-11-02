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
namespace ComplexDiagonalVector
{
namespace A = AscendC;
constexpr int ROWS = COMPLEX_SOLVER_INVERSE_PANEL;
constexpr int MAX_COLUMNS = COMPLEX_SOLVER_INVERSE_COLUMNS;
constexpr int PLANE = ROWS * MAX_COLUMNS;
constexpr int MEMORY_FLOATS = 4 * PLANE + 2 * ROWS * ROWS + 2 * ROWS + 192;
__simd_vf__ inline void Prepare(__ubuf__ float* stage, __ubuf__ float* real, __ubuf__ float* imag, int count, int live)
{
    A::Reg::RegTensor<float> r, i;
    uint32_t remaining = live;
    auto mask = A::Reg::UpdateMask<float>(remaining);
    for (uint16_t col = 0; col < count; ++col)
    {
        A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_DINTLV_B32>(r, i, stage + col * 2 * MAX_COLUMNS);
        A::Reg::StoreAlign(real + col * 64, r, mask);
        A::Reg::StoreAlign(imag + col * 64, i, mask);
    }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(128) inline void PrepareDiagonal(__ubuf__ const float* stage,
                                                                     __ubuf__ float* diagonal, __ubuf__ float* inverse,
                                                                     int count, bool upper)
{
    for (int index = threadIdx.x; index < ROWS * ROWS; index += blockDim.x)
    {
        int row = index % ROWS, col = index / ROWS, target = index;
        diagonal[target] = row < count && col < count ? -stage[col * 2 * MAX_COLUMNS + 2 * row] : 0;
        diagonal[ROWS * ROWS + target] = row < count && col < count ? -stage[col * 2 * MAX_COLUMNS + 2 * row + 1] : 0;
    }
    if (upper && threadIdx.x < count)
    {
        int p = threadIdx.x, offset = p * 2 * MAX_COLUMNS + 2 * p;
        auto z = ComplexSolverKernel::PreparedDivide(
            {1, 0}, ComplexSolverKernel::PrepareDivisor({stage[offset], stage[offset + 1]}));
        inverse[p] = z.real;
        inverse[ROWS + p] = z.imag;
    }
}

template <bool Upper>
__simd_vf__ inline void Solve(__ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* diagonal,
                              __ubuf__ float* inverse, int count, int live)
{
    A::Reg::RegTensor<float> pr, pi, rr, ri, dr, di, negative, fr, fi;
    uint32_t remaining = live;
    auto mask = A::Reg::UpdateMask<float>(remaining);
    for (uint16_t step = 0; step < count; ++step)
    {
        uint16_t p = Upper ? step : count - 1 - step;
        uint16_t offset = p * 64;
        A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
        A::Reg::LoadAlign(pr, real + offset);
        A::Reg::LoadAlign(pi, imag + offset);
        if constexpr (Upper)
        {
            A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(dr, inverse + p);
            A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(di, inverse + ROWS + p);
            rr = pr;
            ri = pi;
            A::Reg::Mul(pr, rr, dr, mask);
            A::Reg::Muls(negative, ri, -1.0f, mask);
            A::Reg::MulAddDst(pr, negative, di, mask);
            A::Reg::Mul(pi, ri, dr, mask);
            A::Reg::MulAddDst(pi, rr, di, mask);
            A::Reg::StoreAlign(real + offset, pr, mask);
            A::Reg::StoreAlign(imag + offset, pi, mask);
        }
        A::Reg::Muls(negative, pi, -1.0f, mask);
        for (uint16_t col = Upper ? p + 1 : 0; col < (Upper ? count : p); ++col)
        {
            A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(fr, diagonal + col * ROWS + p);
            A::Reg::LoadAlign<float, A::Reg::LoadDist::DIST_BRC_B32>(fi, diagonal + ROWS * ROWS + col * ROWS + p);
            A::Reg::LoadAlign(rr, real + col * 64);
            A::Reg::LoadAlign(ri, imag + col * 64);
            A::Reg::MulAddDst(rr, pr, fr, mask);
            A::Reg::MulAddDst(ri, pr, fi, mask);
            A::Reg::MulAddDst(rr, negative, fi, mask);
            A::Reg::MulAddDst(ri, pi, fr, mask);
            A::Reg::StoreAlign(real + col * 64, rr, mask);
            A::Reg::StoreAlign(imag + col * 64, ri, mask);
        }
    }
}

__simd_vf__ inline void Pack(__ubuf__ float* stage, __ubuf__ const float* real, __ubuf__ const float* imag, int count,
                             int live)
{
    A::Reg::RegTensor<float> r, i, lo, hi;
    auto mask = A::Reg::CreateMask<float, A::Reg::MaskPattern::ALL>();
    for (uint16_t col = 0; col < count; ++col)
    {
        A::Reg::LoadAlign(r, const_cast<__ubuf__ float*>(real) + col * 64);
        A::Reg::LoadAlign(i, const_cast<__ubuf__ float*>(imag) + col * 64);
        A::Reg::Interleave(lo, hi, r, i);
        A::Reg::StoreAlign(stage + col * 2 * MAX_COLUMNS, lo, mask);
        A::Reg::StoreAlign(stage + col * 2 * MAX_COLUMNS + 64, hi, mask);
    }
}

class Inverse
{
    __aicore__ inline __ubuf__ float* Ptr(A::LocalTensor<float> x)
    {
        return reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr());
    }
    __aicore__ inline void Transfer(A::LocalTensor<float> stage, A::GlobalTensor<float> gm, int count, int live,
                                    int ldc, bool load)
    {
        int gap = (MAX_COLUMNS - (live + 3) / 4 * 4) / 4;
        if (load)
        {
            A::DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(live * 8),
                                    static_cast<uint32_t>((ldc - live) * 8), static_cast<uint32_t>(gap), 0};
            A::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
            A::SetFlag<A::HardEvent::V_MTE2>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::V_MTE2>(EVENT_ID0);
            A::DataCopyPad(stage, gm, cp, pad);
            A::SetFlag<A::HardEvent::MTE2_V>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::MTE2_V>(EVENT_ID0);
        }
        else
        {
            A::DataCopyExtParams cp{static_cast<uint16_t>(count), static_cast<uint32_t>(live * 8),
                                    static_cast<uint32_t>(gap), static_cast<uint32_t>((ldc - live) * 8), 0};
            A::SetFlag<A::HardEvent::V_MTE3>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::V_MTE3>(EVENT_ID0);
            A::DataCopyPad(gm, stage, cp);
            A::SetFlag<A::HardEvent::MTE3_V>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

   public:
    __aicore__ inline void Run(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* c, __gm__ const int* info,
                               int begin, int end, bool upper, bool batched, __ubuf__ float* memory)
    {
        A::LocalTensor<float> arena(A::TPosition::VECCALC, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)),
                                    MEMORY_FLOATS);
        auto stage = arena, real = arena[2 * PLANE], imag = arena[3 * PLANE + 64], diagonal = arena[4 * PLANE + 128],
             inverse = arena[4 * PLANE + 2 * ROWS * ROWS + 192];
        int height = upper ? end : shape.n, tiles = (height + MAX_COLUMNS - 1) / MAX_COLUMNS;
        for (uint64_t task = A::GetBlockIdx(); task < static_cast<uint64_t>(shape.batch) * tiles;
             task += A::GetBlockNum())
        {
            int batch = task / tiles, first = (task % tiles) * MAX_COLUMNS;
            if (info[batch] != 0) continue;
            int live = height - first < MAX_COLUMNS ? height - first : MAX_COLUMNS;
            auto* input =
                batched ? ComplexSolverKernel::MatrixPointer(a, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(a);
            auto* output =
                batched ? ComplexSolverKernel::MatrixPointer(c, batch) : reinterpret_cast<__gm__ aclFloatComplex*>(c);
            A::GlobalTensor<float> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output) +
                               (static_cast<uint64_t>(begin) * shape.ldc + first) * 2);
            A::GlobalTensor<float> factor;
            factor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input) +
                                   (static_cast<uint64_t>(begin) * shape.lda + begin) * 2);
            Transfer(stage, factor, end - begin, end - begin, shape.lda, true);
            asc_vf_call<PrepareDiagonal>(dim3{128, 1, 1}, Ptr(stage), Ptr(diagonal), Ptr(inverse), end - begin, upper);
            Transfer(stage, gm, end - begin, live, shape.ldc, true);
            asc_vf_call<Prepare>(Ptr(stage), Ptr(real), Ptr(imag), end - begin, live);
            if (upper)
                asc_vf_call<Solve<true>>(Ptr(real), Ptr(imag), Ptr(diagonal), Ptr(inverse), end - begin, live);
            else
                asc_vf_call<Solve<false>>(Ptr(real), Ptr(imag), Ptr(diagonal), Ptr(inverse), end - begin, live);
            asc_vf_call<Pack>(Ptr(stage), Ptr(real), Ptr(imag), end - begin, live);
            Transfer(stage, gm, end - begin, live, shape.ldc, false);
        }
    }
};
}  // namespace ComplexDiagonalVector
#endif
