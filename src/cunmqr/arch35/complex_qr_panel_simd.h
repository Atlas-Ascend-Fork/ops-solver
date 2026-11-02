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
#include "kernel_operator.h"

namespace ComplexQrSimd
{
using namespace AscendC;

template <typename UInt, typename Float, typename Mask>
struct ApplyRegisters
{
    UInt lane, offsets, zeros;
    Float zero, one, vr, vi, tmp, other, tr, ti, cr, ci, dotR, dotI, coefR, coefI;
    Mask full, firstLane, unchanged;
};

template <uint32_t FixedChunks, typename State>
__simd_callee__ inline void ApplyDot(State& state, __ubuf__ float* vector, __ubuf__ float* target, uint32_t active)
{
    Reg::Duplicate(state.dotR, 0.0f);
    Reg::Duplicate(state.dotI, 0.0f);
#pragma unroll
    for (uint32_t r = 0; r < (FixedChunks ? FixedChunks * 64 : active); r += 64)
    {
        Reg::RegTensor<uint32_t> absoluteLane;
        Reg::Adds(absoluteLane, state.lane, r, state.full);
        Reg::MaskReg valid;
        Reg::Compares<uint32_t, CMPMODE::LT>(valid, absoluteLane, active, state.full);
        Reg::Duplicate(state.vr, 0.0f);
        Reg::Duplicate(state.vi, 0.0f);
        Reg::Duplicate(state.cr, 0.0f);
        Reg::Duplicate(state.ci, 0.0f);
        Reg::Gather(state.vr, vector + 2 * r, state.offsets, valid);
        Reg::Gather(state.vi, vector + 2 * r + 1, state.offsets, valid);
        Reg::Gather(state.cr, target + 2 * r, state.offsets, valid);
        Reg::Gather(state.ci, target + 2 * r + 1, state.offsets, valid);
        Reg::Select(state.vr, state.vr, state.zero, valid);
        Reg::Select(state.vi, state.vi, state.zero, valid);
        Reg::Select(state.cr, state.cr, state.zero, valid);
        Reg::Select(state.ci, state.ci, state.zero, valid);
        if (r == 0)
        {
            Reg::Select(state.vr, state.one, state.vr, state.firstLane);
            Reg::Select(state.vi, state.zero, state.vi, state.firstLane);
        }
        Reg::Mul(state.tmp, state.vr, state.cr, state.full);
        Reg::Mul(state.other, state.vi, state.ci, state.full);
        Reg::Add(state.tmp, state.tmp, state.other, state.full);
        Reg::Add(state.dotR, state.dotR, state.tmp, state.full);
        Reg::Mul(state.tmp, state.vr, state.ci, state.full);
        Reg::Mul(state.other, state.vi, state.cr, state.full);
        Reg::Sub(state.tmp, state.tmp, state.other, state.full);
        Reg::Add(state.dotI, state.dotI, state.tmp, state.full);
    }
}

template <typename State>
__simd_callee__ inline void ApplyCoefficients(State& state)
{
    Reg::Reduce<Reg::ReduceType::SUM>(state.dotR, state.dotR, state.full);
    Reg::Reduce<Reg::ReduceType::SUM>(state.dotI, state.dotI, state.full);
    Reg::Duplicate(state.dotR, state.dotR, state.full);
    Reg::Duplicate(state.dotI, state.dotI, state.full);
    // The imaginary tau component is negated for H; this applies H^H otherwise.
    Reg::Mul(state.coefR, state.tr, state.dotR, state.full);
    Reg::Mul(state.tmp, state.ti, state.dotI, state.full);
    Reg::Add(state.coefR, state.coefR, state.tmp, state.full);
    Reg::Mul(state.coefI, state.tr, state.dotI, state.full);
    Reg::Mul(state.tmp, state.ti, state.dotR, state.full);
    Reg::Sub(state.coefI, state.coefI, state.tmp, state.full);
}

template <uint32_t FixedChunks, typename State>
__simd_callee__ inline void ApplyUpdate(State& state, __ubuf__ float* vector, __ubuf__ float* target, uint32_t active)
{
#pragma unroll
    for (uint32_t r = 0; r < (FixedChunks ? FixedChunks * 64 : active); r += 64)
    {
        Reg::RegTensor<uint32_t> absoluteLane;
        Reg::Adds(absoluteLane, state.lane, r, state.full);
        Reg::MaskReg valid;
        Reg::Compares<uint32_t, CMPMODE::LT>(valid, absoluteLane, active, state.full);
        Reg::Gather(state.vr, vector + 2 * r, state.offsets, valid);
        Reg::Gather(state.vi, vector + 2 * r + 1, state.offsets, valid);
        Reg::Gather(state.cr, target + 2 * r, state.offsets, valid);
        Reg::Gather(state.ci, target + 2 * r + 1, state.offsets, valid);
        if (r == 0)
        {
            Reg::Select(state.vr, state.one, state.vr, state.firstLane);
            Reg::Select(state.vi, state.zero, state.vi, state.firstLane);
        }
        Reg::Mul(state.tmp, state.vr, state.coefR, state.full);
        Reg::Mul(state.other, state.vi, state.coefI, state.full);
        Reg::Sub(state.tmp, state.tmp, state.other, state.full);
        Reg::Sub(state.tmp, state.cr, state.tmp, state.full);
        Reg::Select(state.tmp, state.cr, state.tmp, state.unchanged);
        Reg::Scatter(target + 2 * r, state.tmp, state.offsets, valid);
        Reg::Mul(state.tmp, state.vr, state.coefI, state.full);
        Reg::Mul(state.other, state.vi, state.coefR, state.full);
        Reg::Add(state.tmp, state.tmp, state.other, state.full);
        Reg::Sub(state.tmp, state.ci, state.tmp, state.full);
        Reg::Select(state.tmp, state.ci, state.tmp, state.unchanged);
        Reg::Scatter(target + 2 * r + 1, state.tmp, state.offsets, valid);
    }
}

template <uint32_t FixedChunks>
__simd_vf__ inline void Apply(__ubuf__ float* matrix, __ubuf__ float* tau, uint32_t rows, uint32_t stride, uint32_t k,
                              __ubuf__ float* targetBase, bool adjoint)
{
    ApplyRegisters<Reg::RegTensor<uint32_t>, Reg::RegTensor<float>, Reg::MaskReg> state;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(state.lane), int32_t(0));
    state.full = Reg::CreateMask<float>();
    uint32_t oneCount = 1;
    state.firstLane = Reg::UpdateMask<float>(oneCount);
    Reg::Muls(state.offsets, state.lane, uint32_t(2), state.full);
    Reg::Duplicate(state.zeros, uint32_t(0));
    Reg::Duplicate(state.zero, 0.0f);
    Reg::Duplicate(state.one, 1.0f);
    Reg::Gather(state.tr, tau + 2 * k, state.zeros, state.full);
    Reg::Gather(state.ti, tau + 2 * k + 1, state.zeros, state.full);
    if (!adjoint)
    {
        Reg::Muls(state.ti, state.ti, -1.0f, state.full);
    }
    Reg::MaskReg realZero, imagZero;
    Reg::Compares<float, CMPMODE::EQ>(realZero, state.tr, 0.0f, state.full);
    Reg::Compares<float, CMPMODE::EQ>(imagZero, state.ti, 0.0f, state.full);
    Reg::And(state.unchanged, realZero, imagZero, state.full);
    __ubuf__ float* vector = matrix + 2 * (k + k * stride);
    const uint32_t active = rows - k;
    auto target = targetBase + 2 * k;
    ApplyDot<FixedChunks>(state, vector, target, active);
    ApplyCoefficients(state);
    ApplyUpdate<FixedChunks>(state, vector, target, active);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}
}  // namespace ComplexQrSimd
