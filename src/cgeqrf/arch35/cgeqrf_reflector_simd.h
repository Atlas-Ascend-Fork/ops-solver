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
struct ReflectorRegisters
{
    UInt lane, offsets, zeros, signMask, signs, bits;
    Float zero, one, scale, vr, vi, tmp, other, ar, ai, alphaR, alphaI, tail, beta, safeBeta, tr, ti, dr, di, ds,
        denominator;
    Mask full, firstLane, unchanged;
};

template <typename State>
__simd_callee__ inline void LoadReflectorChunk(State& state, __ubuf__ float* vector, uint32_t r, uint32_t active)
{
    Reg::RegTensor<uint32_t> absoluteLane;
    Reg::Adds(absoluteLane, state.lane, r, state.full);
    Reg::MaskReg valid;
    Reg::Compares<uint32_t, CMPMODE::LT>(valid, absoluteLane, active, state.full);
    Reg::Duplicate(state.vr, 0.0f);
    Reg::Duplicate(state.vi, 0.0f);
    Reg::Gather(state.vr, vector + 2 * r, state.offsets, valid);
    Reg::Gather(state.vi, vector + 2 * r + 1, state.offsets, valid);
    Reg::Select(state.vr, state.vr, state.zero, valid);
    Reg::Select(state.vi, state.vi, state.zero, valid);
}

template <uint32_t FixedChunks, typename State>
__simd_callee__ inline void ReflectorScale(State& state, __ubuf__ float* vector, uint32_t active)
{
    Reg::Duplicate(state.scale, 0.0f);
#pragma unroll
    for (uint32_t r = 0; r < (FixedChunks ? FixedChunks * 64 : active); r += 64)
    {
        LoadReflectorChunk(state, vector, r, active);
        Reg::Abs(state.vr, state.vr, state.full);
        Reg::Abs(state.vi, state.vi, state.full);
        Reg::Max(state.tmp, state.vr, state.vi, state.full);
        Reg::Max(state.scale, state.scale, state.tmp, state.full);
    }
    Reg::Reduce<Reg::ReduceType::MAX>(state.scale, state.scale, state.full);
    Reg::Duplicate(state.scale, state.scale, state.full);
    Reg::MaskReg scaleZero;
    Reg::Compares<float, CMPMODE::EQ>(scaleZero, state.scale, 0.0f, state.full);
    Reg::Select(state.scale, state.one, state.scale, scaleZero);
}

template <uint32_t FixedChunks, typename State>
__simd_callee__ inline void ReflectorTail(State& state, __ubuf__ float* vector, uint32_t active)
{
    Reg::Gather(state.alphaR, vector, state.zeros, state.full);
    Reg::Gather(state.alphaI, vector + 1, state.zeros, state.full);
    Reg::Div(state.ar, state.alphaR, state.scale, state.full);
    Reg::Div(state.ai, state.alphaI, state.scale, state.full);
    Reg::Duplicate(state.tail, 0.0f);
#pragma unroll
    for (uint32_t r = 0; r < (FixedChunks ? FixedChunks * 64 : active); r += 64)
    {
        LoadReflectorChunk(state, vector, r, active);
        if (r == 0)
        {
            Reg::Select(state.vr, state.zero, state.vr, state.firstLane);
            Reg::Select(state.vi, state.zero, state.vi, state.firstLane);
        }
        Reg::Div(state.vr, state.vr, state.scale, state.full);
        Reg::Div(state.vi, state.vi, state.scale, state.full);
        Reg::Mul(state.tmp, state.vr, state.vr, state.full);
        Reg::Mul(state.other, state.vi, state.vi, state.full);
        Reg::Add(state.tmp, state.tmp, state.other, state.full);
        Reg::Add(state.tail, state.tail, state.tmp, state.full);
    }
    Reg::Reduce<Reg::ReduceType::SUM>(state.tail, state.tail, state.full);
    Reg::Duplicate(state.tail, state.tail, state.full);
}

template <typename State>
__simd_callee__ inline void ReflectorCoefficients(State& state, __ubuf__ float* tau, uint32_t k)
{
    Reg::MaskReg tailZero, imagZero;
    Reg::Compares<float, CMPMODE::EQ>(tailZero, state.tail, 0.0f, state.full);
    Reg::Compares<float, CMPMODE::EQ>(imagZero, state.alphaI, 0.0f, state.full);
    Reg::And(state.unchanged, tailZero, imagZero, state.full);
    Reg::Mul(state.tmp, state.ar, state.ar, state.full);
    Reg::Mul(state.other, state.ai, state.ai, state.full);
    Reg::Add(state.tmp, state.tmp, state.other, state.full);
    Reg::Add(state.tmp, state.tmp, state.tail, state.full);
    Reg::Sqrt(state.beta, state.tmp, state.full);
    // -copysign(norm, alpha.real), including the sign bit of negative zero.
    Reg::And(state.signs, reinterpret_cast<Reg::RegTensor<uint32_t>&>(state.alphaR), state.signMask, state.full);
    Reg::Xor(state.signs, state.signs, state.signMask, state.full);
    Reg::Xor(state.bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(state.beta), state.signs, state.full);
    state.beta = reinterpret_cast<Reg::RegTensor<float>&>(state.bits);
    Reg::Select(state.safeBeta, state.one, state.beta, state.unchanged);
    Reg::Sub(state.tr, state.beta, state.ar, state.full);
    Reg::Div(state.tr, state.tr, state.safeBeta, state.full);
    Reg::Div(state.ti, state.ai, state.safeBeta, state.full);
    Reg::Muls(state.ti, state.ti, -1.0f, state.full);
    Reg::Select(state.tr, state.zero, state.tr, state.unchanged);
    Reg::Select(state.ti, state.zero, state.ti, state.unchanged);
    Reg::Scatter(tau + 2 * k, state.tr, state.zeros, state.firstLane);
    Reg::Scatter(tau + 2 * k + 1, state.ti, state.zeros, state.firstLane);
    Reg::Sub(state.dr, state.ar, state.beta, state.full);
    state.di = state.ai;
    Reg::Abs(state.tmp, state.dr, state.full);
    Reg::Abs(state.other, state.di, state.full);
    Reg::Max(state.ds, state.tmp, state.other, state.full);
    Reg::Select(state.ds, state.one, state.ds, state.unchanged);
    Reg::Div(state.dr, state.dr, state.ds, state.full);
    Reg::Div(state.di, state.di, state.ds, state.full);
    Reg::Mul(state.tmp, state.dr, state.dr, state.full);
    Reg::Mul(state.other, state.di, state.di, state.full);
    Reg::Add(state.denominator, state.tmp, state.other, state.full);
    Reg::Select(state.denominator, state.one, state.denominator, state.unchanged);
}

template <uint32_t FixedChunks, typename State>
__simd_callee__ inline void StoreReflector(State& state, __ubuf__ float* vector, uint32_t active)
{
#pragma unroll
    for (uint32_t r = 0; r < (FixedChunks ? FixedChunks * 64 : active); r += 64)
    {
        Reg::RegTensor<uint32_t> absoluteLane;
        Reg::Adds(absoluteLane, state.lane, r, state.full);
        Reg::MaskReg valid;
        Reg::Compares<uint32_t, CMPMODE::LT>(valid, absoluteLane, active, state.full);
        if (r == 0)
        {
            Reg::MaskReg tailLanes;
            Reg::Compares<uint32_t, CMPMODE::GT>(tailLanes, state.lane, uint32_t(0), state.full);
            Reg::And(valid, valid, tailLanes, state.full);
        }
        Reg::Gather(state.vr, vector + 2 * r, state.offsets, valid);
        Reg::Gather(state.vi, vector + 2 * r + 1, state.offsets, valid);
        Reg::RegTensor<float> zr, zi, resultR, resultI;
        Reg::Div(zr, state.vr, state.scale, state.full);
        Reg::Div(zi, state.vi, state.scale, state.full);
        Reg::Div(zr, zr, state.ds, state.full);
        Reg::Div(zi, zi, state.ds, state.full);
        Reg::Mul(resultR, zr, state.dr, state.full);
        Reg::Mul(state.tmp, zi, state.di, state.full);
        Reg::Add(resultR, resultR, state.tmp, state.full);
        Reg::Div(resultR, resultR, state.denominator, state.full);
        Reg::Mul(resultI, zi, state.dr, state.full);
        Reg::Mul(state.tmp, zr, state.di, state.full);
        Reg::Sub(resultI, resultI, state.tmp, state.full);
        Reg::Div(resultI, resultI, state.denominator, state.full);
        Reg::Select(resultR, state.vr, resultR, state.unchanged);
        Reg::Select(resultI, state.vi, resultI, state.unchanged);
        Reg::Scatter(vector + 2 * r, resultR, state.offsets, valid);
        Reg::Scatter(vector + 2 * r + 1, resultI, state.offsets, valid);
    }
    Reg::Mul(state.beta, state.beta, state.scale, state.full);
    Reg::Select(state.beta, state.alphaR, state.beta, state.unchanged);
    Reg::Select(state.tmp, state.alphaI, state.zero, state.unchanged);
    Reg::Scatter(vector, state.beta, state.zeros, state.firstLane);
    Reg::Scatter(vector + 1, state.tmp, state.zeros, state.firstLane);
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

// Column-major UB panel; masked gathers never access rounded-up elements.
template <uint32_t FixedChunks = 0>
__simd_vf__ inline void Reflector(__ubuf__ float* matrix, __ubuf__ float* tau, uint32_t rows, uint32_t stride,
                                  uint32_t k)
{
    ReflectorRegisters<Reg::RegTensor<uint32_t>, Reg::RegTensor<float>, Reg::MaskReg> state;
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(state.lane), int32_t(0));
    state.full = Reg::CreateMask<float>();
    uint32_t oneCount = 1;
    state.firstLane = Reg::UpdateMask<float>(oneCount);
    Reg::Muls(state.offsets, state.lane, uint32_t(2), state.full);
    Reg::Duplicate(state.zeros, uint32_t(0));
    Reg::Duplicate(state.signMask, uint32_t(0x80000000));
    Reg::Duplicate(state.zero, 0.0f);
    Reg::Duplicate(state.one, 1.0f);
    auto vector = matrix + 2 * (k + k * stride);
    const uint32_t active = rows - k;
    ReflectorScale<FixedChunks>(state, vector, active);
    ReflectorTail<FixedChunks>(state, vector, active);
    ReflectorCoefficients(state, tau, k);
    StoreReflector<FixedChunks>(state, vector, active);
}

}  // namespace ComplexQrSimd
