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

#include "../../utils/complex_qr/complex_qr.h"
#include "kernel_operator.h"

namespace ComplexQrTransform
{
using namespace AscendC;
constexpr uint32_t Pitch = COMPLEX_QR_PANEL + 4;
constexpr uint32_t Plane = COMPLEX_QR_PANEL * 8;

template <typename UInt, typename Float, typename Mask>
struct TransformRegisters
{
    UInt lane, rows, index, repeated, bits, maskBits;
    Float zero, one, wr, wi, gr, gi, cr, ci, tr, ti, vr, vi, p, q, nextR, nextI;
    Mask full;
};

template <typename State>
__simd_callee__ inline void InitializeWeights(State& state, __ubuf__ float* weights, uint32_t panel, uint32_t actual,
                                              uint32_t column)
{
    state.full = Reg::CreateMask<float>();
    Reg::Arange(reinterpret_cast<Reg::RegTensor<int32_t>&>(state.lane), int32_t(0));
    Reg::Duplicate(state.zero, 0.0f);
    Reg::Duplicate(state.one, column < actual ? 1.0f : 0.0f);
    Reg::Duplicate(state.maskBits, 0xffffe000U);
    for (uint32_t offset = 0; offset < panel; offset += 64)
    {
        Reg::Adds(state.rows, state.lane, offset, state.full);
        Reg::MaskReg diagonal;
        Reg::Compares<uint32_t, CMPMODE::EQ>(diagonal, state.rows, column, state.full);
        Reg::Select(state.wr, state.one, state.zero, diagonal);
        Reg::StoreAlign(weights + offset, state.wr, state.full);
        Reg::StoreAlign(weights + COMPLEX_QR_PANEL + offset, state.zero, state.full);
    }
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
}

template <typename State>
__simd_callee__ inline void UpdateWeights(State& state, __ubuf__ float* gram, __ubuf__ float* weights, uint32_t panel,
                                          uint32_t actual, uint32_t i, bool adjoint)
{
    for (uint32_t offset = 0; offset < panel; offset += 64)
    {
        Reg::Adds(state.rows, state.lane, offset, state.full);
        Reg::Muls(state.index, state.rows, uint32_t(2), state.full);
        Reg::Gather(state.gr, gram + 2 * i * Pitch, state.index, state.full);
        Reg::Gather(state.gi, gram + 2 * i * Pitch + 1, state.index, state.full);
        Reg::LoadAlign(state.wr, weights + offset);
        Reg::LoadAlign(state.wi, weights + COMPLEX_QR_PANEL + offset);
        Reg::Mul(state.p, state.gr, state.vr, state.full);
        Reg::Mul(state.q, state.gi, state.vi, state.full);
        Reg::Sub(state.nextR, state.p, state.q, state.full);
        Reg::Sub(state.nextR, state.wr, state.nextR, state.full);
        Reg::Mul(state.p, state.gr, state.vi, state.full);
        Reg::Mul(state.q, state.gi, state.vr, state.full);
        Reg::Add(state.nextI, state.p, state.q, state.full);
        Reg::Sub(state.nextI, state.wi, state.nextI, state.full);
        Reg::MaskReg active, triangular, diagonal, update;
        Reg::Compares<uint32_t, CMPMODE::LT>(active, state.rows, actual, state.full);
        if (adjoint)
        {
            Reg::Compares<uint32_t, CMPMODE::GT>(triangular, state.rows, i, state.full);
        }
        else
        {
            Reg::Compares<uint32_t, CMPMODE::LT>(triangular, state.rows, i, state.full);
        }
        Reg::And(update, active, triangular, state.full);
        Reg::Select(state.wr, state.nextR, state.wr, update);
        Reg::Select(state.wi, state.nextI, state.wi, update);
        Reg::Compares<uint32_t, CMPMODE::EQ>(diagonal, state.rows, i, state.full);
        Reg::Select(state.wr, state.vr, state.wr, diagonal);
        Reg::Select(state.wi, state.vi, state.wi, diagonal);
        Reg::StoreAlign(weights + offset, state.wr, state.full);
        Reg::StoreAlign(weights + COMPLEX_QR_PANEL + offset, state.wi, state.full);
    }
}

template <typename State>
__simd_callee__ inline void TransformWeights(State& state, __ubuf__ float* gram, __ubuf__ float* tau,
                                             __ubuf__ float* weights, uint32_t panel, uint32_t actual, uint32_t column,
                                             bool adjoint)
{
    const uint32_t steps = column >= actual ? 0 : adjoint ? actual - column : column + 1;
    for (uint32_t step = 0; step < steps; ++step)
    {
        uint32_t i = adjoint ? column + step : column - step;
        Reg::Duplicate(state.repeated, i);
        Reg::Gather(state.cr, weights, state.repeated, state.full);
        Reg::Gather(state.ci, weights + COMPLEX_QR_PANEL, state.repeated, state.full);
        Reg::Duplicate(state.repeated, 2 * i);
        Reg::Gather(state.tr, tau, state.repeated, state.full);
        Reg::Gather(state.ti, tau + 1, state.repeated, state.full);
        if (adjoint)
        {
            Reg::Muls(state.ti, state.ti, -1.0f, state.full);
        }
        Reg::Mul(state.p, state.tr, state.cr, state.full);
        Reg::Mul(state.q, state.ti, state.ci, state.full);
        Reg::Sub(state.vr, state.p, state.q, state.full);
        Reg::Mul(state.p, state.tr, state.ci, state.full);
        Reg::Mul(state.q, state.ti, state.cr, state.full);
        Reg::Add(state.vi, state.p, state.q, state.full);
        UpdateWeights(state, gram, weights, panel, actual, i, adjoint);
        Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
    }
}

template <typename State>
__simd_callee__ inline void StoreWeights(State& state, __ubuf__ float* weights, __ubuf__ float* high,
                                         __ubuf__ float* low, uint32_t panel)
{
    for (uint32_t offset = 0; offset < panel; offset += 64)
    {
        Reg::LoadAlign(state.wr, weights + offset);
        Reg::LoadAlign(state.wi, weights + COMPLEX_QR_PANEL + offset);
        Reg::Adds(state.rows, state.lane, offset, state.full);
        Reg::Muls(state.index, state.rows, uint32_t(8), state.full);
        Reg::And(state.bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(state.wr), state.maskBits, state.full);
        state.p = reinterpret_cast<Reg::RegTensor<float>&>(state.bits);
        Reg::Sub(state.q, state.wr, state.p, state.full);
        Reg::Scatter(high, state.p, state.index, state.full);
        Reg::Scatter(low, state.q, state.index, state.full);
        Reg::Scatter(high + 3 * Plane, state.p, state.index, state.full);
        Reg::Scatter(low + 3 * Plane, state.q, state.index, state.full);
        Reg::And(state.bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(state.wi), state.maskBits, state.full);
        state.p = reinterpret_cast<Reg::RegTensor<float>&>(state.bits);
        Reg::Sub(state.q, state.wi, state.p, state.full);
        Reg::Scatter(high + 2 * Plane, state.p, state.index, state.full);
        Reg::Scatter(low + 2 * Plane, state.q, state.index, state.full);
        Reg::Muls(state.wi, state.wi, -1.0f, state.full);
        Reg::And(state.bits, reinterpret_cast<Reg::RegTensor<uint32_t>&>(state.wi), state.maskBits, state.full);
        state.p = reinterpret_cast<Reg::RegTensor<float>&>(state.bits);
        Reg::Sub(state.q, state.wi, state.p, state.full);
        Reg::Scatter(high + Plane, state.p, state.index, state.full);
        Reg::Scatter(low + Plane, state.q, state.index, state.full);
    }
}

__simd_vf__ inline void FormColumn(__ubuf__ float* gram, __ubuf__ float* tau, __ubuf__ float* weights,
                                   __ubuf__ float* high, __ubuf__ float* low, uint32_t panel, uint32_t actual,
                                   uint32_t column, bool adjoint)
{
    TransformRegisters<Reg::RegTensor<uint32_t>, Reg::RegTensor<float>, Reg::MaskReg> state;
    InitializeWeights(state, weights, panel, actual, column);
    TransformWeights(state, gram, tau, weights, panel, actual, column, adjoint);
    StoreWeights(state, weights, high, low, panel);
}
}  // namespace ComplexQrTransform
