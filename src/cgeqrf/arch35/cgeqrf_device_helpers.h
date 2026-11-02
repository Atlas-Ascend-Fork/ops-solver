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

#include "../../utils/complex_qr/arch35/complex_qr_device_helpers.h"

namespace
{
#ifndef COMPLEX_QR_COOPERATIVE_THREADS
#define COMPLEX_QR_COOPERATIVE_THREADS 512
#endif
constexpr uint32_t kCooperativeThreads = COMPLEX_QR_COOPERATIVE_THREADS;
constexpr uint32_t kCooperativeWarps = kCooperativeThreads / 32;

template <class Pointer, class TauPointer>
__simt_callee__ inline void ReflectorBody(Pointer a, TauPointer tau, uint32_t m, uint64_t lda, uint32_t k)
{
    float scale = 0;
    for (uint32_t r = k + threadIdx.x; r < m; r += 32)
    {
        Complex z = Load(a, r + uint64_t(k) * lda);
        scale = fmaxf(scale, fmaxf(fabsf(z.r), fabsf(z.i)));
    }
    for (int d = 16; d > 0; d /= 2)
    {
        scale = fmaxf(scale, asc_shfl_down(scale, d));
    }
    scale = asc_shfl(scale, 0);
    Complex alpha = Load(a, k + uint64_t(k) * lda);
    float tail = 0;
    if (scale != 0)
    {
        for (uint32_t r = k + 1 + threadIdx.x; r < m; r += 32)
        {
            Complex z = Load(a, r + uint64_t(k) * lda);
            z.r /= scale;
            z.i /= scale;
            tail += z.r * z.r + z.i * z.i;
        }
    }
    tail = Sum(tail);
    if (tail == 0 && alpha.i == 0)
    {
        if (threadIdx.x == 0)
        {
            Store(tau, k, {0, 0});
        }
        return;
    }
    float ar = alpha.r / scale, ai = alpha.i / scale;
    float beta = -copysignf(sqrtf(ar * ar + ai * ai + tail), alpha.r);
    Complex t = {(beta - ar) / beta, -ai / beta};
    Complex denominator = {ar - beta, ai};
    for (uint32_t r = k + 1 + threadIdx.x; r < m; r += 32)
    {
        Complex z = Load(a, r + uint64_t(k) * lda);
        Store(a, r + uint64_t(k) * lda, Divide({z.r / scale, z.i / scale}, denominator));
    }
    if (threadIdx.x == 0)
    {
        Store(tau, k, t);
        Store(a, k + uint64_t(k) * lda, {beta * scale, 0});
    }
}

__simt_callee__ inline void CooperativeApply(__ubuf__ float* vectors, __ubuf__ float* tau, __ubuf__ float* target,
                                             __ubuf__ float* scratch, uint32_t rows, uint32_t stride, uint32_t k,
                                             uint32_t column, bool adjoint = true)
{
    Complex dot = {0, 0};
    for (uint32_t r = k + threadIdx.x; r < rows; r += kCooperativeThreads)
    {
        Complex v = r == k ? Complex{1, 0} : Load(vectors, r + uint64_t(k) * stride);
        dot = Add(dot, Mul(Conj(v), Load(target, r + uint64_t(column) * stride)));
    }
    dot = {Sum(dot.r), Sum(dot.i)};
    if (threadIdx.x % 32 == 0)
    {
        Store(scratch, threadIdx.x / 32, dot);
    }
    asc_syncthreads();
    if (threadIdx.x < 32)
    {
        Complex partial = threadIdx.x < kCooperativeWarps ? Load(scratch, threadIdx.x) : Complex{0, 0};
        partial = {Sum(partial.r), Sum(partial.i)};
        if (threadIdx.x == 0)
        {
            Store(scratch, 16, partial);
        }
    }
    asc_syncthreads();
    Complex coefficient = Load(tau, k);
    dot = Mul(adjoint ? Conj(coefficient) : coefficient, Load(scratch, 16));
    for (uint32_t r = k + threadIdx.x; r < rows; r += kCooperativeThreads)
    {
        Complex v = r == k ? Complex{1, 0} : Load(vectors, r + uint64_t(k) * stride);
        uint64_t index = r + uint64_t(column) * stride;
        Store(target, index, Sub(Load(target, index), Mul(v, dot)));
    }
    asc_syncthreads();
}
}  // namespace
