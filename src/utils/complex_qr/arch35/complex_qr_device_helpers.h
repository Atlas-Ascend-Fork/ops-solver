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
#include "../complex_qr.h"
#include "complex_qr_transfer.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

namespace
{
using AscendC::GetBlockIdx;
using AscendC::GetBlockNum;
struct Complex
{
    float r;
    float i;
};
__simt_callee__ inline Complex Load(__gm__ const float* p, uint64_t j) { return {p[2 * j], p[2 * j + 1]}; }

__simt_callee__ inline void Store(__gm__ float* p, uint64_t j, Complex z)
{
    p[2 * j] = z.r;
    p[2 * j + 1] = z.i;
}

__simt_callee__ inline Complex Load(__ubuf__ const float* p, uint64_t j) { return {p[2 * j], p[2 * j + 1]}; }

__simt_callee__ inline void Store(__ubuf__ float* p, uint64_t j, Complex z)
{
    p[2 * j] = z.r;
    p[2 * j + 1] = z.i;
}

__simt_callee__ inline Complex Add(Complex a, Complex b) { return {a.r + b.r, a.i + b.i}; }

__simt_callee__ inline Complex Sub(Complex a, Complex b) { return {a.r - b.r, a.i - b.i}; }

__simt_callee__ inline Complex Conj(Complex a) { return {a.r, -a.i}; }

__simt_callee__ inline Complex Mul(Complex a, Complex b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }

__simt_callee__ inline Complex Divide(Complex a, Complex b)
{
    // Scale both operands to avoid squaring an extreme denominator.
    float s = fmaxf(fabsf(b.r), fabsf(b.i));
    // Callers handle zero reflectors and singular diagonals before division.
    // Keep an invalid denominator explicit if that precondition is violated.
    if (s == 0.0f)
    {
        const float invalid = __builtin_nanf("");
        return {invalid, invalid};
    }
    float br = b.r / s, bi = b.i / s;
    float ar = a.r / s, ai = a.i / s;
    float d = br * br + bi * bi;
    return {(ar * br + ai * bi) / d, (ai * br - ar * bi) / d};
}

__simt_callee__ inline float Sum(float v)
{
    for (int d = 16; d > 0; d /= 2)
    {
        v += asc_shfl_down(v, d);
    }
    return asc_shfl(v, 0);
}

}  // namespace
