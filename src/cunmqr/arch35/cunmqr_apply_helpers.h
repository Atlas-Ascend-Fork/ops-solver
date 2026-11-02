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
template <class APointer, class TauPointer, class CPointer>
__simt_callee__ inline void ApplyBody(APointer a, TauPointer tau, CPointer c, uint32_t order, uint64_t lda,
                                      uint64_t ldc, uint32_t k, uint32_t vector, bool right, bool adjoint)
{
    Complex t = Load(tau, k);
    if (adjoint)
    {
        t = Conj(t);
    }
    if (t.r == 0 && t.i == 0)
    {
        return;
    }
    Complex dot = {0, 0};
    for (uint32_t r = k + threadIdx.x % 32; r < order; r += 32)
    {
        Complex v = r == k ? Complex{1, 0} : Load(a, r + uint64_t(k) * lda);
        Complex z = Load(c, right ? vector + uint64_t(r) * ldc : r + uint64_t(vector) * ldc);
        dot = Add(dot, Mul(right ? v : Conj(v), z));
    }
    dot = {Sum(dot.r), Sum(dot.i)};
    dot = Mul(t, dot);
    for (uint32_t r = k + threadIdx.x % 32; r < order; r += 32)
    {
        Complex v = r == k ? Complex{1, 0} : Load(a, r + uint64_t(k) * lda);
        uint64_t idx = right ? vector + uint64_t(r) * ldc : r + uint64_t(vector) * ldc;
        Store(c, idx, Sub(Load(c, idx), Mul(right ? Conj(v) : v, dot)));
    }
}

}  // namespace
