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
__simt_callee__ inline uint32_t ZeroDiagonal(__gm__ const float* a, uint32_t n, uint64_t lda)
{
    uint32_t first = n;
    for (uint32_t r = threadIdx.x; r < n; r += 32)
    {
        Complex diagonal = Load(a, r + uint64_t(r) * lda);
        if (diagonal.r == 0 && diagonal.i == 0 && r < first)
        {
            first = r;
        }
    }
    for (int delta = 16; delta > 0; delta /= 2)
    {
        uint32_t other = asc_shfl_down(first, delta);
        first = first < other ? first : other;
    }
    first = asc_shfl(first, 0);
    return first < n ? first + 1 : 0;
}
}  // namespace
