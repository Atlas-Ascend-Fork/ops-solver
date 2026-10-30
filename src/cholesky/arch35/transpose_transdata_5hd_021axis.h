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

// Adapted from cannbot-skills ComputeRConvGeneric (94216ba2f0b569cfad22c7794ad99426a07ab7b6).
// Only its FP32 block transpose is retained; queues and global tiling belong to the caller.
namespace Cholesky
{
// FP32 hardware transpose: a 16-by-8 tile becomes an 8-by-16 tile.
__aicore__ inline void TransposeHardware(const AscendC::LocalTensor<float> &destination,
                                         const AscendC::LocalTensor<float> &source, int rows, int cols, int sourcePitch,
                                         int destinationPitch = 0)
{
    using namespace AscendC;
    if (!destinationPitch) destinationPitch = sourcePitch;
    uint64_t src[16], dst[16];
    int repeats = (rows + 15) / 16;
    TransDataTo5HDParams params{false, false, uint8_t(repeats), uint16_t(repeats == 1 ? 0 : 2),
                                uint16_t(repeats == 1 ? 0 : 2 * sourcePitch)};
    for (int c = 0; c < cols; c += 8)
    {
        for (int r = 0; r < 16; ++r) src[r] = reinterpret_cast<uint64_t>(source[r * sourcePitch + c].GetPhyAddr());
        for (int r = 0; r < 8; ++r)
        {
            dst[2 * r] = reinterpret_cast<uint64_t>(destination[(c + r) * destinationPitch].GetPhyAddr());
            dst[2 * r + 1] = reinterpret_cast<uint64_t>(destination[(c + r) * destinationPitch + 8].GetPhyAddr());
        }
        TransDataTo5HD<float>(dst, src, params);
    }
    PipeBarrier<PIPE_V>();
}

}  // namespace Cholesky
