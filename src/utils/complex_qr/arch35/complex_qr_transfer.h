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

namespace ComplexQrTransfer
{
using namespace AscendC;
template <HardEvent Event>
__aicore__ inline void Sync()
{
    auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(Event));
    SetFlag<Event>(id);
    WaitFlag<Event>(id);
}

// Strides use bytes on GM and 32-byte blocks on UB in Normal padding mode.
__aicore__ inline void Read(LocalTensor<float> dst, GlobalTensor<float> src, uint64_t offset, uint32_t blocks,
                            uint32_t width, uint64_t gmPitch, uint32_t ubPitch)
{
    uint32_t aligned = (width + 7) / 8 * 8;
    DataCopyExtParams params(static_cast<uint16_t>(blocks), width * sizeof(float), (gmPitch - width) * sizeof(float),
                             (ubPitch - aligned) / 8, 0);
    DataCopyPadExtParams<float> padding{true, 0, static_cast<uint8_t>(aligned - width), 0.0f};
    DataCopyPad(dst, src[offset], params, padding);
}

__aicore__ inline void Write(GlobalTensor<float> dst, uint64_t offset, LocalTensor<float> src, uint32_t blocks,
                             uint32_t width, uint64_t gmPitch, uint32_t ubPitch)
{
    uint32_t aligned = (width + 7) / 8 * 8;
    DataCopyExtParams params(static_cast<uint16_t>(blocks), width * sizeof(float), (ubPitch - aligned) / 8,
                             (gmPitch - width) * sizeof(float), 0);
    DataCopyPad(dst[offset], src, params);
}

}  // namespace ComplexQrTransfer
