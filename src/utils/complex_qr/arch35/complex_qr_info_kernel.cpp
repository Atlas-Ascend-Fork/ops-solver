/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../complex_qr.h"
#include "kernel_operator.h"

namespace
{
__global__ __aicore__ void InfoKernel(__gm__ int* info, int value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::GlobalTensor<int> result;
    result.SetGlobalBuffer(info, 1);
    result.SetValue(0, value);
    AscendC::DataCacheCleanAndInvalid<int, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(
        result);
}
}  // namespace

void ComplexQrInfo(int* info, int value, void* stream) { InfoKernel<<<1, nullptr, stream>>>(info, value); }
