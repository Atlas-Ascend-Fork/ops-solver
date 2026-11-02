/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "complex_qr_matmul.h"
#include "cunmqr_transfer_helpers.h"
#include "kernel_operator.h"

namespace
{
__global__ __aicore__ void PackReflectorsKernel(__gm__ const float* a, __gm__ float* ah, __gm__ float* al,
                                                __gm__ float* vh, __gm__ float* vl, uint64_t lda, uint32_t first,
                                                uint32_t rows, uint32_t panel, uint32_t actualRows,
                                                uint32_t actualPanel)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    ComplexQrTransfer::ReflectorsDma(a, ah, al, vh, vl, lda, first, rows, panel, actualRows, actualPanel);
}
}  // namespace

void ComplexQrPackReflectors(const float* a, float* ah, float* al, float* vh, float* vl, uint64_t lda, uint32_t first,
                             uint32_t rows, uint32_t panel, uint32_t actualRows, uint32_t actualPanel, void* stream)
{
    PackReflectorsKernel<<<ComplexQrVectorCores(), nullptr, stream>>>(a, ah, al, vh, vl, lda, first, rows, panel,
                                                                      actualRows, actualPanel);
}
