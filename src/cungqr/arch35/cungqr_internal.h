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

struct ComplexQrCall;

int ComplexQrCheckGenerate(const ComplexQrCall& call);
void LaunchIdentityWeightsKernel(uint32_t blocks, void* stream, const float* ah, const float* al, float* bh, float* bl,
                                 uint32_t rows, uint32_t panel, uint32_t columns, uint32_t packedColumns);
void ComplexQrGenerate(float* a, float* tau, float* work, ComplexQrShape shape, void* stream);

void ComplexQrCopyMatrix(const float* source, float* target, uint32_t rows, uint32_t columns, uint64_t sourceStride,
                         uint64_t targetStride, bool identity, void* stream);

void CungqrValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape shape, void* stream);
