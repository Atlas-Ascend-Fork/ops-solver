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

int ComplexQrCheckFactor(const ComplexQrCall& call);
void LaunchFactorPanelUncachedKernel(uint32_t blocks, void* stream, float* a, float* tau, uint32_t m, uint32_t n,
                                     uint64_t lda, uint32_t first, uint32_t last);
void LaunchReflectorKernel(uint32_t blocks, void* stream, float* a, float* tau, uint32_t m, uint64_t lda, uint32_t k);
void LaunchFactorPanelKernel(uint32_t blocks, void* stream, float* a, float* tau, uint32_t m, uint32_t n, uint64_t lda,
                             uint32_t first, uint32_t last);
void LaunchCachedApplyKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c, uint64_t lda,
                             uint64_t ldc, uint32_t rows, uint32_t cols, uint32_t first, uint32_t panel,
                             uint32_t columnStart);
void ComplexQrFactor(float* a, float* tau, float* work, ComplexQrShape shape, void* stream);
void LaunchCachedApplySimdKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c,
                                 uint64_t lda, uint64_t ldc, uint32_t rows, uint32_t cols, uint32_t first,
                                 uint32_t panel, uint32_t columnStart);

bool ComplexQrFactorWorkspace(const ComplexQrCall& call, uint64_t& scratch);

void CgeqrfValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape shape, void* stream);
