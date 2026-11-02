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

void LaunchSplitWeightsKernel(uint32_t blocks, void* stream, const float* weights, float* high, float* low,
                              uint64_t count);
void LaunchUnpackGramKernel(uint32_t blocks, void* stream, const float* hh, const float* hl, const float* lh,
                            float* gram, uint32_t panel);
void LaunchFormTransformKernel(uint32_t blocks, void* stream, const float* tau, const float* gram, float* high,
                               float* low, uint32_t first, uint32_t panel, uint32_t actualPanel, bool adjoint);
void LaunchApplyKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c, uint32_t order,
                       uint64_t lda, uint64_t ldc, uint32_t k, uint32_t start, uint32_t count, bool right,
                       bool adjoint);
void LaunchCachedSequenceKernel(uint32_t blocks, void* stream, const float* a, const float* tau, float* c,
                                uint32_t order, uint64_t lda, uint64_t ldc, uint32_t kCount, uint32_t count,
                                bool adjoint, bool identity = false, bool right = false);
