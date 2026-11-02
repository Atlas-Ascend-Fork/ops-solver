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

int ComplexQrCheckLeastSquares(const ComplexQrCall& call);
void LaunchSolveKernel(uint32_t blocks, void* stream, const float* a, const float* y, float* x, int* info,
                       ComplexQrShape s);
void ComplexQrLeastSquares(float* a, float* b, float* x, float* work, int* info, ComplexQrShape shape, void* stream);
void LaunchSolveSimdKernel(uint32_t blocks, void* stream, const float* a, const float* y, float* x, int* info,
                           ComplexQrShape s);

void CcgelsValidate(const float* a, const float* tau, const float* b, int* info, ComplexQrShape shape, void* stream);
