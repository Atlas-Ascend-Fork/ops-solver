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
#include <cstdint>
constexpr uint32_t COMPLEX_QR_SPLIT_OUTPUT_LIMIT = 16 * 128 * 128;
constexpr uint32_t COMPLEX_QR_SPLIT_MAX_PARTS = 8;
constexpr uint32_t COMPLEX_QR_SPLIT_FLOATS = COMPLEX_QR_SPLIT_OUTPUT_LIMIT * COMPLEX_QR_SPLIT_MAX_PARTS;
#include "../../utils/complex_qr/complex_qr.h"

// Internal row-major float GEMM used by compensated complex block updates.
bool ComplexQrCompensatedMatmul(const float* ah, const float* al, const float* bh, const float* bl, float* c,
                                uint32_t m, uint32_t n, uint32_t k, void* stream, float* splitScratch = nullptr,
                                bool transposeB = false);
void ComplexQrCompensatedMatmulLaunch(const float* ah, const float* al, const float* bh, const float* bl, float* c,
                                      uint32_t m, uint32_t n, uint32_t k, const void* tiling, bool interior,
                                      uint32_t splitParts, uint32_t splitK, void* stream, bool transposeB);
void ComplexQrReduceSplit(const float* partial, float* output, uint32_t elements, uint32_t parts, void* stream);
