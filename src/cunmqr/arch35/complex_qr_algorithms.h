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
#include "complex_qr_matmul.h"

uint32_t ComplexQrBlocks(uint32_t count);
bool ComplexQrBlockUpdate(float* a, float* tau, float* c, float* work, uint32_t order, uint32_t cols, uint64_t lda,
                          uint64_t ldc, uint32_t first, uint32_t end, uint32_t columnStart, bool adjoint, void* stream,
                          bool identity = false, bool rightLayout = false);

void ComplexQrPackTrailing(const float* a, float* high, float* low, uint64_t lda, uint32_t first, uint32_t end,
                           uint32_t rows, uint32_t cols, uint32_t packedRows, uint32_t packedCols, void* stream,
                           bool rightLayout = false);
void ComplexQrPackTransposed(const float* a, float* high, float* low, uint64_t lda, uint32_t first,
                             uint32_t columnStart, uint32_t rows, uint32_t columns, uint32_t packedRows,
                             uint32_t packedColumns, void* stream);
void ComplexQrPackReflectors(const float* a, float* adjHigh, float* adjLow, float* vHigh, float* vLow, uint64_t lda,
                             uint32_t first, uint32_t rows, uint32_t panel, uint32_t actualRows, uint32_t actualPanel,
                             void* stream);
void ComplexQrUpdateTrailing(float* a, const float* product, uint64_t lda, uint32_t first, uint32_t end, uint32_t rows,
                             uint32_t cols, uint32_t packedRows, uint32_t packedCols, void* stream,
                             bool rightLayout = false);

bool ComplexQrBlockWorkspace(uint32_t order, uint32_t columns, uint64_t& elements);
