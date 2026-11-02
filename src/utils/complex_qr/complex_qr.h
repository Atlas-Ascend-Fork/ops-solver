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

enum class ComplexQrOperation : uint32_t
{
    Factor,
    Generate,
    Apply,
    LeastSquares
};
constexpr uint32_t COMPLEX_QR_PANEL = 128;

// Complex elements occupy eight bytes; all internal partitions start at 32 B.
inline uint64_t ComplexQrAlignedElements(uint64_t elements) { return (elements + 3) & ~uint64_t(3); }

struct ComplexQrShape
{
    ComplexQrOperation operation;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t rhs;
    uint32_t side;
    uint32_t trans;
    uint64_t lda;
    uint64_t ldb;
    uint64_t ldx;
};

inline bool ComplexQrBlockedApply(uint32_t order, uint32_t columns, uint32_t reflectors)
{
    return order > 32 && columns > 32 && reflectors > 16;
}

void ComplexQrInfo(int* info, int value, void* stream);

uint32_t ComplexQrVectorCores();
uint32_t ComplexQrCubeCores();
