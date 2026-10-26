/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cpotrf_launchers.hpp
 * \brief Internal launch interfaces shared by the Cpotrf host and device translation units.
 */

#ifndef CPOTRF_LAUNCHERS_HPP
#define CPOTRF_LAUNCHERS_HPP

#include <cstdint>

void cpotrf_kernel_do(uint8_t* sync, bool transpose, int orgN, int lda, uint8_t* A_org, uint8_t* A_work,
                      uint8_t* offset1_gm, uint8_t* offset2_gm, uint8_t* offset3_gm, uint8_t* offsetIdx_gm,
                      uint8_t* gather_gm, uint32_t numBlocks, void* stream);

#endif  // CPOTRF_LAUNCHERS_HPP
