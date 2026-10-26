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
 * \file spotrf_launchers.hpp
 * \brief Internal launch interface shared by the Spotrf host and device translation units.
 */

#ifndef SPOTRF_LAUNCHERS_HPP
#define SPOTRF_LAUNCHERS_HPP

#include <cstdint>

void spotrf_kernel_do(uint8_t *sync, int orgM, int orgN, int blockM, int blockN, int lda, bool isTranse, uint8_t *A_org,
                      uint8_t *A_work, uint8_t *Gather, uint32_t *offset1, uint32_t *offset2, uint32_t *offset3,
                      uint32_t *OffsetIdx, uint32_t numBlocks, void *stream);

#endif  // SPOTRF_LAUNCHERS_HPP
