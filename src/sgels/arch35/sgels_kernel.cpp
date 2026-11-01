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
 * \file sgels_kernel.cpp
 * \brief aclsolverSSgels 链路最后一段（上三角回代）的 kernel 下发入口
 *
 * host 只传 workspace 基址与各区段的**元素偏移**；偏移一律由 host 的 PlanWorkspace 推导，
 * kernel 侧不重算。
 */

#include "../kernel/trsm_mix.hpp"

void strsm_kernel_do(uint32_t numBlocks, void *stream, GM_ADDR A, GM_ADDR C, GM_ADDR X, GM_ADDR workspace,
                     int32_t n, int32_t nrhs, int32_t lda, int32_t ldc, int32_t ldx, int32_t nb, int64_t offW,
                     int64_t offP, int64_t capP)
{
    constexpr int64_t kElem = static_cast<int64_t>(sizeof(float));
    TrsmParams params{A,
                      C,
                      X,
                      workspace + offW * kElem,
                      workspace + offP * kElem,
                      static_cast<uint32_t>(n),
                      static_cast<uint32_t>(nrhs),
                      static_cast<uint32_t>(lda),
                      static_cast<uint32_t>(ldc),
                      static_cast<uint32_t>(ldx),
                      static_cast<uint32_t>(nb),
                      static_cast<uint64_t>(capP)};
    strsm_mix_kernel<<<numBlocks, nullptr, stream>>>(params);
}
