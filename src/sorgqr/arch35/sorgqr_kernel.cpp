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
 * \file sorgqr_kernel.cpp
 * \brief aclsolverSorgqr 的 kernel 下发入口
 *
 * host 只传 workspace 基址与各区段的**元素偏移**；偏移一律由 host 的 PlanWorkspace 推导，
 * kernel 侧不重算 —— 两侧各写一份布局公式是这类 kernel 上踩过的坑（越界写进相邻缓冲
 * 表现为挂死而不是算错，极难定位）。
 */

#include "../kernel/orgqr_mix.hpp"

void sorgqr_kernel_do(uint32_t numBlocks, void *stream, GM_ADDR A, GM_ADDR tau, GM_ADDR workspace, int32_t m,
                      int32_t n, int32_t k, int32_t lda, int32_t ldy, int32_t nb, int64_t offY, int64_t offT,
                      int64_t offG, int64_t offW, int64_t offP, int64_t capP, GM_ADDR devInfo)
{
    // GM_ADDR 在 kernel 编译单元里是 __gm__ uint8_t*，不能 reinterpret_cast 成裸 float*，
    // 偏移一律按**字节**在 GM_ADDR 上做（host 传进来的是元素偏移）。
    constexpr int64_t kElem = static_cast<int64_t>(sizeof(float));
    OrgqrParams params{A,
                       tau,
                       workspace + offY * kElem,
                       workspace + offT * kElem,
                       workspace + offG * kElem,
                       workspace + offW * kElem,
                       workspace + offP * kElem,
                       static_cast<uint32_t>(m),
                       static_cast<uint32_t>(n),
                       static_cast<uint32_t>(k),
                       static_cast<uint32_t>(lda),
                       static_cast<uint32_t>(ldy),
                       static_cast<uint32_t>(nb),
                       devInfo,
                       static_cast<uint64_t>(capP)};
    sorgqr_mix_kernel<<<numBlocks, nullptr, stream>>>(params);
}
