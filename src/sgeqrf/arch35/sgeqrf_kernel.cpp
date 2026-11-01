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
 * \file sgeqrf_kernel.cpp
 * \brief aclsolverSgeqrf 的 kernel 下发入口
 *
 * host 只传 workspace 基址与各区段的**元素偏移**；偏移一律由 host 的 PlanWorkspace 推导，
 * kernel 侧不重算 —— 两侧各写一份布局公式是这个 kernel 上踩过的坑（越界写进相邻缓冲
 * 表现为挂死而不是算错，极难定位）。
 */

#include "../kernel/geqrf_mix.hpp"

void sgeqrf_kernel_do(uint32_t numBlocks, void *stream, GM_ADDR A, GM_ADDR tau, GM_ADDR workspace, int32_t m,
                      int32_t n, int32_t lda, int32_t ldy, int32_t columns, int32_t alignBD, int64_t offY,
                      int64_t offT, int64_t offPanelW, int64_t offW, GM_ADDR devInfo)
{
    // GM_ADDR 在 kernel 编译单元里是 __gm__ uint8_t*，不能 reinterpret_cast 成裸 float*，
    // 偏移一律按**字节**在 GM_ADDR 上做（host 传进来的是元素偏移）。
    constexpr int64_t kElem = static_cast<int64_t>(sizeof(float));
    Geqrf2Params params{A,
                        tau,
                        workspace + offY * kElem,
                        workspace + offT * kElem,
                        workspace + offPanelW * kElem,  // 面板内交换 T 行的暂存
                        workspace + offW * kElem,       // 尾随更新的 W / W2（复用同一块）
                        static_cast<uint32_t>(m),
                        static_cast<uint32_t>(n),
                        static_cast<uint32_t>(lda),
                        static_cast<uint32_t>(ldy),
                        static_cast<uint32_t>(columns),
                        static_cast<uint32_t>(alignBD),
                        devInfo};
    sgeqrf_mix_kernel<<<numBlocks, nullptr, stream>>>(params);
}
