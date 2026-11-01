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
 * \file geqrf_launch.h
 * \brief sgeqrf kernel 下发入口的声明。host 与 kernel 两侧共用此声明，
 *        避免在 .cpp 里用 extern 直接引用外部函数（codecheck G.EXP.05-CPP）。
 *
 * \warning **仅供 host 侧包含**。本头依赖 utils/gm_addr.h 的 host 版 GM_ADDR
 *          （普通 uint8_t*），而 kernel 侧 CCE 的 tikcfw 内置的同名宏带 __gm__
 *          地址空间标注，两者含义不同 —— 见 utils/gm_addr.h 的说明。
 */

#ifndef ACLSOLVER_GEQRF_LAUNCH_H
#define ACLSOLVER_GEQRF_LAUNCH_H

#include <cstdint>

#include "../../utils/gm_addr.h"

void sgeqrf_kernel_do(uint32_t numBlocks, void *stream, GM_ADDR A, GM_ADDR tau, GM_ADDR workspace, int32_t m,
                             int32_t n, int32_t lda, int32_t ldy, int32_t columns, int32_t alignBD, int64_t offY,
                             int64_t offT, int64_t offPanelW, int64_t offW, GM_ADDR devInfo);

#endif  // ACLSOLVER_GEQRF_LAUNCH_H
