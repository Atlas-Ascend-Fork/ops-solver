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
 * \file trsm_launch.h
 * \brief sgels kernel 下发入口的声明。host 与 kernel 两侧共用此声明，
 *        避免在 .cpp 里用 extern 直接引用外部函数（codecheck G.EXP.05-CPP）。
 *
 * \warning **仅供 host 侧包含**。本头依赖 utils/gm_addr.h 的 host 版 GM_ADDR
 *          （普通 uint8_t*），而 kernel 侧 CCE 的 tikcfw 内置的同名宏带 __gm__
 *          地址空间标注，两者含义不同 —— 见 utils/gm_addr.h 的说明。
 */

#ifndef ACLSOLVER_TRSM_LAUNCH_H
#define ACLSOLVER_TRSM_LAUNCH_H

#include <cstdint>

#include "../../utils/gm_addr.h"

void strsm_kernel_do(uint32_t numBlocks, void *stream, GM_ADDR A, GM_ADDR C, GM_ADDR X, GM_ADDR workspace,
                            int32_t n, int32_t nrhs, int32_t lda, int32_t ldc, int32_t ldx, int32_t nb, int64_t offW,
                            int64_t offP, int64_t capP);

#endif  // ACLSOLVER_TRSM_LAUNCH_H
