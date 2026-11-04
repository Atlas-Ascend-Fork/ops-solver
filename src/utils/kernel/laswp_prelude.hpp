/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file laswp_prelude.hpp
 * \brief sgetrs/cgetrs 行交换 kernel 的公共前导（头文件依赖、跨核同步信号量与访存常量），
 *        消除两份 laswp.hpp 拷贝之间的重复定义（issue #149 CodeCheck 重复代码）。
 */

#ifndef ACLSOLVER_UTILS_KERNEL_LASWP_PRELUDE_H
#define ACLSOLVER_UTILS_KERNEL_LASWP_PRELUDE_H

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

#define VEC_CUBE_FLAGID 0x0
#define CUBE_VEC_FLAGID 0x1
#define ALL_VEC_FLAGID 0x2
#define ALL_CUBE_FLAGID 0x3
#define CACHE_LINE_SIZE_FP32 128  // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096  // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率

#endif  // ACLSOLVER_UTILS_KERNEL_LASWP_PRELUDE_H
