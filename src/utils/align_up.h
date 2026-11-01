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
 * \file align_up.h
 * \brief host 侧排 workspace 布局用的上取整。五个 QR/最小二乘算子共用。
 */

#ifndef ACLSOLVER_UTILS_ALIGN_UP_H
#define ACLSOLVER_UTILS_ALIGN_UP_H

#include <cstdint>

/// 把 x 向上对齐到 a 的整数倍。a == 0 时原样返回（对齐到 0 没有意义），
/// 这样除法处就地有守卫，不依赖调用点保证。
inline int64_t AlignUp64(int64_t x, int64_t a)
{
    return (a == 0) ? x : (x + a - 1) / a * a;
}

#endif  // ACLSOLVER_UTILS_ALIGN_UP_H
