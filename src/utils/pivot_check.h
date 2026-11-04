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
 * \file pivot_check.h
 * \brief getrs 族 host 侧 ipiv 内容防御校验的公共实现（issue #299，对齐 getri 链路
 *        PreparePivots 的口径：sgetrf 输出为 1-based 行号，取值须在 [1, n]，非法时
 *        info 标记 -4 并返回参数错误）。
 */

#ifndef ACLSOLVER_UTILS_PIVOT_CHECK_H
#define ACLSOLVER_UTILS_PIVOT_CHECK_H

#include <cstdint>

#include "assert.h"

// 校验 ipiv 每个元素位于 [1, n]（1-based 行交换序）；非法时写 *info = -4、打印日志
// 并返回 false，由调用方返回参数错误。opName 用于日志中的接口名标识。
inline bool ValidatePivots(const int32_t *ipiv, int64_t n, int32_t *info, const char *opName)
{
    for (int64_t i = 0; i < n; ++i)
    {
        if (ipiv[i] < 1 || ipiv[i] > n)
        {
            if (info != nullptr)
            {
                *info = -4;
            }
            LOG_PRINT("%s invalid ipiv[%lld] = %d, expected in [1, %lld].\n", opName, static_cast<long long>(i),
                      ipiv[i], static_cast<long long>(n));
            return false;
        }
    }
    return true;
}

#endif  // ACLSOLVER_UTILS_PIVOT_CHECK_H
