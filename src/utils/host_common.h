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
 * \file host_common.h
 * \brief QR / 最小二乘五个算子 host 侧都要的那批头与工具，收在一处。
 *
 * 每个 host 还要再 include 自己 kernel 的 launch 头（声明下发入口）。
 */

#ifndef ACLSOLVER_UTILS_HOST_COMMON_H
#define ACLSOLVER_UTILS_HOST_COMMON_H

#include <algorithm>
#include <cstdint>

#include "acl/acl.h"
#include "align_up.h"
#include "assert.h"
#include "cann_ops_solver.h"
#include "gm_addr.h"
#include "tiling/platform/platform_ascendc.h"

/// 把 value 写进 device 侧的 info（info 为空时什么都不做）。
/// 只在"校验失败、不下发 kernel"的路径上用：此时流上没有在排的任务，
/// 同步 memcpy 不构成无必要的 Host 同步。四个 host 原先各抄了一份。
inline aclError WriteInfoSync(int32_t *devInfo, int32_t value)
{
    if (devInfo == nullptr)
    {
        return ACL_SUCCESS;
    }
    return aclrtMemcpy(devInfo, sizeof(int32_t), &value, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE);
}

/// 入参校验失败的统一出口：把 -idx 写进 device 侧 info，再按 INVALID_VALUE 返回。
/// 各 host 的 info 变量名不同（devInfo / dinfo），所以用形参传进来，
/// 各自再定一行别名宏，调用点不必改。
#define SOLVER_PARAM_CHECK(info, cond, idx, msg)                         \
    do                                                                   \
    {                                                                    \
        if (!(cond))                                                     \
        {                                                                \
            (void)WriteInfoSync((info), -(idx));                         \
            SOLVER_ECHECK(false, (msg), ACLSOLVER_STATUS_INVALID_VALUE); \
        }                                                                \
    } while (0)

#endif  // ACLSOLVER_UTILS_HOST_COMMON_H
