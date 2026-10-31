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
 * \file aicore_num.h
 * \brief host 侧 AI Core 核数获取的公共实现，供各 solver 入口统一做核数下限防护。
 */

#ifndef ACLSOLVER_UTILS_AICORE_NUM_H
#define ACLSOLVER_UTILS_AICORE_NUM_H

#include <cstdint>

#include "tiling/platform/platform_ascendc.h"

// 返回当前设备 AI Core 核数；平台实例获取失败时返回 0，由调用方以
// SOLVER_ECHECK(numBlocks > 0, "No available AIcore found on current device.", ...) 统一拦截
// （口径对齐 cpotrf_host.cpp 的既有防护）。
inline uint32_t GetAiCoreNumOrZero()
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (ascendcPlatform == nullptr)
    {
        return 0;
    }
    return ascendcPlatform->GetCoreNumAic();
}

#endif  // ACLSOLVER_UTILS_AICORE_NUM_H
