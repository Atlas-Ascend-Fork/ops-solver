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
 * \file getrs_defs.h
 * \brief sgetrs/cgetrs host 与 kernel 系列共用的访存常量宏（单一来源）。
 * 各文件统一包含本头文件，禁止在本地重复定义，避免改一处漏其余（issue #327）。
 */

#ifndef ACLSOLVER_UTILS_GETRS_DEFS_H
#define ACLSOLVER_UTILS_GETRS_DEFS_H

#define CACHE_LINE_SIZE_FP32 128  // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096  // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率
#define VEC_CUBE_FLAGID 0x0       // vec -> cube 核间同步 flag id
#define CUBE_VEC_FLAGID 0x1       // cube -> vec 核间同步 flag id
#define ALL_VEC_FLAGID 0x2        // vec 多核间同步 flag id
#define ALL_CUBE_FLAGID 0x3       // cube 多核间同步 flag id

#endif  // ACLSOLVER_UTILS_GETRS_DEFS_H
