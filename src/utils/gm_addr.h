/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms of the
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSOLVER_UTILS_GM_ADDR_H
#define ACLSOLVER_UTILS_GM_ADDR_H

#include <cstdint>

// host 侧全局内存地址别名：收敛各 host 文件的本地 #define GM_ADDR（此前多处重复定义且
// 写法不一，issue #145）。
// 注意：kernel 侧（AscendC 编译单元）不能使用本头——CCE 的 tikcfw 已内置定义同名宏
// GM_ADDR（带 __gm__ 地址空间标注），kernel 文件需保留其本地 #define 以维持原有覆盖关系。
// host 侧无 __gm__ 语义，宏内容为普通 uint8_t 指针，多声明陷阱（GM_ADDR a, b; 的 b 退化）
// 由 host 侧统一单声明习惯规避。
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

#endif  // ACLSOLVER_UTILS_GM_ADDR_H
