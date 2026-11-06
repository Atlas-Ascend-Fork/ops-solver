/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software: you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file free_reset.h
 * \brief 幂等释放工具：释放成功后立即置空指针
 *
 * 失败路径的清理序列常被多处共用（如 CHECK_ACLRT 的 cleanup 形参），释放后
 * 置空可防止同一指针被二次 aclrtFree / aclrtFreeHost（issue #186 口径）。
 */

#ifndef ACLSOLVER_UTILS_FREE_RESET_H
#define ACLSOLVER_UTILS_FREE_RESET_H

#include "acl/acl.h"

/// 释放并置空：pointer 为空时什么都不做。freeFunc 取 aclrtFree / aclrtFreeHost。
template <typename T>
inline void FreeAndReset(T*& pointer, aclError (*freeFunc)(void*))
{
    if (pointer != nullptr)
    {
        (void)freeFunc(pointer);
        pointer = nullptr;
    }
}

#endif  // ACLSOLVER_UTILS_FREE_RESET_H
