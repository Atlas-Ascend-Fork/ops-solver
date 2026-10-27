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
 * \file aclsolver_handle_internal.h
 * \brief aclsolver handle internal structure (not exposed externally).
 */

#pragma once

#include <acl/acl.h>

#include <cstddef>
#include <vector>

#include "cann_ops_solver_common.h"

/**
 * @brief aclsolver handle internal structure
 *
 * Contains the current stream and reusable Cholesky scratch for asynchronous execution.
 *
 * This struct definition is only visible in implementation files and is completely
 * opaque to external users. The external interface uses void* / void** as handle types.
 */
struct _aclsolver_handle
{
    /* ========== Stream ========== */
    aclrtStream stream = nullptr;  ///< Current stream for kernel execution
    struct Scratch
    {
        void* ptr;
        size_t bytes;
        aclrtStream stream;
    };
    // Device scratch remains alive until all work on its owning stream completes.
    std::vector<Scratch> choleskyScratch;
};
