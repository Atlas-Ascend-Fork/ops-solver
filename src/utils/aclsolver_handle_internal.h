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
 * Contains the caller stream and reusable LU and Cholesky scratch for asynchronous execution.
 *
 * This struct definition is only visible in implementation files and is completely
 * opaque to external users. The external interface uses void* / void** as handle types.
 */
struct _real_lu_workspace
{
    aclrtContext context = nullptr;
    void *data = nullptr;
    size_t capacity = 0;
    aclrtEvent completion = nullptr;
    bool recorded = false;
    // Preserve ownership if both recording completion and synchronizing fail.
    aclrtStream pendingStream = nullptr;
    bool untracked = false;
    // Validation copies have a separate completion obligation from scratch.
    // A failed stream synchronization transfers their buffers to this slot.
    aclrtStream validationStream = nullptr;
    bool validationPending = false;
    void *validationHost = nullptr;
    void *validationFlags = nullptr;
};

inline aclsolverStatus_t RealLuDrainValidation(_real_lu_workspace &cache)
{
    if (cache.validationPending)
    {
        if (aclrtSynchronizeStream(cache.validationStream) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.validationPending = false;
        cache.validationStream = nullptr;
    }
    if (cache.validationHost != nullptr)
    {
        if (aclrtFreeHost(cache.validationHost) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.validationHost = nullptr;
    }
    if (cache.validationFlags != nullptr)
    {
        if (aclrtFree(cache.validationFlags) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.validationFlags = nullptr;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

struct _aclsolver_handle
{
    /* ========== Stream ========== */
    aclrtStream stream = nullptr;  ///< Current stream for kernel execution
    // Large out-of-place inverse scratch is reused per context and released
    // when this handle is destroyed. Events protect reuse across caller streams.
    std::vector<_real_lu_workspace> realLuWorkspaces;
    struct Scratch
    {
        void *ptr;
        size_t bytes;
        aclrtStream stream;
    };
    // Device scratch remains alive until all work on its owning stream completes.
    std::vector<Scratch> choleskyScratch;
};
