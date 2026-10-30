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
 * \file aclsolver_auxiliary.cpp
 * \brief aclsolver handle lifecycle, stream, version and other auxiliary function implementations.
 */

#include <acl/acl.h>

#include <new>

#include "aclsolver_handle_internal.h"
#include "cann_ops_solver.h"
#include "cholesky/cholesky.h"

namespace
{

/**
 * @brief Safely cast external void* handle to internal _aclsolver_handle*.
 */
inline _aclsolver_handle* ToInternal(aclsolverHandle_t handle) { return reinterpret_cast<_aclsolver_handle*>(handle); }

inline aclsolverStatus_t ReleaseWorkspace(_real_lu_workspace& cache)
{
    if (RealLuDrainValidation(cache) != ACLSOLVER_STATUS_SUCCESS)
    {
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    if (cache.untracked)
    {
        if (aclrtSynchronizeStream(cache.pendingStream) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.untracked = false;
        cache.recorded = false;
        cache.pendingStream = nullptr;
    }
    if (cache.recorded && aclrtSynchronizeEvent(cache.completion) != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    cache.recorded = false;
    if (cache.data != nullptr)
    {
        if (aclrtFree(cache.data) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.data = nullptr;
        cache.capacity = 0;
    }
    if (cache.completion != nullptr)
    {
        if (aclrtDestroyEvent(cache.completion) != ACL_SUCCESS)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
        cache.completion = nullptr;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

inline bool HasWorkspaceResources(const _real_lu_workspace& cache)
{
    return cache.data != nullptr || cache.completion != nullptr || cache.recorded || cache.untracked ||
           cache.validationPending || cache.validationHost != nullptr || cache.validationFlags != nullptr;
}

inline aclsolverStatus_t ReleaseHandleWorkspaces(_aclsolver_handle& handle)
{
    aclsolverStatus_t status = ACLSOLVER_STATUS_SUCCESS;
    if (handle.realLuWorkspaces.empty())
    {
        return status;
    }
    aclrtContext original = nullptr;
    if (aclrtGetCurrentContext(&original) != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    aclrtContext active = original;
    for (auto& cache : handle.realLuWorkspaces)
    {
        if (!HasWorkspaceResources(cache))
        {
            continue;
        }
        if (cache.context != active && aclrtSetCurrentContext(cache.context) != ACL_SUCCESS)
        {
            status = ACLSOLVER_STATUS_EXECUTION_FAILED;
            continue;
        }
        active = cache.context;
        if (ReleaseWorkspace(cache) != ACLSOLVER_STATUS_SUCCESS)
        {
            status = ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
    }
    if (original != nullptr && aclrtSetCurrentContext(original) != ACL_SUCCESS)
    {
        status = ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    return status;
}

inline aclsolverStatus_t ReleaseCholeskyWorkspaces(_aclsolver_handle& handle)
{
    aclsolverStatus_t status = ACLSOLVER_STATUS_SUCCESS;
    auto rememberError = [&status](aclsolverStatus_t error, const char* message)
    {
        Cholesky::ReportError(error, message);
        if (status == ACLSOLVER_STATUS_SUCCESS)
        {
            status = error;
        }
    };
    bool deviceSyncAttempted = false;
    bool deviceSynchronized = false;
    for (size_t i = 0; i < handle.choleskyScratch.size();)
    {
        const auto& block = handle.choleskyScratch[i];
        if (!deviceSynchronized && aclrtSynchronizeStream(block.stream) != ACL_SUCCESS)
        {
            rememberError(ACLSOLVER_STATUS_EXECUTION_FAILED, "cannot synchronize Cholesky scratch stream");
            if (!deviceSyncAttempted)
            {
                deviceSyncAttempted = true;
                deviceSynchronized = aclrtSynchronizeDevice() == ACL_SUCCESS;
            }
            if (!deviceSynchronized)
            {
                // Keep ownership until a retry can prove that no kernel still uses this block.
                ++i;
                continue;
            }
        }
        if (aclrtFree(block.ptr) != ACL_SUCCESS)
        {
            rememberError(ACLSOLVER_STATUS_INTERNAL_ERROR, "cannot release Cholesky scratch");
            ++i;
            continue;
        }
        handle.choleskyScratch.erase(handle.choleskyScratch.begin() + i);
    }
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        // Failure always leaves a valid handle; retry also releases an emptied host handle.
        return status;
    }
    return status;
}

}  // namespace

extern "C"
{
    aclsolverStatus_t aclsolverCreate(aclsolverHandle_t* handle)
    {
        if (handle == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, "solver handle is null");
        }

        // Prevent repeated creation that would overwrite an existing pointer and cause memory leak.
        if (*handle != nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_INVALID_VALUE, "invalid solver handle or stream output");
        }

        // Handle creation requires an initialized CANN context, as in the upstream lifecycle contract.
        aclrtContext currentContext = nullptr;
        if (aclrtGetCurrentContext(&currentContext) != ACL_SUCCESS || currentContext == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_NOT_INITIALIZED,
                                         "solver requires an initialized CANN context");
        }

        auto* h = new (std::nothrow) _aclsolver_handle();
        if (h == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_ALLOC_FAILED, "solver handle allocation failed");
        }

        *handle = reinterpret_cast<void*>(h);
        return ACLSOLVER_STATUS_SUCCESS;
    }

    aclsolverStatus_t aclsolverDestroy(aclsolverHandle_t handle)
    {
        if (handle == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, "solver handle is null");
        }

        auto* h = ToInternal(handle);

        aclsolverStatus_t status = ReleaseHandleWorkspaces(*h);
        // LU cleanup can switch contexts. On failure, preserve ownership and
        // let the caller restore the owning context before retrying Destroy.
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
        status = ReleaseCholeskyWorkspaces(*h);
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }

        // Keep real scratch owned as well while complex cleanup requires a retry.
        if (h->releaseScratch)
        {
            h->releaseScratch(h->solverScratch);
        }

        // The caller owns the stream; only handle-owned scratch is released here.
        h->stream = nullptr;

        delete h;
        return ACLSOLVER_STATUS_SUCCESS;
    }

    aclsolverStatus_t aclsolverSetStream(aclsolverHandle_t handle, aclrtStream stream)
    {
        if (handle == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, "solver handle is null");
        }

        auto* h = ToInternal(handle);
        h->stream = stream;
        return ACLSOLVER_STATUS_SUCCESS;
    }

    aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t handle, aclrtStream* stream)
    {
        if (handle == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_HANDLE_IS_NULLPTR, "solver handle is null");
        }
        if (stream == nullptr)
        {
            return Cholesky::ReportError(ACLSOLVER_STATUS_INVALID_VALUE, "invalid solver handle or stream output");
        }

        auto* h = ToInternal(handle);
        *stream = h->stream;
        return ACLSOLVER_STATUS_SUCCESS;
    }

}  // extern "C"
