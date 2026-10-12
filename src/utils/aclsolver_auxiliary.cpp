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

namespace
{

/**
 * @brief Safely cast external void* handle to internal _aclsolver_handle*.
 */
inline _aclsolver_handle *ToInternal(aclsolverHandle_t handle) { return reinterpret_cast<_aclsolver_handle *>(handle); }

}  // namespace

extern "C"
{
    aclsolverStatus_t aclsolverCreate(aclsolverHandle_t *handle)
    {
        if (handle == nullptr)
        {
            return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        }

        // Prevent repeated creation that would overwrite an existing pointer and cause memory leak.
        if (*handle != nullptr)
        {
            return ACLSOLVER_STATUS_INVALID_VALUE;
        }

        // Contract check (issue #194): the documented lifecycle requires a current
        // CANN context; aclrtGetCurrentContext fails both before aclInit() and
        // after aclInit() without a device selection.
        aclrtContext currentContext = nullptr;
        if (aclrtGetCurrentContext(&currentContext) != ACL_SUCCESS || currentContext == nullptr)
        {
            return ACLSOLVER_STATUS_NOT_INITIALIZED;
        }

        auto *h = new (std::nothrow) _aclsolver_handle();
        if (h == nullptr)
        {
            return ACLSOLVER_STATUS_ALLOC_FAILED;
        }

        *handle = reinterpret_cast<void *>(h);
        return ACLSOLVER_STATUS_SUCCESS;
    }

    aclsolverStatus_t aclsolverDestroy(aclsolverHandle_t handle)
    {
        if (handle == nullptr)
        {
            return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        }

        auto *h = ToInternal(handle);

        // stream is managed by the user; here we only clear handle fields, not freeing device memory.
        h->stream = nullptr;

        delete h;
        return ACLSOLVER_STATUS_SUCCESS;
    }

    aclsolverStatus_t aclsolverSetStream(aclsolverHandle_t handle, aclrtStream stream)
    {
        if (handle == nullptr)
        {
            return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        }

        auto *h = ToInternal(handle);
        h->stream = stream;
        return ACLSOLVER_STATUS_SUCCESS;
    }

    aclsolverStatus_t aclsolverGetStream(aclsolverHandle_t handle, aclrtStream *stream)
    {
        if (handle == nullptr)
        {
            return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        }
        if (stream == nullptr)
        {
            return ACLSOLVER_STATUS_INVALID_VALUE;
        }

        auto *h = ToInternal(handle);
        *stream = h->stream;
        return ACLSOLVER_STATUS_SUCCESS;
    }

}  // extern "C"
