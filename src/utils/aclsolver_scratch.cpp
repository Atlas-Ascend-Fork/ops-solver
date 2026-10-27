/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <new>

#include "aclsolver_handle_internal.h"
#include "assert.h"
#include "cholesky/cholesky.h"

namespace Cholesky
{
aclsolverStatus_t ReportError(aclsolverStatus_t status, const char* message)
{
    SOLVER_ECHECK(status == ACLSOLVER_STATUS_SUCCESS, message, status);
    return status;
}

namespace
{
aclsolverStatus_t AllocateScratch(std::vector<_aclsolver_handle::Scratch>& scratch, aclrtStream stream, size_t capacity,
                                  aclsolverFloatComplex** work)
{
    void* ptr = nullptr;
    if (aclrtMalloc(&ptr, capacity, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS)
    {
        return ReportError(ACLSOLVER_STATUS_ALLOC_FAILED, "Cholesky scratch allocation failed");
    }
    try
    {
        scratch.push_back({ptr, capacity, stream});
    }
    catch (const std::bad_alloc&)
    {
        aclrtFree(ptr);
        return ReportError(ACLSOLVER_STATUS_ALLOC_FAILED, "Cholesky scratch allocation failed");
    }
    *work = static_cast<aclsolverFloatComplex*>(ptr);
    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace

aclsolverStatus_t AcquireScratch(aclsolverHandle_t handle, aclrtStream* stream, size_t bytes,
                                 aclsolverFloatComplex** work)
{
    if (!handle || !stream || !work)
    {
        return ReportError(ACLSOLVER_STATUS_INVALID_VALUE, "invalid Cholesky scratch argument");
    }
    *work = nullptr;
    auto status = aclsolverGetStream(handle, stream);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return ReportError(status, "cannot get Cholesky stream");
    }
    if (!bytes)
    {
        return ACLSOLVER_STATUS_SUCCESS;
    }
    auto& scratch = reinterpret_cast<_aclsolver_handle*>(handle)->choleskyScratch;
    size_t capacity = bytes;
    for (auto it = scratch.rbegin(); it != scratch.rend(); ++it)
    {
        if (it->stream != *stream)
        {
            continue;
        }
        if (it->bytes >= bytes)
        {
            *work = static_cast<aclsolverFloatComplex*>(it->ptr);
        }
        else
        {
            capacity = bytes > 2 * it->bytes ? bytes : 2 * it->bytes;
        }
        break;
    }
    if (*work)
    {
        return ACLSOLVER_STATUS_SUCCESS;
    }
    return AllocateScratch(scratch, *stream, capacity, work);
}
}  // namespace Cholesky
