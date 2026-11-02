/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../complex_qr_host.h"

#include <algorithm>
#include <climits>
#include <limits>

namespace
{

aclsolverStatus_t Report(aclrtStream stream, int* info, int error)
{
    if (info)
    {
        ComplexQrInfo(info, -error, stream);
    }
    return ACLSOLVER_STATUS_INVALID_VALUE;
}
int CheckExecution(const ComplexQrPolicy& policy, const ComplexQrCall& c, void* workspace, size_t bytes, int* niter)
{
    int error = policy.check(c);
    if (!error)
    {
        error = policy.executionCheck(c, nullptr, false);
    }
    size_t needed = 0;
    if (!error && (!policy.workspace(c, needed) || bytes < needed || bytes == SIZE_MAX))
    {
        error = policy.workspaceArgument + 1;
    }
    if (!error && needed && !workspace)
    {
        error = policy.workspaceArgument;
    }

    if (!error)
    {
        error = policy.executionCheck(c, niter, true);
    }
    return error;
}

aclsolverStatus_t ValidateDeviceInput(const ComplexQrPolicy& policy, const ComplexQrCall& c, int* info,
                                      ComplexQrShape shape, aclrtStream stream)
{
    policy.validate(reinterpret_cast<const float*>(c.a), reinterpret_cast<const float*>(c.tau),
                    reinterpret_cast<const float*>(c.b), info, shape, stream);
    // Required to return INVALID_VALUE for nonfinite Device input, as specified by the task.
    if (aclrtSynchronizeStream(stream) != ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    int deviceInfo = 0;
    if (aclrtMemcpy(&deviceInfo, sizeof(deviceInfo), info, sizeof(deviceInfo), ACL_MEMCPY_DEVICE_TO_HOST) !=
        ACL_SUCCESS)
    {
        return ACLSOLVER_STATUS_MAPPING_ERROR;
    }
    if (deviceInfo)
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

}  // namespace

aclsolverStatus_t ComplexQrQueryBytes(const ComplexQrPolicy& policy, aclsolverHandle_t handle,
                                      const ComplexQrCall& call, size_t* bytes)
{
    if (!handle)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!bytes || policy.check(call))
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    if (!policy.workspace(call, *bytes))
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    return ACLSOLVER_STATUS_SUCCESS;
}

aclsolverStatus_t ComplexQrQuery(const ComplexQrPolicy& policy, aclsolverHandle_t handle, const ComplexQrCall& call,
                                 int* elements)
{
    if (!handle)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    if (!elements)
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    size_t bytes = 0;
    const auto status = ComplexQrQueryBytes(policy, handle, call, &bytes);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    if (bytes / sizeof(aclsolverFloatComplex) > INT_MAX)
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    *elements = static_cast<int>(bytes / sizeof(aclsolverFloatComplex));
    return ACLSOLVER_STATUS_SUCCESS;
}

aclsolverStatus_t ComplexQrRun(const ComplexQrPolicy& policy, aclsolverHandle_t handle, const ComplexQrCall& c,
                               void* workspace, size_t bytes, int* info, int* niter)
{
    if (!handle)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    aclrtStream stream = nullptr;
    const auto streamStatus = aclsolverGetStream(handle, &stream);
    if (streamStatus != ACLSOLVER_STATUS_SUCCESS)
    {
        return streamStatus;
    }
    const int error = CheckExecution(policy, c, workspace, bytes, niter);
    if (!info || error)
    {
        return Report(stream, info, error ? error : policy.infoArgument);
    }
    if (niter)
    {
        *niter = 0;
    }
    if (policy.empty(c))
    {
        ComplexQrInfo(info, 0, stream);
        return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS ? ACLSOLVER_STATUS_SUCCESS
                                                                        : ACLSOLVER_STATUS_EXECUTION_FAILED;
    }
    const ComplexQrShape shape = {c.operation,     uint32_t(c.m),    uint32_t(c.n),     uint32_t(c.k),
                                  uint32_t(c.rhs), uint32_t(c.side), uint32_t(c.trans), uint64_t(c.lda),
                                  uint64_t(c.ldb), uint64_t(c.ldx)};
    const auto validation = ValidateDeviceInput(policy, c, info, shape, stream);
    if (validation != ACLSOLVER_STATUS_SUCCESS)
    {
        return validation;
    }
    policy.execute(reinterpret_cast<float*>(const_cast<aclsolverFloatComplex*>(c.a)),
                   reinterpret_cast<float*>(const_cast<aclsolverFloatComplex*>(c.tau)),
                   reinterpret_cast<float*>(const_cast<aclsolverFloatComplex*>(c.b)), reinterpret_cast<float*>(c.x),
                   static_cast<float*>(workspace), info, shape, stream);
    // Kernel launches are asynchronous; do not report success when enqueueing
    // failed. Device execution errors are still observed by stream synchronization.
    return aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL) == ACL_SUCCESS ? ACLSOLVER_STATUS_SUCCESS
                                                                    : ACLSOLVER_STATUS_EXECUTION_FAILED;
}
