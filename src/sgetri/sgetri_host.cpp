/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../utils/real_lu_common.h"

aclsolverStatus_t aclsolverSgetri_bufferSize(aclsolverHandle_t handle, int n, float *A, int lda, int *Lwork)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    if (n < 0 || n > 4096 || lda < std::max(1, n) || Lwork == nullptr || real_lu::Device(Lwork) ||
        (A != nullptr && !real_lu::Device(A)))
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    *Lwork = real_lu::InverseWorkspace(n);
    return ACLSOLVER_STATUS_SUCCESS;
}

namespace
{
int InverseArgument(int n, float *a, int lda, int *piv, float *workspace, int workElements, int *info)
{
    if (n < 0 || n > 4096)
    {
        return 1;
    }
    if (n > 0 && !real_lu::Device(a))
    {
        return 2;
    }
    if (lda < std::max(1, n))
    {
        return 3;
    }
    if (piv != nullptr && !real_lu::Device(piv))
    {
        return 4;
    }
    if (n > 0 &&
        (!real_lu::Device(workspace) || real_lu::Overlap(a, static_cast<size_t>(lda) * n * sizeof(float), workspace,
                                                         static_cast<size_t>(n) * n * sizeof(float))))
    {
        return 5;
    }
    if (workElements < real_lu::InverseWorkspace(n))
    {
        return 6;
    }
    return real_lu::Device(info) ? 0 : 7;
}
aclsolverStatus_t LaunchInverse(aclsolverHandle_t handle, int n, float *a, int lda, int *piv, float *workspace,
                                int *info, bool identity)
{
    auto stream = real_lu::Stream(handle);
    if (real_lu::BlockedInverse(n))
    {
        if (real_lu_blocked_inverse_do(a, a, piv, info, workspace, n, lda, lda, 1, false, identity, stream) != 0)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
    }
    else
    {
        if (lda == n && (n == 4 || n == 8))
        {
            real_lu_small_do(a, workspace, piv, info, n, 1, false, identity, stream);
        }
        else
        {
            real_lu_inverse_do(a, workspace, piv, info, n, lda, n, 1, false, false, stream);
        }
        real_lu_copy_do(workspace, a, info, n, lda, stream);
    }
    return real_lu::Launched();
}
}  // namespace

aclsolverStatus_t aclsolverSgetri(aclsolverHandle_t handle, int n, float *A, int lda, int *devIpiv, float *Workspace,
                                  int Lwork, int *devInfo)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    int arg = InverseArgument(n, A, lda, devIpiv, Workspace, Lwork, devInfo);
    if (arg)
    {
        return real_lu::Invalid(handle, devInfo, 1, arg);
    }
    if (n == 0)
    {
        return real_lu::EmptyInfo(handle, devInfo, 1);
    }
    auto status =
        real_lu::CheckSingle(handle, n, n, A, lda, devIpiv, Workspace, real_lu::InverseWorkspace(n), devInfo, true);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    bool identity = true;
    status = real_lu::CheckPivots(handle, devIpiv, n, devInfo, 1, &identity);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    status = real_lu::CheckFinite(handle, A, n, n, lda, 1, false, devInfo, 2);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    return LaunchInverse(handle, n, A, lda, devIpiv, Workspace, devInfo, identity);
}
