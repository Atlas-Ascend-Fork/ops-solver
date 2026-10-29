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
#include "tiling/platform/platform_ascendc.h"

extern "C" int real_lu_vector_cores()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform == nullptr ? 1 : std::max(1, static_cast<int>(platform->GetCoreNumAiv()));
}
extern "C" int real_lu_cube_cores()
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    return platform == nullptr ? 1 : std::max(1, static_cast<int>(platform->GetCoreNumAic()));
}

aclsolverStatus_t aclsolverSgetrf_bufferSize(aclsolverHandle_t handle, int m, int n, float *A, int lda, int *Lwork)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    if (m < 0 || m > 8192 || n < 0 || lda < std::max(1, m) || Lwork == nullptr || real_lu::Device(Lwork) ||
        (A != nullptr && !real_lu::Device(A)))
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    int size = real_lu::FactorWorkspace(m, n);
    if (size < 0)
    {
        return ACLSOLVER_STATUS_INVALID_VALUE;
    }
    *Lwork = m == 0 || n == 0 ? 0 : size;
    return ACLSOLVER_STATUS_SUCCESS;
}

namespace
{
int FactorArgument(int m, int n, float *a, int lda, float *workspace, int *piv, int *info)
{
    if (m < 0 || m > 8192)
    {
        return 1;
    }
    if (n < 0 || real_lu::FactorWorkspace(m, n) < 0)
    {
        return 2;
    }
    if (m > 0 && n > 0 && !real_lu::Device(a))
    {
        return 3;
    }
    if (lda < std::max(1, m))
    {
        return 4;
    }
    if (m > 0 && n > 0 && !real_lu::Device(workspace))
    {
        return 5;
    }
    if (piv != nullptr && !real_lu::Device(piv))
    {
        return 6;
    }
    return real_lu::Device(info) ? 0 : 7;
}
aclsolverStatus_t LaunchFactor(aclsolverHandle_t handle, int m, int n, float *a, int lda, float *workspace, int *piv,
                               int *info)
{
    if (real_lu::ThinFactor(m, n))
    {
        real_lu_thin_factor_do(a, workspace, m, n, lda, piv, info, real_lu::Stream(handle));
    }
    else if (real_lu::BlockedFactor(m, n))
    {
        if (real_lu_blocked_factor_do(a, workspace, piv, info, m, n, lda, real_lu::Stream(handle)) != 0)
        {
            return ACLSOLVER_STATUS_EXECUTION_FAILED;
        }
    }
    else
    {
        real_lu_factor_do(a, m, n, lda, piv, info, real_lu::Stream(handle));
    }
    return real_lu::Launched();
}
}  // namespace

aclsolverStatus_t aclsolverSgetrf(aclsolverHandle_t handle, int m, int n, float *A, int lda, float *Workspace,
                                  int *devIpiv, int *devInfo)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    int arg = FactorArgument(m, n, A, lda, Workspace, devIpiv, devInfo);
    if (arg)
    {
        return real_lu::Invalid(handle, devInfo, 1, arg);
    }
    if (m == 0 || n == 0)
    {
        return real_lu::EmptyInfo(handle, devInfo, 1);
    }
    auto status =
        real_lu::CheckSingle(handle, m, n, A, lda, devIpiv, Workspace, real_lu::FactorWorkspace(m, n), devInfo, false);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    status = real_lu::CheckFinite(handle, A, m, n, lda, 1, false, devInfo, 3);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    return LaunchFactor(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}
