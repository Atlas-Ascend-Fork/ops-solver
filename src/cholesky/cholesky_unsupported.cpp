/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "cann_ops_solver.h"

namespace
{
aclsolverStatus_t Unsupported(aclsolverHandle_t handle)
{
    return handle ? ACLSOLVER_STATUS_ARCH_MISMATCH : ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
}
}  // namespace

// Non-950 builds keep the public ABI without accessing device data or outputs.
extern "C"
{
    aclsolverStatus_t aclsolverSpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t, int, float *, int,
                                                 int *)
    {
        return Unsupported(handle);
    }
    aclsolverStatus_t aclsolverSpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t, int, float *, int,
                                                 int *)
    {
        return Unsupported(handle);
    }
    aclsolverStatus_t aclsolverSpotrf(aclsolverHandle_t handle, aclsolverFillMode_t, int, float *, int, float *, int,
                                      int *)
    {
        return Unsupported(handle);
    }
    aclsolverStatus_t aclsolverSpotri(aclsolverHandle_t handle, aclsolverFillMode_t, int, float *, int, float *, int,
                                      int *)
    {
        return Unsupported(handle);
    }
    aclsolverStatus_t aclsolverSpotrs(aclsolverHandle_t handle, aclsolverFillMode_t, int, int, const float *, int,
                                      float *, int, int *)
    {
        return Unsupported(handle);
    }
    aclsolverStatus_t aclsolverSpotrfBatched(aclsolverHandle_t handle, aclsolverFillMode_t, int, float *[], int, int *,
                                             int)
    {
        return Unsupported(handle);
    }
    aclsolverStatus_t aclsolverSpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t, int, int, float *[], int,
                                             float *[], int, int *, int)
    {
        return Unsupported(handle);
    }
}
