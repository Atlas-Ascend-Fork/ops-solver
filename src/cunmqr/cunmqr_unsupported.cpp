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

extern "C" aclsolverStatus_t aclsolverCunmqr_bufferSize(aclsolverHandle_t handle, aclsolverSideMode_t,
                                                        aclsolverOperation_t, int, int, int,
                                                        const aclsolverFloatComplex*, int, const aclsolverFloatComplex*,
                                                        const aclsolverFloatComplex*, int, int*)
{
    return Unsupported(handle);
}
extern "C" aclsolverStatus_t aclsolverCunmqr(aclsolverHandle_t handle, aclsolverSideMode_t, aclsolverOperation_t, int,
                                             int, int, const aclsolverFloatComplex*, int, const aclsolverFloatComplex*,
                                             aclsolverFloatComplex*, int, aclsolverFloatComplex*, int, int*)
{
    return Unsupported(handle);
}
