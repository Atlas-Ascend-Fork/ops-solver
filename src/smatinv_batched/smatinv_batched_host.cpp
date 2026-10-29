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
aclsolverStatus_t aclsolverSmatinvBatched(aclsolverHandle_t handle, int n, const float *const A[], int lda,
                                          float *const Ainv[], int lda_inv, int *info, int batchSize)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    int arg = real_lu::BatchedArgument(n, A, lda, nullptr, Ainv, lda_inv, info, batchSize, true);
    if (arg)
    {
        return real_lu::Invalid(handle, info, batchSize < 0 ? 1 : batchSize, arg);
    }
    if (batchSize == 0)
    {
        return ACLSOLVER_STATUS_SUCCESS;
    }
    if (n == 0)
    {
        return real_lu::EmptyInfo(handle, info, batchSize);
    }
    else
    {
        real_lu::BatchedLayout layout;
        auto status = real_lu::CheckArrays(handle, n, A, lda, nullptr, Ainv, lda_inv, info, batchSize, true, layout);
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
        status = real_lu::CheckFinite(handle, layout.contiguous ? layout.input : A, n, n, lda, batchSize,
                                      !layout.contiguous, info, 2);
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
        if (batchSize >= 8 && lda == n && lda_inv == n && layout.contiguous)
        {
            real_lu_small_do(layout.input, layout.output, nullptr, info, n, batchSize, true, false,
                             real_lu::Stream(handle));
        }
        else
        {
            real_lu_inverse_do(A, Ainv, nullptr, info, n, lda, lda_inv, batchSize, true, true, real_lu::Stream(handle));
        }
    }
    return real_lu::Launched();
}
