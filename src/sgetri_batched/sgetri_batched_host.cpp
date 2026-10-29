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
namespace
{
aclsolverStatus_t LaunchBlockedBatch(aclsolverHandle_t handle, int n, const float *const a[], int lda, const int *piv,
                                     float *const c[], int ldc, int *info, int batch, bool identity,
                                     _real_lu_workspace &cache)
{
    int chunk = real_lu::InverseBatchChunk(n, batch);
    auto status = real_lu::ScratchAfterValidation(
        cache, static_cast<size_t>(real_lu::InverseWorkspace(n)) * chunk * sizeof(float));
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    int launched = 0;
    for (int first = 0; first < batch;)
    {
        int count = std::min(chunk, batch - first);
        launched = real_lu_blocked_inverse_do(
            a + first, c + first, piv == nullptr ? nullptr : piv + static_cast<size_t>(first) * n, info + first,
            cache.data, n, lda, ldc, count, true, identity, real_lu::Stream(handle));
        if (launched != 0)
        {
            break;
        }
        first += count;
    }
    status = real_lu::ScratchFinished(handle, cache);
    return status != ACLSOLVER_STATUS_SUCCESS || launched != 0 ? ACLSOLVER_STATUS_EXECUTION_FAILED
                                                               : ACLSOLVER_STATUS_SUCCESS;
}
aclsolverStatus_t RunInverseBatch(aclsolverHandle_t handle, int n, const float *const a[], int lda, const int *piv,
                                  float *const c[], int ldc, int *info, int batch)
{
    real_lu::BatchedLayout layout;
    auto status = real_lu::CheckArrays(handle, n, a, lda, piv, c, ldc, info, batch, false, layout);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    _real_lu_workspace *cache = nullptr;
    if (real_lu::BlockedInverse(n))
    {
        status = real_lu::ScratchBeforeValidation(handle, cache);
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
    }
    status = real_lu::CheckFinite(handle, layout.contiguous ? layout.input : a, n, n, lda, batch, !layout.contiguous,
                                  info, 2);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    if (real_lu::BlockedInverse(n))
    {
        status = LaunchBlockedBatch(handle, n, a, lda, piv, c, ldc, info, batch, layout.identityPivots, *cache);
        if (status != ACLSOLVER_STATUS_SUCCESS)
        {
            return status;
        }
    }
    else if (n <= 32 && batch >= 8 && lda == n && ldc == n && layout.contiguous)
    {
        real_lu_small_do(layout.input, layout.output, piv, info, n, batch, false, layout.identityPivots,
                         real_lu::Stream(handle));
    }
    else
    {
        real_lu_inverse_do(a, c, piv, info, n, lda, ldc, batch, true, false, real_lu::Stream(handle));
    }
    return real_lu::Launched();
}
}  // namespace

aclsolverStatus_t aclsolverSgetriBatched(aclsolverHandle_t handle, int n, const float *const Aarray[], int lda,
                                         const int *PivotArray, float *const Carray[], int ldc, int *infoArray,
                                         int batchSize)
{
    if (handle == nullptr)
    {
        return ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
    }
    int arg = real_lu::BatchedArgument(n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize, false);
    if (arg)
    {
        return real_lu::Invalid(handle, infoArray, batchSize < 0 ? 1 : batchSize, arg);
    }
    if (batchSize == 0)
    {
        return ACLSOLVER_STATUS_SUCCESS;
    }
    if (n == 0)
    {
        return real_lu::EmptyInfo(handle, infoArray, batchSize);
    }
    return RunInverseBatch(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize);
}
