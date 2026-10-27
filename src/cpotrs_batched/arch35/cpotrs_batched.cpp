/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../utils/cholesky/cholesky.h"

extern "C" aclsolverStatus_t aclsolverCpotrsBatched(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, int nrhs,
                                                    const aclsolverFloatComplex* const aarray[], int lda,
                                                    aclsolverFloatComplex* barray[], int ldb, int* info, int batch)
{
    int arg = Cholesky::SolveArg(uplo, n, nrhs, aarray, lda, barray, ldb, batch == 0);
    if (!arg && n > 0 && batch > 0 && nrhs != 1)
    {
        arg = 3;
    }
    if (!arg && !info)
    {
        arg = 8;
    }
    if (!arg && (batch < 0 || batch > Cholesky::MAX_BATCH))
    {
        arg = 9;
    }
    auto status = Cholesky::Finish(handle, info, arg);
    if (status != ACLSOLVER_STATUS_SUCCESS || n == 0 || batch == 0)
    {
        return status;
    }
    aclrtStream stream = nullptr;
    aclsolverFloatComplex* work = nullptr;
    size_t bytes =
        static_cast<size_t>(Cholesky::SolveWorkspace(n, nrhs, batch)) * sizeof(aclsolverFloatComplex) * batch;
    status = Cholesky::AcquireScratch(handle, &stream, bytes, &work);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return status;
    }
    Cholesky::Solve(stream, Cholesky::IsUpper(uplo), n, nrhs, nullptr, aarray, lda, nullptr, barray, ldb, batch, work);
    return ACLSOLVER_STATUS_SUCCESS;
}
