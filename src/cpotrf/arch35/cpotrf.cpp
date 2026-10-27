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

namespace Cholesky
{
aclsolverStatus_t RunInPlace(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n, aclsolverFloatComplex* a,
                             int lda, aclsolverFloatComplex* work, int lwork, int* info, bool inverse)
{
    int arg = FactorArg(uplo, n, a, lda);
    int required = arg ? 0 : (inverse ? InverseWorkspace(n) : FactorWorkspace(n));
    bool needsWork = inverse ? n > 0 : lwork > 0 || required > 0;
    if (!arg && !work && needsWork)
    {
        arg = 5;
    }
    if (!arg && lwork < required)
    {
        arg = 6;
    }
    if (!arg && !info)
    {
        arg = 7;
    }
    auto status = Finish(handle, info, arg);
    if (status != ACLSOLVER_STATUS_SUCCESS || n == 0)
    {
        return status;
    }
    aclrtStream stream = nullptr;
    status = aclsolverGetStream(handle, &stream);
    if (status != ACLSOLVER_STATUS_SUCCESS)
    {
        return ReportError(status, "cannot obtain Cholesky stream");
    }
    if (inverse)
    {
        Inverse(stream, IsUpper(uplo), n, a, lda, work, info);
    }
    else
    {
        Factor(stream, IsUpper(uplo), n, a, nullptr, lda, info, 1, work);
    }
    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace Cholesky

extern "C" aclsolverStatus_t aclsolverCpotrf_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                                        aclsolverFloatComplex* a, int lda, int* lwork)
{
    return Cholesky::QueryWorkspace(handle, uplo, n, a, lda, lwork, false);
}

extern "C" aclsolverStatus_t aclsolverCpotrf(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                             aclsolverFloatComplex* a, int lda, aclsolverFloatComplex* work, int lwork,
                                             int* info)
{
    return Cholesky::RunInPlace(handle, uplo, n, a, lda, work, lwork, info, false);
}
