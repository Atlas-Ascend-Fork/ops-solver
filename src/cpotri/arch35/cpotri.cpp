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

extern "C" aclsolverStatus_t aclsolverCpotri_bufferSize(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                                        aclsolverFloatComplex* a, int lda, int* lwork)
{
    return Cholesky::QueryWorkspace(handle, uplo, n, a, lda, lwork, true);
}

extern "C" aclsolverStatus_t aclsolverCpotri(aclsolverHandle_t handle, aclsolverFillMode_t uplo, int n,
                                             aclsolverFloatComplex* a, int lda, aclsolverFloatComplex* work, int lwork,
                                             int* info)
{
    return Cholesky::RunInPlace(handle, uplo, n, a, lda, work, lwork, info, true);
}
