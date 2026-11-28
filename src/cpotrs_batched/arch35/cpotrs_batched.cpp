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
#include "../../utils/complex_solver_host.h"

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
    // info 回写前校验设备范围（issue #670，对齐实数 spotrfBatched 的 CheckInfoSpan
    // 口径）：Finish 预写 info[0]，batch>0 时后续 Solve 错误/成功路径按 batch 全量
    // 写 info；Host 指针或容量不足会产生设备侧越界写，校验失败直接返回。
    if (info)
    {
        auto span = ComplexSolver::DeviceSpan(info, static_cast<uint64_t>(batch > 0 ? batch : 1) * sizeof(int),
                                              alignof(int), "aclsolverCpotrsBatched", 8);
        if (span != ACLSOLVER_STATUS_SUCCESS)
        {
            return span;
        }
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
    // Kernel launches are asynchronous; do not report success when enqueueing failed
    // (issue #366，对齐实数 Cholesky LaunchStatus 与 getrs/QR 族口径)。
    return ComplexSolver::LaunchStatus("aclsolverCpotrsBatched");
}
