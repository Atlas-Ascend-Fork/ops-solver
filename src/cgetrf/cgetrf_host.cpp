/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <new>
#include <vector>

#include "../utils/complex_solver_host.h"

namespace
{
constexpr const char* OP_NAME = "aclsolverCgetrf";

int InvalidShape(int m, int n, int lda)
{
    if (m < 0 || m > ComplexSolver::MAX_ORDER) return 1;
    if (n < 0 || n > 8192) return 2;
    if (lda < std::max(1, m)) return 4;
    return 0;
}
}  // namespace

aclsolverStatus_t aclsolverCgetrf_bufferSize(aclsolverHandle_t handle, int m, int n, aclFloatComplex* A, int lda,
                                             int* Lwork)
{
    aclrtStream stream = nullptr;
    uint32_t blocks = 0;
    auto status = ComplexSolver::Context(handle, stream, blocks, OP_NAME);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int bad = InvalidShape(m, n, lda);
    if (bad != 0) return ComplexSolver::Reject(OP_NAME, bad, nullptr, 0, stream, blocks);
    if (Lwork == nullptr) return ComplexSolver::Reject(OP_NAME, 5, nullptr, 0, stream, blocks);
    if (m > 0 && n > 0)
    {
        status = ComplexSolver::DeviceSpan(A, static_cast<uint64_t>(lda) * n * sizeof(*A), alignof(aclFloatComplex),
                                           OP_NAME, 3);
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    }
    // Distributed panels use a reusable two-plane panel and synchronization scratch.
    *Lwork = ComplexSolverGetrfWorkspace(m, n);
    return ACLSOLVER_STATUS_SUCCESS;
}

aclsolverStatus_t aclsolverCgetrf(aclsolverHandle_t handle, int m, int n, aclFloatComplex* A, int lda,
                                  aclFloatComplex* Workspace, int* devIpiv, int* devInfo)
{
    aclrtStream stream = nullptr;
    uint32_t blocks = 0;
    auto status = ComplexSolver::Context(handle, stream, blocks, OP_NAME);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int bad = InvalidShape(m, n, lda);
    if (bad != 0) return ComplexSolver::Reject(OP_NAME, bad, devInfo, 1, stream, blocks);
    status = ComplexSolver::DeviceSpan(devInfo, static_cast<uint64_t>(1) * sizeof(int), alignof(int), OP_NAME, 7);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    if (m > 0 && n > 0)
    {
        status = ComplexSolver::PointerArgument(A, false, OP_NAME, 3, devInfo, 1, stream, blocks,
                                                static_cast<uint64_t>(lda) * n * sizeof(*A), alignof(aclFloatComplex));
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
        status = ComplexSolver::PointerArgument(devIpiv, true, OP_NAME, 6, devInfo, 1, stream, blocks,
                                                static_cast<uint64_t>(std::min(m, n)) * sizeof(int), alignof(int));
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
        int work = ComplexSolverGetrfWorkspace(m, n);
        status = ComplexSolver::PointerArgument(Workspace, work == 0, OP_NAME, 5, devInfo, 1, stream, blocks,
                                                static_cast<uint64_t>(work) * sizeof(aclFloatComplex),
                                                alignof(aclFloatComplex));
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    }
    ComplexSolverFillInfo(devInfo, 1, 0, 1, stream);
    if (m > 0 && n > 0) ComplexSolverGetrf({m, n, lda, 0, 1}, A, Workspace, devIpiv, devInfo, blocks, stream);
    return ComplexSolver::LaunchStatus(OP_NAME);
}
