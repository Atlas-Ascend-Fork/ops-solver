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
constexpr const char* OP_NAME = "aclsolverCgetri";

int InvalidShape(int n, int lda)
{
    if (n < 0 || n > ComplexSolver::MAX_ORDER) return 1;
    if (lda < std::max(1, n)) return 3;
    return 0;
}

aclsolverStatus_t ValidateBuffers(int n, int lda, aclFloatComplex* a, int* pivots, aclFloatComplex* workspace,
                                  int* info, aclrtStream stream, uint32_t blocks)
{
    const void* pointers[] = {a, pivots, workspace};
    const int arguments[] = {2, 4, 5};
    const uint64_t bytes[] = {static_cast<uint64_t>(lda) * n * sizeof(*a), static_cast<uint64_t>(n) * sizeof(int),
                              static_cast<uint64_t>(n) * n * sizeof(*workspace)};
    const size_t alignments[] = {alignof(aclFloatComplex), alignof(int), alignof(aclFloatComplex)};
    for (int i = 0; i < 3; ++i)
    {
        auto status = ComplexSolver::PointerArgument(pointers[i], i == 1, OP_NAME, arguments[i], info, 1, stream,
                                                     blocks, bytes[i], alignments[i]);
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    }
    if (ComplexSolver::Overlap(a, static_cast<uint64_t>(lda) * n * sizeof(*a), workspace,
                               static_cast<uint64_t>(n) * n * sizeof(*workspace)))
        return ComplexSolver::Reject(OP_NAME, 5, info, 1, stream, blocks);
    return ACLSOLVER_STATUS_SUCCESS;
}
}  // namespace

aclsolverStatus_t aclsolverCgetri_bufferSize(aclsolverHandle_t handle, int n, aclFloatComplex* A, int lda, int* Lwork)
{
    aclrtStream stream = nullptr;
    uint32_t blocks = 0;
    auto status = ComplexSolver::Context(handle, stream, blocks, OP_NAME);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int bad = InvalidShape(n, lda);
    if (bad != 0) return ComplexSolver::Reject(OP_NAME, bad, nullptr, 0, stream, blocks);
    if (Lwork == nullptr) return ComplexSolver::Reject(OP_NAME, 4, nullptr, 0, stream, blocks);
    if (n > 0)
    {
        status = ComplexSolver::DeviceSpan(A, static_cast<uint64_t>(lda) * n * sizeof(*A), alignof(aclFloatComplex),
                                           OP_NAME, 2);
        if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    }
    *Lwork = n * n;
    return ACLSOLVER_STATUS_SUCCESS;
}

aclsolverStatus_t aclsolverCgetri(aclsolverHandle_t handle, int n, aclFloatComplex* A, int lda, int* devIpiv,
                                  aclFloatComplex* Workspace, int Lwork, int* devInfo)
{
    aclrtStream stream = nullptr;
    uint32_t blocks = 0;
    auto status = ComplexSolver::Context(handle, stream, blocks, OP_NAME);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int bad = InvalidShape(n, lda);
    if (bad != 0) return ComplexSolver::Reject(OP_NAME, bad, devInfo, 1, stream, blocks);
    if (Lwork < n * n) return ComplexSolver::Reject(OP_NAME, 6, devInfo, 1, stream, blocks);
    status = ComplexSolver::DeviceSpan(devInfo, static_cast<uint64_t>(1) * sizeof(int), alignof(int), OP_NAME, 7);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    if (n == 0)
    {
        ComplexSolverFillInfo(devInfo, 1, 0, 1, stream);
        return ComplexSolver::LaunchStatus(OP_NAME);
    }
    status = ValidateBuffers(n, lda, A, devIpiv, Workspace, devInfo, stream, blocks);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    ComplexSolverGetri({n, n, lda, n, 1}, A, devIpiv, Workspace, devInfo, false, blocks, stream);
    ComplexSolverCopyInverse(n, lda, Workspace, A, devInfo, blocks, stream);
    return ComplexSolver::CompleteGetri(devIpiv, devInfo, 1, stream, OP_NAME);
}
