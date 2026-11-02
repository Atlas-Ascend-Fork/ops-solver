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
constexpr const char* OP_NAME = "aclsolverCgetriBatched";

int InvalidShape(int n, int lda, int ldc, int batch)
{
    if (n < 0 || n > ComplexSolver::MAX_ORDER) return 1;
    if (lda < std::max(1, n)) return 3;
    if (ldc < std::max(1, n)) return 6;
    if (batch < 0 || batch > ComplexSolver::MAX_BATCH) return 8;
    return 0;
}
}  // namespace

aclsolverStatus_t aclsolverCgetriBatched(aclsolverHandle_t handle, int n, const aclFloatComplex* const A[], int lda,
                                         const int* PivotArray, aclFloatComplex* const C[], int ldc, int* info,
                                         int batchSize)
{
    aclrtStream stream = nullptr;
    uint32_t blocks = 0;
    auto status = ComplexSolver::Context(handle, stream, blocks, OP_NAME);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int bad = InvalidShape(n, lda, ldc, batchSize);
    if (bad != 0)
        return ComplexSolver::Reject(
            OP_NAME, bad, info, ComplexSolver::ErrorInfoCount(batchSize, ComplexSolver::MAX_BATCH), stream, blocks);
    if (batchSize == 0) return ACLSOLVER_STATUS_SUCCESS;
    status = ComplexSolver::DeviceSpan(info, static_cast<uint64_t>(batchSize) * sizeof(int), alignof(int), OP_NAME, 7);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    if (n == 0)
    {
        ComplexSolverFillInfo(info, batchSize, 0, std::min<uint32_t>(blocks, batchSize), stream);
        return ComplexSolver::LaunchStatus(OP_NAME);
    }
    status = ComplexSolver::PointerArgument(A, false, OP_NAME, 2, info, batchSize, stream, blocks,
                                            static_cast<uint64_t>(batchSize) * sizeof(uint64_t), alignof(uint64_t));
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    status = ComplexSolver::PointerArgument(C, false, OP_NAME, 5, info, batchSize, stream, blocks,
                                            static_cast<uint64_t>(batchSize) * sizeof(uint64_t), alignof(uint64_t));
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    status = ComplexSolver::PointerArgument(PivotArray, true, OP_NAME, 4, info, batchSize, stream, blocks,
                                            static_cast<uint64_t>(n) * batchSize * sizeof(int), alignof(int));
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    ComplexSolverShape shape{n, n, lda, ldc, batchSize};
    status = ComplexSolver::ValidateMatrices(A, C, shape, 2, 5, bad, stream, OP_NAME);
    if (status == ACLSOLVER_STATUS_INVALID_VALUE)
        return ComplexSolver::Reject(OP_NAME, bad, info, batchSize, stream, blocks);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    ComplexSolverGetri(shape, A, PivotArray, C, info, true, blocks, stream);
    return ComplexSolver::CompleteGetri(PivotArray, info, batchSize, stream, OP_NAME);
}
