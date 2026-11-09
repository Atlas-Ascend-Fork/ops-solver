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
constexpr const char* OP_NAME = "aclsolverCmatinvBatched";

int InvalidShape(int n, int lda, int ldc, int batch)
{
    if (n < 0 || n > 32) return 1;
    if (lda < std::max(1, n)) return 3;
    if (ldc < std::max(1, n)) return 5;
    if (batch < 0 || batch > ComplexSolver::MAX_BATCH_MATINV) return 7;
    return 0;
}
}  // namespace

aclsolverStatus_t aclsolverCmatinvBatched(aclsolverHandle_t handle, int n, const aclFloatComplex* const A[], int lda,
                                          aclFloatComplex* const C[], int ldc, int* info, int batchSize)
{
    aclrtStream stream = nullptr;
    uint32_t blocks = 0;
    auto status = ComplexSolver::Context(handle, stream, blocks, OP_NAME);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    int bad = InvalidShape(n, lda, ldc, batchSize);
    if (bad != 0)
        return ComplexSolver::Reject(OP_NAME, bad, info,
                                     ComplexSolver::ErrorInfoCount(batchSize, ComplexSolver::MAX_BATCH_MATINV), stream,
                                     blocks);
    if (batchSize == 0) return ACLSOLVER_STATUS_SUCCESS;
    status = ComplexSolver::DeviceSpan(info, static_cast<uint64_t>(batchSize) * sizeof(int), alignof(int), OP_NAME, 6);
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    if (n == 0)
    {
        ComplexSolverFillInfo(info, batchSize, 0, std::min<uint32_t>(blocks, batchSize), stream);
        return ComplexSolver::LaunchStatus(OP_NAME);
    }
    status = ComplexSolver::PointerArgument(A, false, OP_NAME, 2, info, batchSize, stream, blocks,
                                            static_cast<uint64_t>(batchSize) * sizeof(uint64_t), alignof(uint64_t));
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    status = ComplexSolver::PointerArgument(C, false, OP_NAME, 4, info, batchSize, stream, blocks,
                                            static_cast<uint64_t>(batchSize) * sizeof(uint64_t), alignof(uint64_t));
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    ComplexSolverShape shape{n, n, lda, ldc, batchSize};
    // 跨缓冲重叠校验（issue #371，对齐实数 batched 家族全区间口径）：info 与 A/C 指针
    // 数组指向的全部矩阵区间比对。归参为 info 自身时其区间不可信，不回写 info、直接
    // 返回 INVALID_VALUE（实数族 j==3 同口径）。
    const ComplexSolver::ExtraView extraViews[] = {
        {info, static_cast<uint64_t>(batchSize) * sizeof(int), 6},
    };
    status = ComplexSolver::ValidateMatrices(A, C, shape, 2, 4, bad, stream, OP_NAME, extraViews, 1);
    if (status == ACLSOLVER_STATUS_INVALID_VALUE)
    {
        return bad == 6 ? status : ComplexSolver::Reject(OP_NAME, bad, info, batchSize, stream, blocks);
    }
    if (status != ACLSOLVER_STATUS_SUCCESS) return status;
    ComplexSolverMatinv(shape, A, C, info, blocks, stream);
    return ComplexSolver::LaunchStatus(OP_NAME);
}
