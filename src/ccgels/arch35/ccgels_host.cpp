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

#include "../../cgeqrf/arch35/cgeqrf_internal.h"
#include "../../cungqr/arch35/cungqr_internal.h"
#include "../../cunmqr/arch35/complex_qr_algorithms.h"
#include "../../cunmqr/arch35/cunmqr_internal.h"
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "../../utils/complex_qr/complex_qr_host.h"
#include "../../utils/complex_solver_host.h"
#include "ccgels_internal.h"

namespace
{
// 文档任务范围（docs/zh/ccgels.md：1 <= n <= m <= 4096，1 <= nrhs <= 128，issue #369），
// 与实数版 sgels_host.cpp 的 GELS_MAX_DIM / nrhs 上限口径对齐。
constexpr int CC_GELS_MAX_ORDER = 4096;
constexpr int CC_GELS_MAX_RHS = 128;
}  // namespace

namespace
{
bool Empty(const ComplexQrCall& c) { return c.m == 0 || c.n == 0 || c.rhs == 0; }
bool WorkspaceSize(const ComplexQrCall& c, size_t& bytes)
{
    uint64_t elements = 0, scratch = 0;
    if (!Empty(c))
    {
        if (!ComplexQrAddWorkspace(elements, c.m, c.rhs))
        {
            return false;
        }
        elements = ComplexQrAlignedElements(elements);
        if (!ComplexQrAddWorkspace(elements, 1, c.n))
        {
            return false;
        }
        elements = ComplexQrAlignedElements(elements);
        if (ComplexQrBlockedApply(c.m, c.rhs, c.n) && !ComplexQrBlockWorkspace(c.m, c.rhs, scratch))
        {
            return false;
        }
        uint64_t factorScratch = 0;
        if (!ComplexQrFactorWorkspace(c, factorScratch))
        {
            return false;
        }
        scratch = std::max(scratch, factorScratch);
    }
    return ComplexQrWorkspaceBytes(elements, scratch, bytes);
}
int CheckExecution(const ComplexQrCall&, int* niter, bool afterWorkspace) { return afterWorkspace && !niter ? 12 : 0; }
void Execute(float* a, float*, float* b, float* x, float* work, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrLeastSquares(a, b, x, work, info, s, stream);
}
const ComplexQrPolicy POLICY = {
    ComplexQrCheckLeastSquares, Empty, WorkspaceSize, CheckExecution, CcgelsValidate, Execute, 10, 13};
}  // namespace

extern "C" aclsolverStatus_t aclsolverCCgels_bufferSize(aclsolverHandle_t h, int m, int n, int nrhs,
                                                        aclsolverFloatComplex* a, int lda, aclsolverFloatComplex* b,
                                                        int ldb, aclsolverFloatComplex* x, int ldx, void* dwork,
                                                        size_t* lwork)
{
    (void)dwork;
    return ComplexQrQueryBytes(
        POLICY, h, {ComplexQrOperation::LeastSquares, m, n, 0, nrhs, 0, 0, lda, ldb, ldx, a, nullptr, b, x}, lwork);
}
extern "C" aclsolverStatus_t aclsolverCCgels(aclsolverHandle_t h, int m, int n, int nrhs, aclsolverFloatComplex* a,
                                             int lda, aclsolverFloatComplex* b, int ldb, aclsolverFloatComplex* x,
                                             int ldx, void* work, size_t lwork, int* niter, int* info)
{
    return ComplexQrRun(POLICY, h,
                        {ComplexQrOperation::LeastSquares, m, n, 0, nrhs, 0, 0, lda, ldb, ldx, a, nullptr, b, x}, work,
                        lwork, info, niter);
}

// dX 独立性与任务范围校验（issue #368/#369），由 ComplexQrCheckLeastSquares 在 active
// 时调用；返回 0 表示通过，否则为参数错误码。
// - dX 必须是独立输出（issue #368，对齐实数版 sgels 的别名拒绝口径）：回代边读 dA 中的
//   R 边写 dX，同址或区间交叠会把未读取的 R 覆盖成解，静默算错。按字节区间判定（列主序
//   逻辑足迹），避免仅比较指针相等放过部分重叠。
// - 文档任务范围强制（issue #369）：超范围形状的 workspace 布局与 kernel 分块未定义；
//   执行与 bufferSize 查询共用本校验，口径一致。
static int CheckLeastSquaresAliasAndRange(const ComplexQrCall& c)
{
    const uint64_t xBytes = static_cast<uint64_t>(c.rhs) * c.ldx * sizeof(aclsolverFloatComplex);
    const uint64_t aBytes = static_cast<uint64_t>(c.n) * c.lda * sizeof(aclsolverFloatComplex);
    const uint64_t bBytes = static_cast<uint64_t>(c.rhs) * c.ldb * sizeof(aclsolverFloatComplex);
    if (ComplexSolver::Overlap(c.x, xBytes, c.a, aBytes) || ComplexSolver::Overlap(c.x, xBytes, c.b, bBytes))
    {
        return 8;
    }
    if (c.rhs > CC_GELS_MAX_RHS)
    {
        return 3;
    }
    if (c.m > CC_GELS_MAX_ORDER || c.n > CC_GELS_MAX_ORDER)
    {
        return 1;
    }
    return 0;
}

int ComplexQrCheckLeastSquares(const ComplexQrCall& c)
{
    if (c.m < 0 || c.m < c.n)
    {
        return 1;
    }
    if (c.n < 0)
    {
        return 2;
    }
    if (c.rhs < 0)
    {
        return 3;
    }
    const bool active = c.m && c.n && c.rhs;
    if (active && !c.a)
    {
        return 4;
    }
    if (c.lda < std::max(1, c.m))
    {
        return 5;
    }
    if (active && !c.b)
    {
        return 6;
    }
    if (c.ldb < std::max(1, c.m))
    {
        return 7;
    }
    if (active && !c.x)
    {
        return 8;
    }
    if (c.ldx < std::max(1, c.n))
    {
        return 9;
    }
    return active ? CheckLeastSquaresAliasAndRange(c) : 0;
}

void ComplexQrLeastSquares(float* a, float* b, float* x, float* work, int* info, ComplexQrShape s, void* stream)
{
    float* y = work;
    float* factors = work + 2 * ComplexQrAlignedElements(uint64_t(s.m) * s.rhs);
    float* scratch = factors + 2 * ComplexQrAlignedElements(s.n);
    ComplexQrCopyMatrix(b, y, s.m, s.rhs, s.ldb, s.m, false, stream);
    ComplexQrFactor(a, factors, scratch, s, stream);
    ComplexQrShape apply = s;
    apply.k = s.n;
    apply.n = s.rhs;
    apply.ldb = s.m;
    apply.trans = 2;
    ComplexQrApply(a, factors, y, scratch, apply, stream);
    LaunchSolveKernel(ComplexQrBlocks(s.rhs), stream, a, y, x, info, s);
}
