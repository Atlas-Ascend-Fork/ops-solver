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

#include "../../cunmqr/arch35/complex_qr_algorithms.h"
#include "../../cunmqr/arch35/cunmqr_internal.h"
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "../../utils/complex_qr/complex_qr_host.h"
#include "cungqr_internal.h"

namespace
{
bool Empty(const ComplexQrCall& c) { return c.m == 0 || c.n == 0; }
bool WorkspaceSize(const ComplexQrCall& c, size_t& bytes)
{
    uint64_t elements = 0, scratch = 0;
    if (!Empty(c))
    {
        if (!ComplexQrAddWorkspace(elements, c.m, c.k))
        {
            return false;
        }
        elements = ComplexQrAlignedElements(elements);
        if (ComplexQrBlockedApply(c.m, c.n, c.k) && !ComplexQrBlockWorkspace(c.m, c.n, scratch))
        {
            return false;
        }
    }
    return ComplexQrWorkspaceBytes(elements, scratch, bytes);
}
int CheckExecution(const ComplexQrCall&, int*, bool) { return 0; }
void Execute(float* a, float* tau, float*, float*, float* work, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrGenerate(a, tau, work, s, stream);
    ComplexQrInfo(info, 0, stream);
}
const ComplexQrPolicy POLICY = {
    ComplexQrCheckGenerate, Empty, WorkspaceSize, CheckExecution, CungqrValidate, Execute, 7, 9};
}  // namespace

extern "C" aclsolverStatus_t aclsolverCungqr_bufferSize(aclsolverHandle_t h, int m, int n, int k,
                                                        const aclsolverFloatComplex* a, int lda,
                                                        const aclsolverFloatComplex* tau, int* lwork)
{
    return ComplexQrQuery(POLICY, h,
                          {ComplexQrOperation::Generate, m, n, k, 0, 0, 0, lda, 0, 0, a, tau, nullptr, nullptr}, lwork);
}
extern "C" aclsolverStatus_t aclsolverCungqr(aclsolverHandle_t h, int m, int n, int k, aclsolverFloatComplex* a,
                                             int lda, const aclsolverFloatComplex* tau, aclsolverFloatComplex* work,
                                             int lwork, int* info)
{
    return ComplexQrRun(POLICY, h,
                        {ComplexQrOperation::Generate, m, n, k, 0, 0, 0, lda, 0, 0, a, tau, nullptr, nullptr}, work,
                        lwork < 0 ? SIZE_MAX : size_t(lwork) * sizeof(aclsolverFloatComplex), info);
}

int ComplexQrCheckGenerate(const ComplexQrCall& c)
{
    if (c.m < 0)
    {
        return 1;
    }
    if (c.n < 0 || c.n > c.m)
    {
        return 2;
    }
    if (c.k < 0 || c.k > c.n)
    {
        return 3;
    }
    if (c.m && c.n && !c.a)
    {
        return 4;
    }
    if (c.lda < std::max(1, c.m))
    {
        return 5;
    }
    if (c.k && !c.tau)
    {
        return 6;
    }
    return 0;
}

void ComplexQrGenerate(float* a, float* tau, float* work, ComplexQrShape s, void* stream)
{
    if (s.k)
    {
        ComplexQrCopyMatrix(a, work, s.m, s.k, s.lda, s.m, false, stream);
    }
    ComplexQrCopyMatrix(a, a, s.m, s.n, s.lda, s.lda, true, stream);
    s.ldb = s.lda;
    s.lda = s.m;
    float* scratch = s.k ? work + 2 * ComplexQrAlignedElements(uint64_t(s.m) * s.k) : work;
    ComplexQrApply(work, tau, a, scratch, s, stream);
}
