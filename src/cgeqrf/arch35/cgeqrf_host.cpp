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
#include "../../cunmqr/arch35/cunmqr_launch.h"
#include "../../utils/complex_qr/complex_qr_host.h"
#include "cgeqrf_internal.h"

namespace
{
bool Empty(const ComplexQrCall& c) { return c.m == 0 || c.n == 0; }
bool WorkspaceSize(const ComplexQrCall& c, size_t& bytes)
{
    uint64_t elements = 0, scratch = 0;
    if (!Empty(c))
    {
        if (!ComplexQrFactorWorkspace(c, scratch))
        {
            return false;
        }
    }
    return ComplexQrWorkspaceBytes(elements, scratch, bytes);
}
int CheckExecution(const ComplexQrCall& c, int*, bool afterWorkspace)
{
    return !afterWorkspace && !Empty(c) && !c.tau ? 5 : 0;
}
void Execute(float* a, float* tau, float*, float*, float* work, int* info, ComplexQrShape s, void* stream)
{
    ComplexQrFactor(a, tau, work, s, stream);
    ComplexQrInfo(info, 0, stream);
}
const ComplexQrPolicy POLICY = {
    ComplexQrCheckFactor, Empty, WorkspaceSize, CheckExecution, CgeqrfValidate, Execute, 6, 8};
}  // namespace

extern "C" aclsolverStatus_t aclsolverCgeqrf_bufferSize(aclsolverHandle_t h, int m, int n, aclsolverFloatComplex* a,
                                                        int lda, int* lwork)
{
    return ComplexQrQuery(
        POLICY, h, {ComplexQrOperation::Factor, m, n, 0, 0, 0, 0, lda, 0, 0, a, nullptr, nullptr, nullptr}, lwork);
}
extern "C" aclsolverStatus_t aclsolverCgeqrf(aclsolverHandle_t h, int m, int n, aclsolverFloatComplex* a, int lda,
                                             aclsolverFloatComplex* tau, aclsolverFloatComplex* work, int lwork,
                                             int* info)
{
    return ComplexQrRun(POLICY, h, {ComplexQrOperation::Factor, m, n, 0, 0, 0, 0, lda, 0, 0, a, tau, nullptr, nullptr},
                        work, lwork < 0 ? SIZE_MAX : size_t(lwork) * sizeof(aclsolverFloatComplex), info);
}

int ComplexQrCheckFactor(const ComplexQrCall& c)
{
    if (c.m < 0)
    {
        return 1;
    }
    if (c.n < 0)
    {
        return 2;
    }
    if (c.m && c.n && !c.a)
    {
        return 3;
    }
    if (c.lda < std::max(1, c.m))
    {
        return 4;
    }
    return 0;
}

bool ComplexQrFactorWorkspace(const ComplexQrCall& c, uint64_t& scratch)
{
    scratch = 0;
    return !(c.m > 64 && c.n > 32) || ComplexQrBlockWorkspace(c.m, c.n, scratch);
}

static void FactorPanel(float* a, float* tau, ComplexQrShape s, void* stream, uint32_t first, uint32_t end)
{
    for (uint32_t inner = first; inner < end;)
    {
        uint32_t width = 8;
        while (width > 1 && uint64_t(s.m - inner) * width > 16384)
        {
            width /= 2;
        }
        uint32_t innerEnd = inner + width < end ? inner + width : end;
        LaunchFactorPanelKernel(1, stream, a, tau, s.m, innerEnd, s.lda, inner, innerEnd);
        if (innerEnd < end)
        {
            if (s.m - inner <= 8192 && uint64_t(s.m - inner) * (innerEnd - inner) <= 16384)
            {
                LaunchCachedApplyKernel(ComplexQrVectorCores(), stream, a, tau, a, s.lda, s.lda, s.m - inner,
                                        end - innerEnd, inner, innerEnd - inner, innerEnd);
            }
            else
            {
                for (uint32_t k = inner; k < innerEnd; ++k)
                {
                    LaunchApplyKernel(ComplexQrBlocks(end - innerEnd), stream, a, tau, a, s.m, s.lda, s.lda, k,
                                      innerEnd, end - innerEnd, false, true);
                }
            }
        }
        inner = innerEnd;
    }
}

void ComplexQrFactor(float* a, float* tau, float* work, ComplexQrShape s, void* stream)
{
    uint32_t count = s.m < s.n ? s.m : s.n;
    if (uint64_t((s.m + 3) / 4 * 4) * s.n <= 16384)
    {
        LaunchFactorPanelKernel(1, stream, a, tau, s.m, s.n, s.lda, 0, count);
        return;
    }
    if (s.m > 64 && s.n > 1)
    {
        for (uint32_t first = 0; first < count; first += COMPLEX_QR_PANEL)
        {
            uint32_t end = first + COMPLEX_QR_PANEL < count ? first + COMPLEX_QR_PANEL : count;
            FactorPanel(a, tau, s, stream, first, end);
            if (end < s.n &&
                !ComplexQrBlockUpdate(a, tau, a, work, s.m, s.n - end, s.lda, s.lda, first, end, end, true, stream))
            {
                for (uint32_t k = first; k < end; ++k)
                {
                    LaunchApplyKernel(ComplexQrBlocks(s.n - end), stream, a, tau, a, s.m, s.lda, s.lda, k, end,
                                      s.n - end, false, true);
                }
            }
        }
        return;
    }
    for (uint32_t k = 0; k < count; ++k)
    {
        LaunchReflectorKernel(1, stream, a, tau, s.m, s.lda, k);
        if (k + 1 < s.n)
        {
            LaunchApplyKernel(ComplexQrBlocks(s.n - k - 1), stream, a, tau, a, s.m, s.lda, s.lda, k, k + 1, s.n - k - 1,
                              false, true);
        }
    }
}
