/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <limits>

#include "cann_ops_solver.h"
#include "complex_qr.h"

struct ComplexQrCall
{
    ComplexQrOperation operation;
    int m;
    int n;
    int k;
    int rhs;
    int side;
    int trans;
    int lda;
    int ldb;
    int ldx;
    const aclsolverFloatComplex* a;
    const aclsolverFloatComplex* tau;
    const aclsolverFloatComplex* b;
    aclsolverFloatComplex* x;
};

// The common runtime owns resource/error handling only. Each operator supplies
// its argument contract, workspace layout, validation and execution callbacks.
struct ComplexQrPolicy
{
    int (*check)(const ComplexQrCall&);
    bool (*empty)(const ComplexQrCall&);
    bool (*workspace)(const ComplexQrCall&, size_t&);
    int (*executionCheck)(const ComplexQrCall&, int*, bool);
    void (*validate)(const float*, const float*, const float*, int*, ComplexQrShape, void*);
    void (*execute)(float*, float*, float*, float*, float*, int*, ComplexQrShape, void*);
    int workspaceArgument;
    int infoArgument;
};
inline bool ComplexQrAddWorkspace(uint64_t& elements, uint64_t count, uint64_t size)
{
    const uint64_t limit = (std::numeric_limits<size_t>::max() - 31) / sizeof(aclsolverFloatComplex);
    if (elements > limit)
    {
        return false;
    }
    if (count == 0)
    {
        return true;
    }
    if (size > (limit - elements) / count)
    {
        return false;
    }
    elements += count * size;
    return true;
}
aclsolverStatus_t ComplexQrQuery(const ComplexQrPolicy& policy, aclsolverHandle_t handle, const ComplexQrCall& call,
                                 int* elements);
aclsolverStatus_t ComplexQrQueryBytes(const ComplexQrPolicy& policy, aclsolverHandle_t handle,
                                      const ComplexQrCall& call, size_t* bytes);
aclsolverStatus_t ComplexQrRun(const ComplexQrPolicy& policy, aclsolverHandle_t handle, const ComplexQrCall& call,
                               void* workspace, size_t bytes, int* info, int* niter = nullptr);

inline bool ComplexQrWorkspaceBytes(uint64_t elements, uint64_t scratch, size_t& bytes)
{
    if (!ComplexQrAddWorkspace(elements, 1, scratch))
    {
        return false;
    }
    bytes = (elements * sizeof(aclsolverFloatComplex) + 31) & ~size_t(31);
    return true;
}
