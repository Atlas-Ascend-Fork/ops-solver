/* Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <stdint.h>
#include <stdio.h>

#include "cann_ops_solver.h"

int main(void)
{
    for (int valid = 0; valid < 2; ++valid)
    {
        /* No handle internals or device pointer may be accessed on this path. */
        aclsolverHandle_t h = (aclsolverHandle_t)(uintptr_t)valid;
        aclsolverFloatComplex data = {3.0f, -7.0f};
        int info = 123, work = 456, niter = 789;
        size_t bytes = 987;
        aclsolverStatus_t expected = valid ? ACLSOLVER_STATUS_ARCH_MISMATCH : ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        aclsolverStatus_t results[] = {
            aclsolverCgeqrf_bufferSize(h, 1, 1, &data, 1, &work),
            aclsolverCgeqrf(h, 1, 1, &data, 1, &data, &data, 1, &info),
            aclsolverCungqr_bufferSize(h, 1, 1, 1, &data, 1, &data, &work),
            aclsolverCungqr(h, 1, 1, 1, &data, 1, &data, &data, 1, &info),
            aclsolverCunmqr_bufferSize(h, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_N, 1, 1, 1, &data, 1, &data, &data, 1,
                                       &work),
            aclsolverCunmqr(h, ACLSOLVER_SIDE_LEFT, ACLSOLVER_OP_N, 1, 1, 1, &data, 1, &data, &data, 1, &data, 1,
                            &info),
            aclsolverCCgels_bufferSize(h, 1, 1, 1, &data, 1, &data, 1, &data, 1, &data, &bytes),
            aclsolverCCgels(h, 1, 1, 1, &data, 1, &data, 1, &data, 1, &data, 8, &niter, &info),
        };
        for (size_t index = 0; index < sizeof(results) / sizeof(results[0]); ++index)
        {
            if (results[index] != expected)
            {
                fprintf(stderr, "ABI case %zu handle=%d returned %d, expected %d\n", index, valid, results[index],
                        expected);
                return 1;
            }
        }
        if (data.real != 3.0f || data.imag != -7.0f || info != 123 || work != 456 || niter != 789 || bytes != 987)
        {
            fprintf(stderr, "Unsupported architecture modified an output\n");
            return 1;
        }
    }
    puts("All eight ABI symbols: null-handle and unsupported-architecture contracts passed");
    return 0;
}
