/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>

#include "cann_ops_solver.h"

int main()
{
    // No ACL initialization is needed: unsupported entries must not dereference
    // the opaque handle, matrix pointers or output pointers.
    int handleStorage = 0;
    for (aclsolverHandle_t handle : {static_cast<void *>(nullptr), static_cast<void *>(&handleStorage)})
    {
        const auto expected = handle ? ACLSOLVER_STATUS_ARCH_MISMATCH : ACLSOLVER_STATUS_HANDLE_IS_NULLPTR;
        for (auto uplo : {ACLSOLVER_FILL_MODE_LOWER, ACLSOLVER_FILL_MODE_UPPER})
        {
            int lwork = -777, info = -777;
            float matrix = 123.0f;
            float *array[] = {&matrix};
            const aclsolverStatus_t results[] = {
                aclsolverSpotrf_bufferSize(handle, uplo, 1, &matrix, 1, &lwork),
                aclsolverSpotri_bufferSize(handle, uplo, 1, &matrix, 1, &lwork),
                aclsolverSpotrf(handle, uplo, 1, &matrix, 1, nullptr, 0, &info),
                aclsolverSpotri(handle, uplo, 1, &matrix, 1, nullptr, 0, &info),
                aclsolverSpotrs(handle, uplo, 1, 1, &matrix, 1, &matrix, 1, &info),
                aclsolverSpotrfBatched(handle, uplo, 1, array, 1, &info, 1),
                aclsolverSpotrsBatched(handle, uplo, 1, 1, array, 1, array, 1, &info, 1),
            };
            for (auto status : results)
            {
                if (status != expected)
                {
                    std::cerr << "Unexpected unsupported-architecture status: " << status << '\n';
                    return 1;
                }
            }
            if (lwork != -777 || info != -777 || matrix != 123.0f)
            {
                std::cerr << "Unsupported entry modified an output\n";
                return 1;
            }
        }
    }
    std::cout << "[PASS] seven Cholesky symbols: architecture status and unchanged outputs\n";
    return 0;
}
