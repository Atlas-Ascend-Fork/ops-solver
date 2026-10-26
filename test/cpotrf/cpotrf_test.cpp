/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file cpotrf_test.cpp
 * \brief Test program for aclsolverCpotrf (Cholesky factorization)
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

int main(int argc, char** argv)
{
    int deviceId, N;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    N = (argc > 2) ? std::atoi(argv[2]) : 128;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = 2 * N * N * sizeof(float);

    std::complex<float>* A = nullptr;
    auto cleanup = [&]() -> aclError
    {
        if (A != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(A));
            A = nullptr;
        }
        CHECK_ACL(aclrtDestroyStream(stream));
        if (handle != nullptr)
        {
            CHECK_ACL(aclsolverDestroy(handle));
            handle = nullptr;
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return ACL_SUCCESS;
    };

    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    CHECK_RET(ReadFile("./test/cpotrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t*)A, N, 2 * N, 8, 8);

    int32_t infoVal = 0;
    int32_t* info = &infoVal;

    // Upper Cholesky
    auto ret = aclsolverCpotrf(handle, 'U', N, A, N, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCpotrf (U) failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] A (U):" << std::endl;
    PrintPartOfMatrix<float>((uint8_t*)A, N, 2 * N, 8, 8);

    WriteFile("./test/cpotrf/data/output/U_gm.bin", A, aMatrixFileSize);

    // Re-read original matrix for lower factorization
    CHECK_RET(ReadFile("./test/cpotrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    ret = aclsolverCpotrf(handle, 'L', N, A, N, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCpotrf (L) failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] A (L):" << std::endl;
    PrintPartOfMatrix<float>((uint8_t*)A, N, 2 * N, 8, 8);

    WriteFile("./test/cpotrf/data/output/L_gm.bin", A, aMatrixFileSize);

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
