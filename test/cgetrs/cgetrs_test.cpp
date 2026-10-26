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
 * \file cgetrs_test.cpp
 * \brief
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

int main(int argc, char **argv)
{
    int deviceId, N, NRHS;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    N = (argc > 2) ? std::atoi(argv[2]) : 32;
    NRHS = (argc > 3) ? std::atoi(argv[3]) : 32;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = N * N * sizeof(std::complex<float>);
    size_t ipivFileSize = N * sizeof(int32_t);
    size_t bMatrixFileSize = N * NRHS * sizeof(std::complex<float>);

    std::complex<float> *A = nullptr;
    std::complex<float> *B = nullptr;
    int32_t *ipiv = nullptr;

    auto cleanup = [&]() -> aclError
    {
        if (A != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(A));
        }
        if (B != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(B));
        }
        if (ipiv != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(ipiv));
        }
        CHECK_ACL(aclrtDestroyStream(stream));
        if (handle != nullptr)
        {
            CHECK_ACL(aclsolverDestroy(handle));
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return ACL_SUCCESS;
    };

    CHECK_ACL(aclrtMallocHost((void **)(&A), aMatrixFileSize));
    CHECK_RET(ReadFile("./test/cgetrs/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    CHECK_ACL(aclrtMallocHost((void **)(&B), bMatrixFileSize));
    CHECK_RET(ReadFile("./test/cgetrs/data/input/B_gm.bin", bMatrixFileSize, B, bMatrixFileSize),
              LOG_PRINT("ReadFile B_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    CHECK_ACL(aclrtMallocHost((void **)(&ipiv), ipivFileSize));
    CHECK_RET(ReadFile("./test/cgetrs/data/input/W_gm.bin", ipivFileSize, ipiv, ipivFileSize),
              LOG_PRINT("ReadFile W_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)A, N, N * 2, 8, 8);

    std::cout << "[Input] ipiv:" << std::endl;
    PrintPartOfMatrix<int32_t>((uint8_t *)ipiv, 1, N, 8, 8);

    std::cout << "[Input] B:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)B, N, NRHS * 2, 8, 8);

    int32_t infoVal = 0;
    int32_t *info = &infoVal;

    auto ret = aclsolverCgetrs(handle, 0, N, NRHS, A, N, ipiv, B, NRHS, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCgetrs failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] X:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t *)B, N, NRHS * 2, 8, 8);

    WriteFile("./test/cgetrs/data/output/X_gm.bin", B, bMatrixFileSize);

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}