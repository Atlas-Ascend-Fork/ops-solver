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

// 负向用例（issue #670）：info 容量不足（1 字节 < sizeof(int)）时必须在任何设备写
// 之前拒绝并返回 INVALID_VALUE；对齐实数 spotrfBatched 的 CheckInfoSpan 设备范围
// 校验口径，正常 Device info 的错误回写行为不变。
static aclError CheckUndersizedInfoSpan(aclsolverHandle_t handle)
{
    const int negN = 64;
    void* negA = nullptr;
    void* negWorkspace = nullptr;
    void* tinyInfo = nullptr;
    CHECK_ACL(aclrtMalloc(&negA, (size_t)negN * negN * sizeof(aclFloatComplex), ACL_MEM_MALLOC_HIGHEST_FIRST));
    CHECK_ACL(aclrtMalloc(&negWorkspace, (size_t)negN * negN * sizeof(aclFloatComplex), ACL_MEM_MALLOC_HIGHEST_FIRST));
    CHECK_ACL(aclrtMalloc(&tinyInfo, 1, ACL_MEM_MALLOC_HIGHEST_FIRST));
    auto negRet = aclsolverCpotrf(handle, ACLSOLVER_FILL_MODE_UPPER, negN, (aclFloatComplex*)negA, negN,
                                  (aclFloatComplex*)negWorkspace, negN * negN, (int32_t*)tinyInfo);
    CHECK_RET(negRet == ACLSOLVER_STATUS_INVALID_VALUE,
              LOG_PRINT("undersized info expected ACLSOLVER_STATUS_INVALID_VALUE, got %d\n", negRet);
              aclrtFree(negA); aclrtFree(negWorkspace); aclrtFree(tinyInfo); return ACL_ERROR_INVALID_PARAM);
    CHECK_ACL(aclrtFree(negA));
    CHECK_ACL(aclrtFree(negWorkspace));
    CHECK_ACL(aclrtFree(tinyInfo));
    return ACL_SUCCESS;
}

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

    auto negSpanRet = CheckUndersizedInfoSpan(handle);
    CHECK_RET(negSpanRet == ACL_SUCCESS, LOG_PRINT("undersized info span negative case failed: %d\n", negSpanRet);
              cleanup(); return negSpanRet);

    std::cout << "[Input] A:" << std::endl;
    PrintPartOfMatrix<float>((uint8_t*)A, N, 2 * N, 8, 8);

    int32_t infoVal = 0;
    int32_t* info = &infoVal;

    // Upper Cholesky
    auto ret = aclsolverCpotrf(handle, 'U', N, A, N, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCpotrf (U) failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] A (U):" << std::endl;
    PrintPartOfMatrix<float>((uint8_t*)A, N, 2 * N, 8, 8);

    CHECK_RET(WriteFile("./test/cpotrf/data/output/U_gm.bin", A, aMatrixFileSize),
              LOG_PRINT("WriteFile U_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    // Re-read original matrix for lower factorization
    CHECK_RET(ReadFile("./test/cpotrf/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    ret = aclsolverCpotrf(handle, 'L', N, A, N, info);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCpotrf (L) failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] A (L):" << std::endl;
    PrintPartOfMatrix<float>((uint8_t*)A, N, 2 * N, 8, 8);

    CHECK_RET(WriteFile("./test/cpotrf/data/output/L_gm.bin", A, aMatrixFileSize),
              LOG_PRINT("WriteFile L_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
