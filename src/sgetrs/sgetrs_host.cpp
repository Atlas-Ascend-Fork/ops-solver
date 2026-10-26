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
 * \file sgetrs_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/assert.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "sgetrs_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_N 16                 // base case N大小
#define BLOCK_M 16                 // base case M大小
#define CACHE_LINE_SIZE_FP32 128   // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096   // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率

#define GM_ADDR uint8_t *

aclError aclsolverSgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs,
                         float *A, const int64_t lda, int32_t *ipiv, float *B, const int64_t ldb, int32_t *info)
{
    SOLVER_ECHECK(n > 0 && nrhs > 0 && lda >= n && ldb >= nrhs && A != nullptr && B != nullptr &&
                      ipiv != nullptr && info != nullptr,
                  "aclsolverSgetrs invalid param: n, nrhs, lda, ldb <= 0 or lda < n or ldb < nrhs, "
                  "or A, B, ipiv, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);

    SOLVER_ECHECK(trans <= 1, "aclsolverSgetrs invalid param: trans must be 0 or 1.", ACL_ERROR_INVALID_PARAM);

    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr)
    {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    int N, NRHS, blockDim, blockN, blockM;

    N = n;
    NRHS = nrhs;
    blockDim = numBlocks;

    blockN = BLOCK_N;
    blockM = BLOCK_M;

    int resizedN = (N + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;

    int strideN = (N + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideN & -strideN) >= ADDITIONAL_PAD_SIZE) strideN += CACHE_LINE_SIZE_FP32;

    int strideNRHS = (NRHS + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((strideNRHS & -strideNRHS) >= ADDITIONAL_PAD_SIZE) strideNRHS += CACHE_LINE_SIZE_FP32;

    size_t aMatrixFileSize = N * lda * sizeof(float);
    size_t bMatrixFileSize = N * ldb * sizeof(float);
    size_t wFileSize = N * sizeof(int);

    size_t resizedAMatrixFileSize = resizedN * strideN * sizeof(float);
    size_t resizedBMatrixFileSize = resizedN * strideNRHS * sizeof(float);

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;

    uint8_t *wHost = reinterpret_cast<uint8_t *>(ipiv);
    uint8_t *wDevice = nullptr;
    uint8_t *pDevice = nullptr;

    uint8_t *bMatrixHost = reinterpret_cast<uint8_t *>(B);
    uint8_t *bMatrixDevice = nullptr;
    uint8_t *bMatrixDeviceWork = nullptr;

    uint8_t *sync = nullptr;

    auto cleanup = [&]()
    {
        if (aMatrixDevice) aclrtFree(aMatrixDevice);
        if (aMatrixDeviceWork) aclrtFree(aMatrixDeviceWork);
        if (wDevice) aclrtFree(wDevice);
        if (pDevice) aclrtFree(pDevice);
        if (bMatrixDevice) aclrtFree(bMatrixDevice);
        if (bMatrixDeviceWork) aclrtFree(bMatrixDeviceWork);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, resizedAMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST),
                cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&pDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(wDevice, wFileSize, wHost, wFileSize, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemset(pDevice, wFileSize, -1, wFileSize), cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&bMatrixDevice, bMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(bMatrixDevice, bMatrixFileSize, bMatrixHost, bMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&bMatrixDeviceWork, resizedBMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST),
                cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    sgetrs_kernel_do(sync, N, NRHS, trans, lda, ldb, blockM, blockN, aMatrixDevice, aMatrixDeviceWork, wDevice, pDevice,
                     bMatrixDevice, bMatrixDeviceWork, blockDim, stream);

    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    std::cout << "finish" << std::endl;

    CHECK_ACLRT(aclrtMemcpy(bMatrixHost, bMatrixFileSize, bMatrixDevice, bMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                cleanup());

    cleanup();

    *info = 0;

    return ACL_SUCCESS;
}
