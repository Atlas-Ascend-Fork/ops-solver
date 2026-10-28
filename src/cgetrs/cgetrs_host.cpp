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
 * \file cgetrs_host.cpp
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
#include "cgetrs_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_N 16                // base case N大小
#define BLOCK_M 16                // base case M大小
#define CACHE_LINE_SIZE_FP32 128  // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096  // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率
#define TILE_LENGTH 4096          // 实虚结合时tile大小

#define GM_ADDR uint8_t *

void GenerateTiling(uint8_t *gatherBuf)
{
    try
    {
        {
            auto buf = reinterpret_cast<uint32_t *>(gatherBuf);
            int idx = 0;
            for (int i = 0; i < TILE_LENGTH; ++i)
            {
                buf[idx++] = i * sizeof(float);
                buf[idx++] = (i + TILE_LENGTH) * sizeof(float);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        exit(1);
    }
}

aclError aclsolverCgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs,
                         std::complex<float> *A, const int64_t lda, int32_t *ipiv, std::complex<float> *B,
                         const int64_t ldb, int32_t *info)
{
    // n > 8192 会使 SplitRealImag 的 alignedN > TILE_LENGTH，CopyIn 第二段超过 UB buffer（issue #266 检视），
    // 与 aclsolverSgetrf 的文档口径（valid range [1, 8192]）保持一致。
    // kernel 以 uint16_t 字节计数承载 DataCopyExtParams 行间隙，复数实/虚交织布局每元素占 8 字节：
    // SplitRealImag::CopyIn 行间隙为 (lda - n) * 8 / (ldb - nrhs) * 8 字节，须 <= 65535，即元素差 <= 8191；
    // trans == 0 时 restore 的 MergeRealImag::CopyOut 在 nrhs > 4096 的第二段行间隙为
    // (ldb - nrhs + 4096) * 8 字节，故 nrhs > 4096 时需进一步限制 ldb - nrhs <= 4095（issue #266 检视）。
    SOLVER_ECHECK(n > 0 && n <= 8192 && nrhs > 0 && lda >= n && ldb >= nrhs && A != nullptr && B != nullptr &&
                      ipiv != nullptr && info != nullptr && lda - n <= 8191 && ldb - nrhs <= 8191 &&
                      (nrhs <= 4096 || ldb - nrhs <= 4095),
                  "aclsolverCgetrs invalid param: n out of [1, 8192], n, nrhs, lda, ldb <= 0, lda < n, "
                  "ldb < nrhs, lda - n > 8191, ldb - nrhs > 8191 (or > 4095 when nrhs > 4096), "
                  "or A, B, ipiv, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);

    SOLVER_ECHECK(trans <= 2, "aclsolverSgetrs invalid param: trans must be 0, 1 or 2.", ACL_ERROR_INVALID_PARAM);

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

    size_t aMatrixFileSize = N * lda * sizeof(float) * 2;
    size_t bMatrixFileSize = N * ldb * sizeof(float) * 2;
    size_t wFileSize = N * sizeof(int);

    size_t resizedAMatrixFileSize = resizedN * strideN * sizeof(float) * 2;
    size_t resizedBMatrixFileSize = resizedN * strideNRHS * sizeof(float) * 3;

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;

    uint8_t *wHost = reinterpret_cast<uint8_t *>(ipiv);
    uint8_t *wDevice = nullptr;
    uint8_t *pDevice = nullptr;

    uint8_t *bMatrixHost = reinterpret_cast<uint8_t *>(B);
    uint8_t *bMatrixDevice = nullptr;
    uint8_t *bMatrixDeviceWork = nullptr;

    uint8_t *gatherHost = nullptr;
    uint8_t *gatherDevice = nullptr;

    uint8_t *sync = nullptr;

    auto cleanup = [&]()
    {
        if (aMatrixDevice) aclrtFree(aMatrixDevice);
        if (aMatrixDeviceWork) aclrtFree(aMatrixDeviceWork);
        if (wDevice) aclrtFree(wDevice);
        if (pDevice) aclrtFree(pDevice);
        if (bMatrixDevice) aclrtFree(bMatrixDevice);
        if (bMatrixDeviceWork) aclrtFree(bMatrixDeviceWork);
        if (gatherDevice) aclrtFree(gatherDevice);
        if (gatherHost) aclrtFreeHost(gatherHost);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, resizedAMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&pDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(wDevice, wFileSize, wHost, wFileSize, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemset(pDevice, wFileSize, -1, wFileSize), cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&bMatrixDevice, bMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(bMatrixDevice, bMatrixFileSize, bMatrixHost, bMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&bMatrixDeviceWork, resizedBMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    size_t gatherSize = TILE_LENGTH * 2 * sizeof(uint32_t);
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    GenerateTiling(gatherHost);

    CHECK_ACLRT(aclrtMemcpy(gatherDevice, gatherSize, gatherHost, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    cgetrs_kernel_do(sync, N, NRHS, trans, lda, ldb, blockM, blockN, aMatrixDevice, aMatrixDeviceWork, wDevice, pDevice,
                     bMatrixDevice, bMatrixDeviceWork, gatherDevice, blockDim, stream);

    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    std::cout << "finish" << std::endl;

    CHECK_ACLRT(aclrtMemcpy(bMatrixHost, bMatrixFileSize, bMatrixDevice, bMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                cleanup());

    cleanup();

    *info = 0;

    return ACL_SUCCESS;
}
