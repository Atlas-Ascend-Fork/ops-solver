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
 * \file cmatinv_batched_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/assert.h"
#include "../utils/gm_addr.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "tiling/platform/platform_ascendc.h"

extern void cmatinv_batched_kernel_do(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag, GM_ADDR dOffset, GM_ADDR dAinv,
                                      GM_ADDR workSpace, GM_ADDR tilingGm, uint32_t numBlocks, void *stream);

static constexpr uint32_t COMPLEX_ELENUM = 2;
static constexpr uint32_t MAX_CORE_CNT = 40;
static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;

static constexpr int64_t DTYPE_COMPLEX32 = 0;
static constexpr int64_t DTYPE_COMPLEX64 = 1;

static constexpr int ELEMENT_NUM_PER_BLOCK = 8;
static constexpr int ELEMENT_NUM_PER_REPEAT = 64;
static constexpr uint32_t BASE_COMPLEX64_NUM = 8192;  // 64kb / 8b
static constexpr uint32_t MAX_COMPUTE_NUM = 128;      // max compute num per repeat

struct CmatinvBatchedTilingData
{
    uint32_t dtype;
    uint32_t n;
    uint32_t startOffset[40];
    uint32_t calNum[40];
};

// 获取归一化后的 AIV 核心数：平台信息查询失败返回 0 时按 1 处理（避免以 0 block 启动内核），
// 并按 maxCnt 截断，与 cheevj 的 GetCheevjBlockCount 防护口径一致
uint32_t GetNormalizedAivBlockCount(uint32_t maxCnt)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr)
    {
        numBlocks = ascendcPlatform->GetCoreNumAiv();
    }
    if (numBlocks == 0)
    {
        numBlocks = 1;
    }
    return numBlocks > maxCnt ? maxCnt : numBlocks;
}

CmatinvBatchedTilingData CalTilingData(uint32_t vecCoreNum, uint32_t dtype, uint32_t n, uint32_t batchSize)
{
    CmatinvBatchedTilingData tilingData;

    if (vecCoreNum == 0)
    {
        vecCoreNum = 1;
    }

    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;

    // init tiling data
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++)
    {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    uint32_t matPerCore = batchSize / vecCoreNum;
    uint32_t remainMatNum = batchSize % vecCoreNum;
    if (matPerCore == 0)
    {
        for (uint32_t i = 0; i < remainMatNum; i++)
        {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i * n * n * COMPLEX_ELENUM;  // complex element num
        }
    }
    else
    {
        uint32_t currComputeNum;
        uint32_t currOffset = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++)
        {
            if (i < remainMatNum)
            {
                currComputeNum = matPerCore + 1;
            }
            else
            {
                currComputeNum = matPerCore;
            }
            tilingData.calNum[i] = currComputeNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currComputeNum * n * n * COMPLEX_ELENUM;  // complex element num
        }
    }
    tilingData.dtype = dtype;
    tilingData.n = n;

    return tilingData;
}

aclError aclsolverCmatinvBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                                 std::complex<float> *Ainv, const int64_t lda_inv, int32_t *info, int64_t batchSize)
{
    // info 遵循 LAPACK 语义（issue #164）：成功为 0，参数错误由下方 ECHECK 以负值
    // 语义返回（aclError）；奇异信息见各 kernel 主元哨兵约定
    if (info != nullptr)
    {
        *info = 0;
    }
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }

    uint32_t numBlocks = GetNormalizedAivBlockCount(MAX_CORE_CNT);

    // info 允许为空（issue #244，对齐 getrf 族 #180 的 nullptr 容错口径），
    // 与上方 info != nullptr 判空清零分支统一，消除自相矛盾
    SOLVER_ECHECK(n > 0 && batchSize > 0 && lda > 0 && lda_inv > 0 && A != nullptr && Ainv != nullptr,
                  "CmatinvBatched get invalid param: n, batchSize, lda, lda_inv <= 0, or A, Ainv is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "CmatinvBatched get n > 256 or batchSize > 3000, which exceeds the supported limit.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(lda == n && lda_inv == n,
                  "CmatinvBatched only supports lda == n and lda_inv == n in current version.",
                  ACL_ERROR_INVALID_PARAM);

    // 参数校验完成后, n >= 32 的场景按 CgetriBatched 语义执行(适用其约束)
    if (n >= MATRIX_SHAPE_LIMIT)
    {
        LOG_PRINT("CmatinvBatched forwards to CgetriBatched for n >= 32.\n");
        return aclsolverCgetriBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
    }

    uint32_t N = static_cast<uint32_t>(n);
    uint32_t alignedN = (N + (ELEMENT_NUM_PER_BLOCK - 1)) / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    uint32_t computeNumPerRepeat = BASE_COMPLEX64_NUM / (N * alignedN);
    computeNumPerRepeat = std::min(computeNumPerRepeat, MAX_COMPUTE_NUM);
    computeNumPerRepeat = computeNumPerRepeat / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;
    std::vector<float> uniBatchRealData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatRealData(N * alignedN, 0.0f);
    std::vector<float> uniBatchImagData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniMatImagData(N * alignedN, 0.0f);
    for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++)
    {
        for (uint32_t colIdx = 0; colIdx < alignedN; colIdx++)
        {
            if (rowIdx == colIdx)
            {
                uniMatRealData[alignedN * rowIdx + colIdx] = 1.0f;
            }
            else
            {
                uniMatRealData[alignedN * rowIdx + colIdx] = 0.0f;
            }
            uniMatImagData[alignedN * rowIdx + colIdx] = 0.0f;
        }
    }
    errno_t rc;
    for (uint32_t i = 0; i < computeNumPerRepeat; i++)
    {
        rc = memcpy_s(static_cast<float *>(uniBatchRealData.data() + N * alignedN * i), N * alignedN * sizeof(float),
                      uniMatRealData.data(), N * alignedN * sizeof(float));
        SOLVER_ECHECK(rc == EOK, "Repeat uniMatRealData by memcpy_s failed.", ACL_ERROR_INTERNAL_ERROR);
        rc = memcpy_s(static_cast<float *>(uniBatchImagData.data() + N * alignedN * i), N * alignedN * sizeof(float),
                      uniMatImagData.data(), N * alignedN * sizeof(float));
        SOLVER_ECHECK(rc == EOK, "Repeat uniMatImagData by memcpy_s failed.", ACL_ERROR_INTERNAL_ERROR);
    }

    uint32_t offsetSize = N * alignedN * computeNumPerRepeat * 2;
    offsetSize = (offsetSize + (ELEMENT_NUM_PER_REPEAT - 1)) / ELEMENT_NUM_PER_REPEAT * ELEMENT_NUM_PER_REPEAT;
    std::vector<uint32_t> offsetData(offsetSize, 0);
    uint32_t k = 0;
    uint32_t realBase = 0;
    uint32_t imagBase = 32 * 1024;
    for (uint32_t batchIdx = 0; batchIdx < computeNumPerRepeat; batchIdx++)
    {
        for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++)
        {
            for (uint32_t colIdx = 0; colIdx < N; colIdx++)
            {
                offsetData[k++] = realBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
                offsetData[k++] = imagBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
            }
        }
    }

    uint8_t *hostA = reinterpret_cast<uint8_t *>(A);
    uint8_t *hostAinv = reinterpret_cast<uint8_t *>(Ainv);
    uint8_t *hostUniReal = reinterpret_cast<uint8_t *>(uniBatchRealData.data());
    uint8_t *hostUniImag = reinterpret_cast<uint8_t *>(uniBatchImagData.data());
    uint8_t *hostOffset = reinterpret_cast<uint8_t *>(offsetData.data());
    auto sizeA = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeAinv = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeUniReal = N * alignedN * computeNumPerRepeat * sizeof(float);
    auto sizeUniImag = N * alignedN * computeNumPerRepeat * sizeof(float);
    auto sizeOffset = offsetSize * sizeof(uint32_t);

    uint8_t *d_A = nullptr;
    uint8_t *d_Ainv = nullptr;
    uint8_t *d_UniReal = nullptr;
    uint8_t *d_UniImag = nullptr;
    uint8_t *d_Offset = nullptr;
    uint8_t *tilingDevice = nullptr;
    uint8_t *workSpace = nullptr;

    auto cleanup = [&]()
    {
        if (d_A) aclrtFree(d_A);
        if (d_UniReal) aclrtFree(d_UniReal);
        if (d_UniImag) aclrtFree(d_UniImag);
        if (d_Offset) aclrtFree(d_Offset);
        if (d_Ainv) aclrtFree(d_Ainv);
        if (workSpace) aclrtFree(workSpace);
        if (tilingDevice) aclrtFree(tilingDevice);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&d_A, sizeA, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_Ainv, sizeAinv, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_UniReal, sizeUniReal, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_UniImag, sizeUniImag, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_Offset, sizeOffset, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_A, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_Ainv, sizeAinv, hostAinv, sizeAinv, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_UniReal, sizeUniReal, hostUniReal, sizeUniReal, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_UniImag, sizeUniImag, hostUniImag, sizeUniImag, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_Offset, sizeOffset, hostOffset, sizeOffset, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());

    CmatinvBatchedTilingData tiling = CalTilingData(numBlocks, DTYPE_COMPLEX64, n, batchSize);
    CHECK_ACLRT(aclrtMalloc((void **)&tilingDevice, sizeof(CmatinvBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST),
                cleanup());
    CHECK_ACLRT(aclrtMemcpy(tilingDevice, sizeof(CmatinvBatchedTilingData), &tiling, sizeof(CmatinvBatchedTilingData),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&workSpace, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    cmatinv_batched_kernel_do(d_A, d_UniReal, d_UniImag, d_Offset, d_Ainv, workSpace, tilingDevice, numBlocks, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(hostAinv, sizeAinv, d_Ainv, sizeAinv, ACL_MEMCPY_DEVICE_TO_HOST), cleanup());

    cleanup();

    return ACL_SUCCESS;
}
