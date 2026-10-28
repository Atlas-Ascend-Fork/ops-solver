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

#define BLOCK_N 16                // base case N大小
#define BLOCK_M 16                // base case M大小
#define CACHE_LINE_SIZE_FP32 128  // 128 floats per cache line
#define ADDITIONAL_PAD_SIZE 4096  // 当一行数据为4096的整数倍时，额外 pad 1个 cache line，提高访存效率

#define GM_ADDR uint8_t *

// sgetrs 全部 host/device 缓冲；未申请的指针为空，ReleaseSgetrsBuffers 可安全跳过
struct SgetrsBuffers
{
    uint8_t *aHost = nullptr;        // 用户输入矩阵 A（host 侧首地址）
    uint8_t *aDevice = nullptr;      // 上载后的 A
    uint8_t *aDeviceWork = nullptr;  // A 的 pad 工作区
    uint8_t *wHost = nullptr;        // 行交换序列 ipiv（host 侧首地址）
    uint8_t *wDevice = nullptr;      // 上载后的 ipiv
    uint8_t *pDevice = nullptr;      // laswp 输出的排列表
    uint8_t *bHost = nullptr;        // 用户输入/输出矩阵 B（host 侧首地址）
    uint8_t *bDevice = nullptr;      // 上载后的 B
    uint8_t *bDeviceWork = nullptr;  // B 的 pad 工作区
    uint8_t *sync = nullptr;         // 硬件同步地址
};

// 释放所有已申请的 HBM 内存（原 cleanup lambda 提取）
static void ReleaseSgetrsBuffers(SgetrsBuffers &buf)
{
    if (buf.aDevice) aclrtFree(buf.aDevice);
    if (buf.aDeviceWork) aclrtFree(buf.aDeviceWork);
    if (buf.wDevice) aclrtFree(buf.wDevice);
    if (buf.pDevice) aclrtFree(buf.pDevice);
    if (buf.bDevice) aclrtFree(buf.bDevice);
    if (buf.bDeviceWork) aclrtFree(buf.bDeviceWork);
}

// 行宽对齐到 cache line；2 的幂次对齐值达到阈值时额外 pad 一个 cache line，提高访存效率
static int CalcPadStride(int x)
{
    int stride = (x + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((stride & -stride) >= ADDITIONAL_PAD_SIZE) stride += CACHE_LINE_SIZE_FP32;
    return stride;
}

// A 上载并申请其 pad 工作区
static aclError AllocAndUploadA(SgetrsBuffers &buf, size_t aSize, size_t aWorkSize)
{
    CHECK_ACLRT(aclrtMalloc((void **)&buf.aDevice, aSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy(buf.aDevice, aSize, buf.aHost, aSize, ACL_MEMCPY_HOST_TO_DEVICE),
                ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.aDeviceWork, aWorkSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSgetrsBuffers(buf));
    return ACL_SUCCESS;
}

// 行交换序列上载，并把排列表初始化为 -1
static aclError AllocAndUploadPivots(SgetrsBuffers &buf, size_t wSize)
{
    CHECK_ACLRT(aclrtMalloc((void **)&buf.wDevice, wSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.pDevice, wSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy(buf.wDevice, wSize, buf.wHost, wSize, ACL_MEMCPY_HOST_TO_DEVICE),
                ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMemset(buf.pDevice, wSize, -1, wSize), ReleaseSgetrsBuffers(buf));
    return ACL_SUCCESS;
}

// B 上载并申请其 pad 工作区
static aclError AllocAndUploadB(SgetrsBuffers &buf, size_t bSize, size_t bWorkSize)
{
    CHECK_ACLRT(aclrtMalloc((void **)&buf.bDevice, bSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy(buf.bDevice, bSize, buf.bHost, bSize, ACL_MEMCPY_HOST_TO_DEVICE),
                ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.bDeviceWork, bWorkSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSgetrsBuffers(buf));
    return ACL_SUCCESS;
}

// 下发 kernel，等待完成后把结果按 ldb 行宽搬回 host
static aclError LaunchSgetrs(SgetrsBuffers &buf, int n, int nrhs, uint64_t trans, int64_t lda, int64_t ldb, int blockM,
                             int blockN, uint32_t blockNum, aclrtStream stream)
{
    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&buf.sync), ReleaseSgetrsBuffers(buf));
    sgetrs_kernel_do(buf.sync, n, nrhs, trans, lda, ldb, blockM, blockN, buf.aDevice, buf.aDeviceWork, buf.wDevice,
                     buf.pDevice, buf.bDevice, buf.bDeviceWork, blockNum, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ReleaseSgetrsBuffers(buf));
    size_t bSize = n * ldb * sizeof(float);
    CHECK_ACLRT(aclrtMemcpy(buf.bHost, bSize, buf.bDevice, bSize, ACL_MEMCPY_DEVICE_TO_HOST),
                ReleaseSgetrsBuffers(buf));
    return ACL_SUCCESS;
}

aclError aclsolverSgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs, float *A,
                         const int64_t lda, int32_t *ipiv, float *B, const int64_t ldb, int32_t *info)
{
    // n > 8192 会使 kernel pad.hpp 中 linesPerIter = TILE_LENGTH / alignedN 变为 0 并除零（issue #265），
    // nrhs > 8192 会使 pad/restore 按 nrhs 行宽搬运超出 TILE_LENGTH 的 UB buffer（issue #266 检视），
    // 与 aclsolverSgetrf 的文档口径（valid range [1, 8192]）保持一致。
    // kernel 以 uint16_t 字节计数承载 DataCopyExtParams 行间隙，(lda - n) * 4 / (ldb - nrhs) * 4
    // 不得超过 65535，即行间隙元素数 <= 16383（issue #266）。
    SOLVER_ECHECK(n > 0 && n <= 8192 && nrhs > 0 && nrhs <= 8192 && lda >= n && ldb >= nrhs && A != nullptr &&
                      B != nullptr && ipiv != nullptr && info != nullptr && lda - n <= 16383 && ldb - nrhs <= 16383,
                  "aclsolverSgetrs invalid param: n, nrhs out of [1, 8192], n, nrhs, lda, ldb <= 0, lda < n, "
                  "ldb < nrhs, lda - n > 16383, ldb - nrhs > 16383, or A, B, ipiv, info is nullptr.",
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

    int N = static_cast<int>(n);
    int NRHS = static_cast<int>(nrhs);
    int blockN = BLOCK_N;
    int blockM = BLOCK_M;

    // A/B 侧 pad 后行宽（A 侧不含额外 pad，B 侧按 stride 规则可能追加一个 cache line）
    int resizedN = (N + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = CalcPadStride(N);
    int strideNRHS = CalcPadStride(NRHS);

    SgetrsBuffers buf;
    buf.aHost = reinterpret_cast<uint8_t *>(A);
    buf.wHost = reinterpret_cast<uint8_t *>(ipiv);
    buf.bHost = reinterpret_cast<uint8_t *>(B);
    CHECK_ACLRT(AllocAndUploadA(buf, N * lda * sizeof(float), resizedN * strideN * sizeof(float)),
                ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(AllocAndUploadPivots(buf, N * sizeof(int)), ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(AllocAndUploadB(buf, N * ldb * sizeof(float), resizedN * strideNRHS * sizeof(float)),
                ReleaseSgetrsBuffers(buf));
    CHECK_ACLRT(LaunchSgetrs(buf, N, NRHS, trans, lda, ldb, blockM, blockN, numBlocks, stream),
                ReleaseSgetrsBuffers(buf));

    ReleaseSgetrsBuffers(buf);

    *info = 0;

    return ACL_SUCCESS;
}
