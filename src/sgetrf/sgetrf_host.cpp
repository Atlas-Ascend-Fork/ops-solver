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
 * \file sgetrf_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <cstdint>

#include "../utils/assert.h"
#include "../utils/gm_addr.h"
#include "../utils/lu_host_common.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

extern void sgetrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockN, int tileM, GM_ADDR A_org, GM_ADDR A_work,
                             GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm, uint32_t numBlocks,
                             void *stream);

aclError aclsolverSgetrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, float *A, const int64_t lda,
                         int32_t *ipiv, int32_t *info)
{
    // info 遵循 LAPACK 语义（issue #164）：成功为 0，参数错误由下方 ECHECK 以负值
    // 语义返回（aclError）；奇异信息见各 kernel 主元哨兵约定
    if (info != nullptr)
    {
        *info = 0;
    }
    // info 允许为空（issue #180）：对齐 LAPACK 惯例，调用方不关心奇异信息时
    // 可传 nullptr；此处的判空初始化与奇异列上报处的判空防护因此都是可达分支
    SOLVER_ECHECK(m > 0 && n > 0 && lda > 0 && A != nullptr && ipiv != nullptr,
                  "aclsolverSgetrf invalid param: m, n, lda <= 0, or A, ipiv is nullptr.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m <= INT32_MAX && n <= INT32_MAX, "aclsolverSgetrf invalid param: m or n exceeds int32 range.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m * n <= INT32_MAX, "aclsolverSgetrf invalid param: m * n exceeds INT32_MAX elements.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m <= LU_MAX_ROW, "aclsolverSgetrf invalid param: m exceeds the 8192-row kernel limit.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(
        lda == n,
        "aclsolverSgetrf only supports lda == n in current version, matrix A must be stored contiguously as m * n.",
        ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }
    // kernel hardcoded gemm1BlockNum=8, gemm2BlockNum=20-8, must use 20 blocks
    const uint32_t numBlocks = LU_MAX_NUM_BLOCKS;

    const int64_t blockN = LU_BLOCK_N;
    const int64_t tileM = LU_TILE_M;
    // 尺寸推导统一在 int64_t 域计算，避免 m/n 较大时 int 乘加溢出导致缓冲尺寸回绕
    int64_t t = (std::min(m, n) + blockN - 1) / blockN * blockN;
    int64_t alignedM = LuAlignUp(m, LU_ROW_ALIGNED);
    int64_t strideN = LuAlignUp(n, LU_COL_ALIGNED);

    size_t aMatrixFileSize = static_cast<size_t>(m) * n * sizeof(float);
    size_t wFileSize = t * sizeof(int);

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;
    uint8_t *wHost = nullptr;
    uint8_t *wDevice = nullptr;
    uint8_t *workDevice = nullptr;
    uint8_t *gatherHost1 = nullptr, *gatherHost2 = nullptr;
    uint8_t *gatherDevice1 = nullptr, *gatherDevice2 = nullptr;
    uint8_t *sync = nullptr;

    auto cleanup = [&]()
    {
        if (aMatrixDevice) aclrtFree(aMatrixDevice);
        if (aMatrixDeviceWork) aclrtFree(aMatrixDeviceWork);
        if (wDevice) aclrtFree(wDevice);
        if (wHost) aclrtFreeHost(wHost);
        if (workDevice) aclrtFree(workDevice);
        if (gatherDevice1) aclrtFree(gatherDevice1);
        if (gatherDevice2) aclrtFree(gatherDevice2);
        if (gatherHost1) aclrtFreeHost(gatherHost1);
        if (gatherHost2) aclrtFreeHost(gatherHost2);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    size_t aMatrixWorkSize = static_cast<size_t>(strideN) * static_cast<size_t>(alignedM) * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void **)(&wHost), wFileSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemset(wDevice, wFileSize, -1, wFileSize), cleanup());

    int64_t workM = LuAlignUp(m, LU_TILE_M);
    size_t workSize = static_cast<size_t>(workM) * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&workDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    size_t gatherSize = tileM * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost1), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost2), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice1, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice2, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    // gather1/gather2 重排表收敛为 utils 共享实现（issue #144）
    if (!GenerateGatherTables(tileM, blockN, n, strideN, alignedM, gatherHost1, gatherHost2, nullptr, nullptr,
                              EYE_FLOATS_PER_REAL_ELEMENT))
    {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice1, gatherSize, gatherHost1, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice2, gatherSize, gatherHost2, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());
    sgetrf_kernel_do(sync, m, n, blockN, tileM, aMatrixDevice, aMatrixDeviceWork, wDevice, workDevice, gatherDevice1,
                     gatherDevice2, numBlocks, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(aMatrixHost, aMatrixFileSize, aMatrixDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                cleanup());

    // Copy pivot information back to ipiv
    int32_t *wHostInt = reinterpret_cast<int32_t *>(wHost);
    CHECK_ACLRT(aclrtMemcpy(wHost, wFileSize, wDevice, wFileSize, ACL_MEMCPY_DEVICE_TO_HOST), cleanup());
    for (int64_t i = 0; i < std::min(m, n); ++i)
    {
        // W 主元哨兵初始化为 -1（0xFFFFFFFF），kernel 只回写真实主元行号；
        // 残留哨兵说明该列主元为零（矩阵奇异），按 LAPACK 语义上报首个
        // 奇异列号（1-based）。与复数孪生算子 cgetrf_host.cpp 口径对齐，
        // 修复实矩阵奇异时静默返回成功且 info==0 的不对称行为（issue #205）
        if (info != nullptr && *info == 0 && wHostInt[i] == -1)
        {
            *info = static_cast<int32_t>(i + 1);
        }
        ipiv[i] = wHostInt[i] + 1;  // Convert to 1-based indexing
    }

    cleanup();

    return ACL_SUCCESS;
}
