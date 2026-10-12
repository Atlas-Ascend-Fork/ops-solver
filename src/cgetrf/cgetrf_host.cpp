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
 * \file cgetrf_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <complex>
#include <cstdint>

#include "../utils/assert.h"
#include "../utils/gm_addr.h"
#include "../utils/lu_host_common.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

extern void cgetrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                             GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                             GM_ADDR gather3_gm, uint32_t numBlocks, void *stream);

aclError aclsolverCgetrf(aclsolverHandle_t handle, const int64_t m, const int64_t n, std::complex<float> *A,
                         const int64_t lda, int32_t *ipiv, int32_t *info)
{
    // info 遵循 LAPACK 语义（issue #164）：成功为 0；参数错误为负；矩阵奇异
    // （W 主元哨兵未消元）为大于 0 的主元列号（1-based）
    if (info != nullptr)
    {
        *info = 0;
    }
    // info 允许为空（issue #180）：对齐 LAPACK 惯例，调用方不关心奇异信息时
    // 可传 nullptr；此处的判空初始化与奇异列上报处的判空防护因此都是可达分支
    SOLVER_ECHECK(m > 0 && n > 0 && lda > 0 && A != nullptr && ipiv != nullptr,
                  "aclsolverCgetrf invalid param: m, n, lda <= 0, or A, ipiv is nullptr.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m <= INT32_MAX && n <= INT32_MAX, "aclsolverCgetrf invalid param: m or n exceeds int32 range.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m * n <= INT32_MAX, "aclsolverCgetrf invalid param: m * n exceeds INT32_MAX elements.",
                  ACL_ERROR_INVALID_PARAM);
    // kernel 的虚实分离/合并按 n<=8192 设计（SplitRealImag/MergeRealImag 一次
    // 拷入缓冲为 8192 float，且入口注释设计上限 192MB），n 超限时 kernel 越界
    // 写本地缓冲（issue #165），host 侧必须显式拒绝。
    SOLVER_ECHECK(n <= LU_CGETRF_MAX_N, "aclsolverCgetrf invalid param: n exceeds the kernel design limit 8192.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(m <= LU_MAX_ROW, "aclsolverCgetrf invalid param: m exceeds the 8192-row kernel limit.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(
        lda == n,
        "aclsolverCgetrf only supports lda == n in current version, matrix A must be stored contiguously as m * n.",
        ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }
    // kernel hardcoded gemm1BlockNum=8, gemm2BlockNum=20-8, must use 20 blocks
    const uint32_t numBlocks = LU_MAX_NUM_BLOCKS;

    const int64_t blockM = LU_ROW_ALIGNED;
    const int64_t blockN = LU_BLOCK_N;
    const int64_t tileM = LU_TILE_M;
    // 尺寸推导统一在 int64_t 域计算，避免 m/n 较大时 int 乘加溢出导致缓冲尺寸回绕
    int64_t t = (std::min(m, n) + blockN - 1) / blockN * blockN;
    int64_t alignedM = LuAlignUp(m, LU_ROW_ALIGNED);
    int64_t strideN = LuAlignUp(n, LU_COL_ALIGNED);

    size_t aMatrixFileSize = static_cast<size_t>(m) * n * sizeof(float) * EYE_FLOATS_PER_COMPLEX_ELEMENT;
    size_t wFileSize = t * sizeof(int);

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;
    uint8_t *wHost = nullptr;
    uint8_t *wDevice = nullptr;
    uint8_t *workDevice = nullptr;
    uint8_t *gatherHost1 = nullptr, *gatherHost2 = nullptr, *gatherHost3 = nullptr;
    uint8_t *gatherDevice1 = nullptr, *gatherDevice2 = nullptr, *gatherDevice3 = nullptr;
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
        if (gatherDevice3) aclrtFree(gatherDevice3);
        if (gatherHost1) aclrtFreeHost(gatherHost1);
        if (gatherHost2) aclrtFreeHost(gatherHost2);
        if (gatherHost3) aclrtFreeHost(gatherHost3);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    size_t aMatrixWorkSize = static_cast<size_t>(strideN) * static_cast<size_t>(alignedM) * sizeof(float) * 3;
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void **)(&wHost), wFileSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemset(wDevice, wFileSize, -1, wFileSize), cleanup());

    int64_t workM = LuAlignUp(m, LU_TILE_M);
    size_t workSize = static_cast<size_t>(workM) * blockN * sizeof(float) * EYE_FLOATS_PER_COMPLEX_ELEMENT;
    CHECK_ACLRT(aclrtMalloc((void **)&workDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    static constexpr int TILE_LENGTH = 8192;
    size_t gatherSize = tileM * blockN * sizeof(float);
    size_t gatherSize3 = TILE_LENGTH * sizeof(uint32_t);
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost1), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost2), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost3), gatherSize3), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice1, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice2, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice3, gatherSize3, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    // gather 表收敛为 utils 共享实现；gather3 按 kernel 的 ceil8 对齐口径生成（issue #137/#144）
    if (!GenerateGatherTables(tileM, blockN, n, strideN, alignedM, gatherHost1, gatherHost2, gatherHost3, nullptr,
                              EYE_FLOATS_PER_REAL_ELEMENT))
    {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice1, gatherSize, gatherHost1, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice2, gatherSize, gatherHost2, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(
        aclrtMemcpyAsync(gatherDevice3, gatherSize3, gatherHost3, gatherSize3, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    cgetrf_kernel_do(sync, m, n, blockM, blockN, tileM, aMatrixDevice, aMatrixDeviceWork, wDevice, workDevice,
                     gatherDevice1, gatherDevice2, gatherDevice3, numBlocks, stream);
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
        // 奇异列号（1-based，issue #164）
        if (info != nullptr && *info == 0 && wHostInt[i] == -1)
        {
            *info = static_cast<int32_t>(i + 1);
        }
        ipiv[i] = wHostInt[i] + 1;  // Convert to 1-based indexing
    }

    cleanup();

    return ACL_SUCCESS;
}
