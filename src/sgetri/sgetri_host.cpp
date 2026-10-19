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
 * \file sgetri_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <cstdint>

#include "../utils/assert.h"
#include "../utils/eye_matrix.h"
#include "../utils/gm_addr.h"
#include "../utils/lu_host_common.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

extern void sgetri_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                             GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                             GM_ADDR eye_gm, uint32_t numBlocks, void *stream);
// gather 表与 eye 矩阵生成收敛为 utils 共享实现（issue #144）；
// eye 的初始化范围扩展到 paddedM 行，消除 +128 行 padding 区未初始化上设备（issue #136）

aclError aclsolverSgetri(aclsolverHandle_t handle, const int64_t n, float *A, const int64_t lda, int32_t *info)
{
    // info 遵循 LAPACK 语义（issue #164）：成功为 0，参数错误由返回值上报；
    // info 允许为空（issue #244，对齐 getrf 族 #180 的 nullptr 容错口径），
    // 原先"前置判空容错 + ECHECK 必传"的自相矛盾分支已统一为本口径
    if (info != nullptr)
    {
        *info = 0;
    }
    SOLVER_ECHECK(n > 0 && lda > 0 && A != nullptr, "aclsolverSgetri invalid param: n, lda <= 0, or A is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= INT32_MAX, "aclsolverSgetri invalid param: n exceeds int32 range.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n * n <= INT32_MAX, "aclsolverSgetri invalid param: n * n exceeds INT32_MAX elements.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= LU_MAX_GETRI_N, "aclsolverSgetri invalid param: n exceeds the kernel design limit 8192.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(
        lda == n,
        "aclsolverSgetri only supports lda == n in current version, matrix A must be stored contiguously as n * n.",
        ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }

    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;
    uint8_t *wDevice = nullptr;
    uint8_t *workDevice = nullptr;
    uint8_t *eyeMatrixHost = nullptr;
    uint8_t *eyeMatrixDevice = nullptr;
    uint8_t *gatherHost1 = nullptr, *gatherHost2 = nullptr;
    uint8_t *gatherDevice1 = nullptr, *gatherDevice2 = nullptr;
    uint8_t *sync = nullptr;

    auto cleanup = [&]()
    {
        if (aMatrixDevice) aclrtFree(aMatrixDevice);
        if (aMatrixDeviceWork) aclrtFree(aMatrixDeviceWork);
        if (eyeMatrixDevice) aclrtFree(eyeMatrixDevice);
        if (eyeMatrixHost) aclrtFreeHost(eyeMatrixHost);
        if (wDevice) aclrtFree(wDevice);
        if (workDevice) aclrtFree(workDevice);
        if (gatherDevice1) aclrtFree(gatherDevice1);
        if (gatherDevice2) aclrtFree(gatherDevice2);
        if (gatherHost1) aclrtFreeHost(gatherHost1);
        if (gatherHost2) aclrtFreeHost(gatherHost2);
    };

    // 与 sgetrf/cgetrf 对齐：kernel 侧 LU 实现按固定 20 块设计（blockIdx 8/10 负责
    // A/eye 主元行交换、coreIdx>=4 承担 trsm/gemm），块数不足时这些角色无人执行而
    // 求逆结果静默错误，故必须固定使用 LU_MAX_NUM_BLOCKS（issue #147）
    const uint32_t numBlocks = LU_MAX_NUM_BLOCKS;

    const int64_t blockM = LU_ROW_ALIGNED;
    const int64_t blockN = LU_BLOCK_N;
    const int64_t tileM = LU_TILE_M;
    // 尺寸推导统一在 int64_t 域计算，避免 n 较大时 int 乘加溢出导致缓冲尺寸回绕
    int64_t t = (n + blockN - 1) / blockN * blockN;
    int64_t alignedM = LuAlignUp(n, LU_ROW_ALIGNED);
    int64_t strideN = LuAlignUp(n, LU_COL_ALIGNED);
    // eye/工作区按 (alignedM + 128 行 padding) 布局，padding 区需清零后上设备（issue #136）
    const int64_t paddedM = alignedM + EYE_ROW_PADDING;

    size_t aMatrixFileSize = static_cast<size_t>(n) * n * sizeof(float);
    size_t wFileSize = t * sizeof(int);

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    size_t aMatrixWorkSize = static_cast<size_t>(strideN) * paddedM * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemset(wDevice, wFileSize, -1, wFileSize), cleanup());

    int64_t workM = LuAlignUp(n, LU_TILE_M);
    size_t workSize = static_cast<size_t>(workM) * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&workDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void **)(&eyeMatrixHost), aMatrixWorkSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&eyeMatrixDevice, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    size_t gatherSize = tileM * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost1), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost2), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice1, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice2, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    if (!GenerateGatherTables(tileM, blockN, n, strideN, paddedM, gatherHost1, gatherHost2, nullptr, eyeMatrixHost,
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
        aclrtMemcpy(eyeMatrixDevice, aMatrixWorkSize, eyeMatrixHost, aMatrixWorkSize, ACL_MEMCPY_HOST_TO_DEVICE),
        cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    sgetri_kernel_do(sync, n, n, blockM, blockN, tileM, aMatrixDevice, aMatrixDeviceWork, wDevice, workDevice,
                     gatherDevice1, gatherDevice2, eyeMatrixDevice, numBlocks, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(aMatrixHost, aMatrixFileSize, aMatrixDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                cleanup());

    cleanup();

    return ACL_SUCCESS;
}
