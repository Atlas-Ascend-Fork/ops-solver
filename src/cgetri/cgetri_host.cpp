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
 * \file cgetri_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <complex>
#include <cstdint>

#include "../utils/assert.h"
#include "../utils/eye_matrix.h"
#include "../utils/gm_addr.h"
#include "../utils/lu_host_common.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

extern void cgetri_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int tileM, GM_ADDR A_org,
                             GM_ADDR A_work, GM_ADDR W, GM_ADDR work_gm, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                             GM_ADDR gather3_gm, GM_ADDR eye_gm, uint32_t numBlocks, void *stream);
// gather 表与 eye 矩阵生成收敛为 utils 共享实现（issue #144）；
// gather3 按 kernel 的 ceil8 对齐口径生成（issue #137）；
// eye 的初始化范围扩展到 paddedM 行，消除 +128 行 padding 区未初始化上设备（issue #136）

aclError aclsolverCgetri(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                         int32_t *info)
{
    // info 遵循 LAPACK 语义（issue #164）：成功为 0，参数错误由下方 ECHECK 以负值
    // 语义返回（aclError）；奇异信息见各 kernel 主元哨兵约定
    if (info != nullptr)
    {
        *info = 0;
    }
    // info 允许为空（issue #244，对齐 getrf 族 #180 的 nullptr 容错口径），
    // 与上方 info != nullptr 判空清零分支统一，消除自相矛盾
    SOLVER_ECHECK(n > 0 && lda > 0 && A != nullptr, "aclsolverCgetri invalid param: n, lda <= 0, or A is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= INT32_MAX, "aclsolverCgetri invalid param: n exceeds int32 range.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n * n <= INT32_MAX, "aclsolverCgetri invalid param: n * n exceeds INT32_MAX elements.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= LU_MAX_GETRI_N, "aclsolverCgetri invalid param: n exceeds the kernel design limit 8192.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(
        lda == n,
        "aclsolverCgetri only supports lda == n in current version, matrix A must be stored contiguously as n * n.",
        ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }

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

    size_t aMatrixFileSize = static_cast<size_t>(n) * n * sizeof(float) * EYE_FLOATS_PER_COMPLEX_ELEMENT;
    size_t wFileSize = t * sizeof(int);

    uint8_t *aMatrixHost = reinterpret_cast<uint8_t *>(A);
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;
    uint8_t *wDevice = nullptr;
    uint8_t *workDevice = nullptr;
    uint8_t *eyeMatrixHost = nullptr;
    uint8_t *eyeMatrixDevice = nullptr;
    uint8_t *gatherHost1 = nullptr, *gatherHost2 = nullptr, *gatherHost3 = nullptr;
    uint8_t *gatherDevice1 = nullptr, *gatherDevice2 = nullptr, *gatherDevice3 = nullptr;
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
        if (gatherDevice3) aclrtFree(gatherDevice3);
        if (gatherHost1) aclrtFreeHost(gatherHost1);
        if (gatherHost2) aclrtFreeHost(gatherHost2);
        if (gatherHost3) aclrtFreeHost(gatherHost3);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());

    // A_work 布局为 strideN 列 × (alignedM + 128 行 padding) 的三个平面（实部/虚部/输出）
    const int64_t paddedM = alignedM + EYE_ROW_PADDING;
    size_t aMatrixWorkSize = static_cast<size_t>(strideN) * paddedM * sizeof(float) * 3;
    CHECK_ACLRT(aclrtMalloc((void **)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMalloc((void **)&wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemset(wDevice, wFileSize, -1, wFileSize), cleanup());

    int64_t workM = LuAlignUp(n, LU_TILE_M);
    size_t workSize = static_cast<size_t>(workM) * blockN * sizeof(float) * EYE_FLOATS_PER_COMPLEX_ELEMENT;
    CHECK_ACLRT(aclrtMalloc((void **)&workDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void **)(&eyeMatrixHost), aMatrixWorkSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&eyeMatrixDevice, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    size_t gatherSize = tileM * blockN * sizeof(float);
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost1), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost2), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void **)(&gatherHost3), gatherSize), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice1, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice2, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&gatherDevice3, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    // eyeMatrixHost 为三平面布局（实部/虚部/输出，aMatrixWorkSize = strideN*paddedM*3）：
    // GenerateEyeMatrix 仅初始化第 1 平面的单位阵，后两个平面在此整体清零，
    // 消除未初始化宿主内存随整体 H2D 上设备（#136 检视补充：三平面均需确定内容）
    if (memset_s(eyeMatrixHost, aMatrixWorkSize, 0, aMatrixWorkSize) != EOK)
    {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }
    if (!GenerateGatherTables(tileM, blockN, n, strideN, paddedM, gatherHost1, gatherHost2, gatherHost3, eyeMatrixHost,
                              EYE_FLOATS_PER_COMPLEX_ELEMENT))
    {
        cleanup();
        return ACL_ERROR_INTERNAL_ERROR;
    }
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice1, gatherSize, gatherHost1, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice2, gatherSize, gatherHost2, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(gatherDevice3, gatherSize, gatherHost3, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                cleanup());
    CHECK_ACLRT(
        aclrtMemcpy(eyeMatrixDevice, aMatrixWorkSize, eyeMatrixHost, aMatrixWorkSize, ACL_MEMCPY_HOST_TO_DEVICE),
        cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    cgetri_kernel_do(sync, n, n, blockM, blockN, tileM, aMatrixDevice, aMatrixDeviceWork, wDevice, workDevice,
                     gatherDevice1, gatherDevice2, gatherDevice3, eyeMatrixDevice, numBlocks, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(aMatrixHost, aMatrixFileSize, aMatrixDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                cleanup());

    cleanup();

    return ACL_SUCCESS;
}
