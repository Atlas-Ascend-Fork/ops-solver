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

#include "../utils/aicore_num.h"
#include "../utils/assert.h"
#include "../utils/getrs_defs.h"
#include "../utils/pivot_check.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "cgetrs_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"

#define BLOCK_M 16           // base case M大小
#define GATHER_SEG_LEN 4096  // gather 段宽（区别于 kernel 侧 laswp.hpp 的 TILE_LENGTH=8192，issue #328）

// 公共 ABI 契约（issue #354）：aclsolverCgetrs 形参已改为 C 兼容的 aclFloatComplex，
// 与 std::complex<float> 布局逐字节等价（两个连续 IEEE float32），实现侧按该断言安全直传。
static_assert(sizeof(aclFloatComplex) == sizeof(std::complex<float>) &&
                  alignof(aclFloatComplex) == alignof(std::complex<float>),
              "aclFloatComplex must be layout-compatible with std::complex<float>");

#define GM_ADDR uint8_t *

void GenerateTiling(uint8_t *gatherBuf)
{
    auto buf = reinterpret_cast<uint32_t *>(gatherBuf);
    int idx = 0;
    for (int i = 0; i < GATHER_SEG_LEN; ++i)
    {
        buf[idx++] = i * sizeof(float);
        buf[idx++] = (i + GATHER_SEG_LEN) * sizeof(float);
    }
}

namespace
{
// sgetrs 全部 host/device 缓冲；未申请的指针为空，ReleaseCgetrsBuffers 可安全调用
struct CgetrsBuffers
{
    uint8_t *aMatrixHost = nullptr;
    uint8_t *aMatrixDevice = nullptr;
    uint8_t *aMatrixDeviceWork = nullptr;
    uint8_t *wHost = nullptr;
    uint8_t *wDevice = nullptr;
    uint8_t *pDevice = nullptr;
    uint8_t *bMatrixHost = nullptr;
    uint8_t *bMatrixDevice = nullptr;
    uint8_t *bMatrixDeviceWork = nullptr;
    uint8_t *gatherHost = nullptr;
    uint8_t *gatherDevice = nullptr;
};

// 释放所有已申请的 HBM / host 内存；未申请的指针为空，可安全跳过
void ReleaseCgetrsBuffers(CgetrsBuffers &buf)
{
    if (buf.aMatrixDevice) aclrtFree(buf.aMatrixDevice);
    if (buf.aMatrixDeviceWork) aclrtFree(buf.aMatrixDeviceWork);
    if (buf.wDevice) aclrtFree(buf.wDevice);
    if (buf.pDevice) aclrtFree(buf.pDevice);
    if (buf.bMatrixDevice) aclrtFree(buf.bMatrixDevice);
    if (buf.bMatrixDeviceWork) aclrtFree(buf.bMatrixDeviceWork);
    if (buf.gatherDevice) aclrtFree(buf.gatherDevice);
    if (buf.gatherHost) aclrtFreeHost(buf.gatherHost);
}

// 行宽向上取整到 cache line，2 的整数次幂达到阈值时再 pad 一个 cache line，提高访存效率
int CalcPadStride(int x)
{
    int stride = (x + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    if ((stride & -stride) >= ADDITIONAL_PAD_SIZE) stride += CACHE_LINE_SIZE_FP32;
    return stride;
}

// 入参契约校验（口径注释见 aclsolverCgetrs 函数头），失败返回对应错误码
aclError CheckCgetrsArgs(const uint64_t trans, const int64_t n, const int64_t nrhs, const int64_t lda,
                         const int64_t ldb, aclFloatComplex *A, aclFloatComplex *B, int32_t *ipiv, int32_t *info)
{
    SOLVER_ECHECK(n > 0 && n <= 8192 && nrhs > 0 && nrhs <= 8192 && lda >= n && ldb >= nrhs && A != nullptr &&
                      B != nullptr && ipiv != nullptr && info != nullptr && lda - n <= 8191 && ldb - nrhs <= 8191 &&
                      (nrhs <= 4096 || ldb - nrhs <= 4095),
                  "aclsolverCgetrs invalid param: n, nrhs out of [1, 8192], n, nrhs, lda, ldb <= 0, lda < n, "
                  "ldb < nrhs, lda - n > 8191, ldb - nrhs > 8191 (or > 4095 when nrhs > 4096), "
                  "or A, B, ipiv, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(trans <= 2, "aclsolverCgetrs invalid param: trans must be 0, 1 or 2.", ACL_ERROR_INVALID_PARAM);
    // ipiv 内容校验对齐 getri 链路防御口径（issue #299），实现见 utils/pivot_check.h
    if (!ValidatePivots(ipiv, n, info, "aclsolverCgetrs"))
    {
        return ACL_ERROR_INVALID_PARAM;
    }
    return ACL_SUCCESS;
}

// 申请并上传 A/ipiv/B 与工作区缓冲（含 gather 表设备缓冲），失败时释放已申请部分
aclError AllocAndUploadCgetrs(CgetrsBuffers &buf, int N, int NRHS, int lda, int ldb, int resizedN, int strideN,
                              int strideNRHS)
{
    size_t aMatrixFileSize = N * lda * sizeof(float) * 2;
    size_t bMatrixFileSize = N * ldb * sizeof(float) * 2;
    size_t wFileSize = N * sizeof(int);
    size_t resizedAMatrixFileSize = resizedN * strideN * sizeof(float) * 2;
    size_t resizedBMatrixFileSize = resizedN * strideNRHS * sizeof(float) * 3;
    size_t gatherSize = GATHER_SEG_LEN * 2 * sizeof(uint32_t);

    CHECK_ACLRT(aclrtMalloc((void **)&buf.aMatrixDevice, aMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(
        aclrtMemcpy(buf.aMatrixDevice, aMatrixFileSize, buf.aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
        ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.aMatrixDeviceWork, resizedAMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.wDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.pDevice, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy(buf.wDevice, wFileSize, buf.wHost, wFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMemset(buf.pDevice, wFileSize, -1, wFileSize), ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.bMatrixDevice, bMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(
        aclrtMemcpy(buf.bMatrixDevice, bMatrixFileSize, buf.bMatrixHost, bMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
        ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.bMatrixDeviceWork, resizedBMatrixFileSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void **)(&buf.gatherHost), gatherSize), ReleaseCgetrsBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.gatherDevice, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCgetrsBuffers(buf));
    return ACL_SUCCESS;
}
}  // namespace

aclError aclsolverCgetrs(aclsolverHandle_t handle, const uint64_t trans, const int64_t n, const int64_t nrhs,
                         aclFloatComplex *A, const int64_t lda, int32_t *ipiv, aclFloatComplex *B, const int64_t ldb,
                         int32_t *info)
{
    // n > 8192 会使 SplitRealImag 后 alignedN > TILE_LENGTH，CopyIn 第二段超出 UB buffer（issue #266 检视）；
    // 与 aclsolverSgetrf 的文档口径（valid range [1, 8192]）保持一致；
    // nrhs 无上界时 int64 转 int 静默截断可致按错值分配与寻址，与 aclsolverSgetrs 的 nrhs 契约保持一致（issue #277）；
    // kernel 侧 uint16_t 字节计数承载 DataCopyExtParams 行间隙，复数实虚交织布局每元素占 8 字节；
    // SplitRealImag::CopyIn 行间隙为 (lda - n) * 8 / (ldb - nrhs) * 8 字节，须 <= 65535，即元素差 <= 8191；
    // trans == 0 时 restore 的 MergeRealImag::CopyOut 对 nrhs > 4096 的第二段行间隙为
    // (ldb - nrhs + 4096) * 8 字节，故 nrhs > 4096 时需进一步限制 ldb - nrhs <= 4095（issue #266 检视）。
    aclError check = CheckCgetrsArgs(trans, n, nrhs, lda, ldb, A, B, ipiv, info);
    if (check != ACL_SUCCESS)
    {
        return check;
    }

    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        // GetStream 失败被吞时 stream 落 nullptr，kernel 同步静默改用默认流，破坏调用方
        // 流语义（issue #311）；这里校验返回值并透传底层错误码，对齐 cheevj GetCheevjStream
        // 的 #255 修复口径。
        const aclError streamRet = aclsolverGetStream(handle, &stream);
        SOLVER_ECHECK(streamRet == ACL_SUCCESS, "aclsolverCgetrs failed to get stream from handle.", streamRet);
    }

    // 核数防护对齐 cgeqrf/cpotrf 家族口径（issue #362）：平台实例获取失败与核数为 0
    // 都以 0 返回，统一在下方 SOLVER_ECHECK 拦截，避免 blockDim=0 直接下发 kernel。
    uint32_t numBlocks = GetAiCoreNumOrZero();
    SOLVER_ECHECK(numBlocks > 0, "No available AIcore found on current device.", ACL_ERROR_RT_FEATURE_NOT_SUPPORT);

    int N = n, NRHS = nrhs, blockDim = numBlocks, blockM = BLOCK_M;
    int resizedN = (N + CACHE_LINE_SIZE_FP32 - 1) / CACHE_LINE_SIZE_FP32 * CACHE_LINE_SIZE_FP32;
    int strideN = CalcPadStride(N);
    int strideNRHS = CalcPadStride(NRHS);

    CgetrsBuffers buf;
    buf.aMatrixHost = reinterpret_cast<uint8_t *>(A);
    buf.wHost = reinterpret_cast<uint8_t *>(ipiv);
    buf.bMatrixHost = reinterpret_cast<uint8_t *>(B);
    CHECK_ACLRT(AllocAndUploadCgetrs(buf, N, NRHS, lda, ldb, resizedN, strideN, strideNRHS), ReleaseCgetrsBuffers(buf));

    // GenerateTiling 无失败分支（issue #364），调用处恒假的错误分支已随签名改为 void 一并清理
    GenerateTiling(buf.gatherHost);
    size_t gatherSize = GATHER_SEG_LEN * 2 * sizeof(uint32_t);
    CHECK_ACLRT(aclrtMemcpy(buf.gatherDevice, gatherSize, buf.gatherHost, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE),
                ReleaseCgetrsBuffers(buf));

    cgetrs_kernel_do(N, NRHS, trans, lda, ldb, blockM, buf.aMatrixDevice, buf.aMatrixDeviceWork, buf.wDevice,
                     buf.pDevice, buf.bMatrixDevice, buf.bMatrixDeviceWork, buf.gatherDevice, blockDim, stream);

    // Kernel launches are asynchronous; do not report success when enqueueing failed (issue #361)。
    // 失败路径的释放语义对齐 cpotrf_host.cpp（issue #350 口径）：先等在途任务落定再释放，
    // 同步本身也失败时滞留（有界泄漏优于 use-after-free）。
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS)
    {
        if (aclrtSynchronizeStream(stream) == ACL_SUCCESS)
        {
            ReleaseCgetrsBuffers(buf);
        }
        return enqueueStatus;
    }
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ReleaseCgetrsBuffers(buf));

    size_t bMatrixFileSize = N * ldb * sizeof(float) * 2;
    CHECK_ACLRT(
        aclrtMemcpy(buf.bMatrixHost, bMatrixFileSize, buf.bMatrixDevice, bMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
        ReleaseCgetrsBuffers(buf));

    ReleaseCgetrsBuffers(buf);

    *info = 0;

    return ACL_SUCCESS;
}
