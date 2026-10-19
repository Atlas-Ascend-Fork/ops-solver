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
 * \file cgetri_batched_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

#include "../utils/assert.h"
#include "../utils/gm_addr.h"
#include "../utils/lu_host_common.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

extern void cgetri_batched_kernel_do(GM_ADDR sync, GM_ADDR A_org, GM_ADDR W, GM_ADDR gather1_gm, GM_ADDR gather2_gm,
                                     GM_ADDR gather3_gm, GM_ADDR eye_gm, GM_ADDR A_work, GM_ADDR work_gm, GM_ADDR A_inv,
                                     GM_ADDR workspace, GM_ADDR tiling_gm, uint32_t numBlocks, void *stream);

static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;

static constexpr int64_t DTYPE_COMPLEX32 = 0;
static constexpr int64_t DTYPE_COMPLEX64 = 1;

static constexpr int64_t EYE_MATRIX_NUM = 3;
static constexpr int64_t BASE_BLOCK_ELENUM = 16;
static constexpr int64_t COL_ALIGNED_ELENUM = 128;
static constexpr int64_t TILE_ELENUM = 512;
static constexpr int64_t TILE_LENGTH = 4096;
static constexpr int64_t COMPLEX_ELENUM = 2;

static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

struct CgetriBatchedTilingData
{
    uint32_t dtype;
    uint32_t n;
    uint32_t batchSize;
    uint32_t blockM;
    uint32_t blockN;
    uint32_t tileM;
};

// Host-side staging buffers for a full batch can reach several GiB at the
// documented upper bounds (n=256, batch=3000).  Reject combinations whose
// staging footprint exceeds a conservative budget instead of attempting a
// multi-GiB host allocation, and translate any allocation failure into an
// error code rather than letting std::bad_alloc escape the C boundary
// (issue #127).
aclError CgetriBatchedImpl(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                           std::complex<float> *Ainv, const int64_t lda_inv, int32_t *info, int64_t batchSize);

aclError aclsolverCgetriBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                                std::complex<float> *Ainv, const int64_t lda_inv, int32_t *info, int64_t batchSize)
{
    // info 遵循 LAPACK 语义（issue #164）：成功为 0，参数错误由下方 ECHECK 以负值
    // 语义返回（aclError）；奇异信息见各 kernel 主元哨兵约定
    if (info != nullptr)
    {
        *info = 0;
    }
    // 参数范围校验先于一切估算算术执行：体积估算的乘法仅在通过范围校验的 n/batchSize 上
    // 运行，消除大 n（约 1.75e9 以上）下 int64 有符号溢出 UB 与误导性报错（issue #139，
    // 检视意见：此前仅前移了校验到守卫 CHECK 之前，估算算术本身仍在校验前执行）
    // info 允许为空（issue #244，对齐 getrf 族 #180 的 nullptr 容错口径），
    // 与上方 info != nullptr 判空清零分支统一，消除自相矛盾
    SOLVER_ECHECK(n > 0 && batchSize > 0 && lda > 0 && lda_inv > 0 && A != nullptr && Ainv != nullptr,
                  "CgetriBatched get invalid param: n, batchSize, lda, lda_inv <= 0, or A, Ainv is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "CgetriBatched get n > 256 or batchSize > 3000, which exceeds the supported limit.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n >= MATRIX_SHAPE_LIMIT,
                  "CgetriBatched only supports n >= 32. For n < 32, use "
                  "CmatinvBatched instead.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(lda == n && lda_inv == n, "CgetriBatched only supports lda == n and lda_inv == n in current version.",
                  ACL_ERROR_INVALID_PARAM);

    // Staging guard: cap the combined host staging estimate.
    // 估算口径（issue #232 修正 #207 的单位错误）：按 Impl 实际分配构成重算——
    // 逐 batch 缓冲仅 wBatchData（batchNum*t 个 uint32_t）与 eyeBatchMatData
    // （batchNum*eyeMatEleNum 个 float）；aWorkData/workGmData/gatherOffset×3
    // 均为单份共享缓冲（不乘 batchSize）。前置校验已强制 n ≤ 256、batchSize ≤
    // 3000，该估算式在合法域内上界有限（约 4GiB），守卫作为越界/未来参数域
    // 扩展时的预检保留，合法域内不触发（有界性依据见上）。
    const int64_t alignedN = (n + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM;
    const int64_t paddedM = (n + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM * BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM;
    const int64_t eyeMatEleNum = alignedN * paddedM * EYE_MATRIX_NUM;
    const int64_t workEleNum = ((n + TILE_ELENUM - 1) / TILE_ELENUM * TILE_ELENUM) * BASE_BLOCK_ELENUM * COMPLEX_ELENUM;
    const int64_t wPivotEleNum = ((n + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM);
    constexpr double kMaxStagingGiB = 8.0;
    const double perBatchBytes =
        static_cast<double>(wPivotEleNum) * sizeof(uint32_t) + static_cast<double>(eyeMatEleNum) * sizeof(float);
    const double sharedBytes = (static_cast<double>(eyeMatEleNum) + static_cast<double>(workEleNum)) * sizeof(float);
    const double stagingGiB =
        (static_cast<double>(batchSize) * perBatchBytes + sharedBytes) / (1024.0 * 1024.0 * 1024.0);
    SOLVER_ECHECK(stagingGiB <= kMaxStagingGiB,
                  "CgetriBatched host staging buffers exceed the supported memory budget for this n/batchSize.",
                  ACL_ERROR_INVALID_PARAM);
    try
    {
        return CgetriBatchedImpl(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
    }
    catch (const std::bad_alloc &)
    {
        // OOM 与参数错误分口径上报（issue #168）：bad_alloc 属运行时资源问题，
        // 与同函数其他 std::exception 的 INTERNAL_ERROR 口径一致
        std::cerr << "CgetriBatched host allocation failed for n=" << n << " batchSize=" << batchSize << std::endl;
        return ACL_ERROR_INTERNAL_ERROR;
    }
    catch (const std::exception &e)
    {
        std::cerr << "CgetriBatched unexpected host failure: " << e.what() << std::endl;
        return ACL_ERROR_INTERNAL_ERROR;
    }
}

aclError CgetriBatchedImpl(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A, const int64_t lda,
                           std::complex<float> *Ainv, const int64_t lda_inv, int32_t *info, int64_t batchSize)
{
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }

    // 与 sgetrf/cgetrf 对齐：kernel 侧 LU 实现按固定 20 块设计（blockIdx 8/10 负责
    // A/eye 主元行交换、coreIdx>=4 承担 trsm/gemm），块数不足时这些角色无人执行而
    // 求逆结果静默错误，故必须固定使用 LU_MAX_NUM_BLOCKS（issue #147）
    const uint32_t numBlocks = LU_MAX_NUM_BLOCKS;

    // 参数校验已由外层 aclsolverCgetriBatched 在 staging 估算前单点完成（issue #139/#158），
    // 此处不再重复：同一文案双点维护一旦只改其一即产生口径漂移。

    int64_t M = n;
    int64_t N = n;
    int64_t batchNum = batchSize;
    int64_t tileM = TILE_ELENUM;
    int64_t blockM = tileM;
    int64_t blockN = BASE_BLOCK_ELENUM;

    // 1、 gen batch w data
    int64_t t = (std::min(M, N) + blockN - 1) / blockN * blockN;
    int64_t wEleNum = batchNum * t;
    std::vector<uint32_t> wBatchData(wEleNum, 0);
    // 哨兵初始化失败必须中止（issue #167）：kernel 依赖 -1（0xFFFFFFFF）区分
    // 未消元主元，残留 0 会被当作真实主元行号 0；口径对齐 cgetri_host 的 EOK 检查
    if (memset_s(wBatchData.data(), wEleNum * sizeof(uint32_t), -1, wEleNum * sizeof(uint32_t)) != EOK)
    {
        // Impl 内 cleanup lambda 在下方才可见，此处以异常交由外层 catch 统一
        // 上报 INTERNAL_ERROR（issue #167 口径：哨兵初始化失败必须中止）
        throw std::runtime_error("CgetriBatched pivot sentinel memset_s failed");
    }

    // 2、gen work data
    int64_t aMatWorkEleNum =
        ((N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM) *
        ((M + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM * BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) * EYE_MATRIX_NUM;
    int64_t workM = (M + tileM - 1) / tileM * tileM;
    int64_t workEleNum = workM * blockN * COMPLEX_ELENUM;
    std::vector<float> aWorkData(aMatWorkEleNum, 0.0f);
    std::vector<float> workGmData(workEleNum, 0.0f);

    // 3、generate offset and eye mat data
    int64_t gatherEleNum = tileM * blockN;
    int64_t eyeMatEleNum = ((N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM) *
                           ((M + BASE_BLOCK_ELENUM - 1) / BASE_BLOCK_ELENUM * BASE_BLOCK_ELENUM + COL_ALIGNED_ELENUM) *
                           EYE_MATRIX_NUM;

    std::vector<uint32_t> gatherOffset1(gatherEleNum, 0);
    std::vector<uint32_t> gatherOffset2(gatherEleNum, 0);
    std::vector<uint32_t> gatherOffset3(gatherEleNum, 0);
    std::vector<float> eyeBatchMatData(batchNum * eyeMatEleNum, 0.0f);

    int64_t strideN = (N + COL_ALIGNED_ELENUM - 1) / COL_ALIGNED_ELENUM * COL_ALIGNED_ELENUM;

    // 三张 gather 重排表收敛为 utils 共享实现（issue #171，第 6 处拷贝归一）：
    // gather1/gather2 语义与共享实现完全一致；gather3 共享实现含 n 非 8 倍数时
    // 尾行补齐口径（issue #161 修复）。
    // 表生成移出 batch 循环（issue #179）：GenerateGatherTables 的输入
    // (tileM, blockN, N, strideN) 逐 batch 完全相同，三张表每个 batch 一模一样，
    // 循环前生成一次即可；逐 batch 真正变化的只有下方 eye 的对角线写入
    {
        auto *g1 = reinterpret_cast<uint8_t *>(gatherOffset1.data());
        auto *g2 = reinterpret_cast<uint8_t *>(gatherOffset2.data());
        auto *g3 = reinterpret_cast<uint8_t *>(gatherOffset3.data());
        // 共享实现 eyeBuf=nullptr 时为纯整数运算，无失败路径（见 lu_host_common.h 注释）
        (void)GenerateGatherTables(tileM, blockN, N, strideN, N, g1, g2, g3, nullptr, EYE_FLOATS_PER_REAL_ELEMENT);
    }
    for (int64_t b = 0; b < batchNum; ++b)
    {
        // 批量 eye：第 b 个 batch 的单位阵写入共享 eye 布局的对应平面
        for (int64_t i = 0; i < N; ++i)
        {
            eyeBatchMatData[b * eyeMatEleNum + i * (strideN + 1)] = 1.0f;
        }
    }

    uint8_t *host_wBatch = reinterpret_cast<uint8_t *>(wBatchData.data());
    uint8_t *host_aWork = reinterpret_cast<uint8_t *>(aWorkData.data());
    uint8_t *host_workGm = reinterpret_cast<uint8_t *>(workGmData.data());
    uint8_t *host_gather1Offset = reinterpret_cast<uint8_t *>(gatherOffset1.data());
    uint8_t *host_gather2Offset = reinterpret_cast<uint8_t *>(gatherOffset2.data());
    uint8_t *host_gather3Offset = reinterpret_cast<uint8_t *>(gatherOffset3.data());
    uint8_t *host_eyeBatchMat = reinterpret_cast<uint8_t *>(eyeBatchMatData.data());

    size_t size_wBatch = wEleNum * sizeof(uint32_t);
    size_t size_aWork = aMatWorkEleNum * sizeof(float);
    size_t size_workGm = workEleNum * sizeof(float);
    size_t size_gather1Offset = gatherEleNum * sizeof(uint32_t);
    size_t size_gather2Offset = gatherEleNum * sizeof(uint32_t);
    size_t size_gather3Offset = gatherEleNum * sizeof(uint32_t);
    size_t size_eyeBatchMat = batchNum * eyeMatEleNum * sizeof(float);

    uint8_t *d_wBatch = nullptr;
    uint8_t *d_aWork = nullptr;
    uint8_t *d_workGm = nullptr;
    uint8_t *d_gather1Offset = nullptr;
    uint8_t *d_gather2Offset = nullptr;
    uint8_t *d_gather3Offset = nullptr;
    uint8_t *d_eyeBatchMat = nullptr;
    uint8_t *d_A = nullptr;
    uint8_t *d_Ainv = nullptr;
    uint8_t *tilingDevice = nullptr;
    uint8_t *workSpace = nullptr;
    uint8_t *sync = nullptr;

    auto cleanup = [&]()
    {
        if (d_A) aclrtFree(d_A);
        if (d_Ainv) aclrtFree(d_Ainv);
        if (d_wBatch) aclrtFree(d_wBatch);
        if (d_gather1Offset) aclrtFree(d_gather1Offset);
        if (d_gather2Offset) aclrtFree(d_gather2Offset);
        if (d_gather3Offset) aclrtFree(d_gather3Offset);
        if (d_eyeBatchMat) aclrtFree(d_eyeBatchMat);
        if (d_aWork) aclrtFree(d_aWork);
        if (d_workGm) aclrtFree(d_workGm);
        if (tilingDevice) aclrtFree(tilingDevice);
        if (workSpace) aclrtFree(workSpace);
    };

    CHECK_ACLRT(aclrtMalloc((void **)&d_wBatch, size_wBatch, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_aWork, size_aWork, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_workGm, size_workGm, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_gather1Offset, size_gather1Offset, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_gather2Offset, size_gather2Offset, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_gather3Offset, size_gather3Offset, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_eyeBatchMat, size_eyeBatchMat, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMemcpy(d_wBatch, size_wBatch, host_wBatch, size_wBatch, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_aWork, size_aWork, host_aWork, size_aWork, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_workGm, size_workGm, host_workGm, size_workGm, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_gather1Offset, size_gather1Offset, host_gather1Offset, size_gather1Offset,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_gather2Offset, size_gather2Offset, host_gather2Offset, size_gather2Offset,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_gather3Offset, size_gather3Offset, host_gather3Offset, size_gather3Offset,
                            ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());
    CHECK_ACLRT(
        aclrtMemcpy(d_eyeBatchMat, size_eyeBatchMat, host_eyeBatchMat, size_eyeBatchMat, ACL_MEMCPY_HOST_TO_DEVICE),
        cleanup());

    uint8_t *hostA = reinterpret_cast<uint8_t *>(A);
    uint8_t *hostAinv = reinterpret_cast<uint8_t *>(Ainv);
    auto sizeA = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeAinv = batchSize * n * n * sizeof(std::complex<float>);
    CHECK_ACLRT(aclrtMalloc((void **)&d_A, sizeA, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&d_Ainv, sizeAinv, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_A, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());
    CHECK_ACLRT(aclrtMemcpy(d_Ainv, sizeAinv, hostAinv, sizeAinv, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());

    CgetriBatchedTilingData tilingData;
    tilingData.dtype = DTYPE_COMPLEX64;
    tilingData.n = n;
    tilingData.batchSize = batchSize;
    tilingData.blockM = BASE_BLOCK_ELENUM;
    tilingData.blockN = BASE_BLOCK_ELENUM;
    tilingData.tileM = TILE_ELENUM;
    CHECK_ACLRT(aclrtMalloc((void **)&tilingDevice, sizeof(CgetriBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST),
                cleanup());
    CHECK_ACLRT(aclrtMemcpy(tilingDevice, sizeof(CgetriBatchedTilingData), &tilingData, sizeof(CgetriBatchedTilingData),
                            ACL_MEMCPY_HOST_TO_DEVICE),
                cleanup());
    CHECK_ACLRT(aclrtMalloc((void **)&workSpace, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    cgetri_batched_kernel_do(sync, d_A, d_wBatch, d_gather1Offset, d_gather2Offset, d_gather3Offset, d_eyeBatchMat,
                             d_aWork, d_workGm, d_Ainv, workSpace, tilingDevice, numBlocks, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(hostAinv, sizeAinv, d_Ainv, sizeAinv, ACL_MEMCPY_DEVICE_TO_HOST), cleanup());

    cleanup();

    return ACL_SUCCESS;
}
