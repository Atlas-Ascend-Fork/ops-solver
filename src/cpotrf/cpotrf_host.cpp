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
 * \file cpotrf_host.cpp
 * \brief
 */

#include <securec.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <queue>
#include <vector>

#include "../utils/assert.h"
#include "../utils/free_reset.h"
#include "../utils/gm_addr.h"
#include "../utils/potrf_gemm_work.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "cpotrf_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"

void GenerateTiling(bool transpose, int M, int N, uint8_t* gatherBuf)
{
    static constexpr int m = 64;
    static constexpr int n = 64;
    auto buf = reinterpret_cast<uint32_t*>(gatherBuf);
    if (transpose)
    {
        int idx = 0;
        static constexpr int TILE_LENGTH = 4096;

        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
            {
                buf[idx++] = (i * n + j) * sizeof(float);
                buf[idx++] = (i * n + j + TILE_LENGTH) * sizeof(float);
            }

        buf = reinterpret_cast<uint32_t*>(gatherBuf + 2 * TILE_LENGTH * sizeof(float));
        idx = 0;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) buf[idx++] = 2 * (i * n + j) * sizeof(float);
    }
    else
    {
        int idx = 0;
        static constexpr int TILE_LENGTH = 4096;

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
            {
                buf[idx++] = (i * n + j) * sizeof(float);
                buf[idx++] = (i * n + j + TILE_LENGTH) * sizeof(float);
            }
    }
}

namespace
{
// 按块数与矩阵规模计算各列的 tile 配额 b 与按体量分摊的 workload（原 GenerateGemmWork 前段）
void ComputeTileQuota(int n, int blockNum, std::vector<int>& b, std::vector<int>& workload)
{
    const int len = n - 2;
    int total = 0;
    int denominator = blockNum;
    for (int i = 1; i <= n - 2; ++i)
    {
        int numerator = n - i;
        int val = (numerator + denominator - 1) / denominator;
        b[i - 1] = val;
        total += val;
    }
    if (total > 0)
    {
        for (int i = 1; i <= len; ++i)
        {
            int64_t num = static_cast<int64_t>(b[i - 1]) * n * (n - 1) * (n - 2);
            int denom = 6LL * total;
            workload[i - 1] = static_cast<int>((num + denom - 1) / denom);
        }
    }
}

// 写入单个 gemm 地址对：发单表偏移以 uint32 存储，中间乘积一律按 int64 计算后再收窄，
// 防止分块数较大时 int 乘法回绕为负（issue #308）
void StoreTileOffsets(uint32_t* aOffset, uint32_t* bOffset, uint32_t* cOffset, int& idx, int n, int y, int x, int z,
                      int ptr)
{
    const int64_t aOff = static_cast<int64_t>(y) * 128 * n * 128 + static_cast<int64_t>(x) * 128;
    aOffset[idx] = static_cast<uint32_t>(aOff);
    bOffset[idx] = static_cast<uint32_t>(aOff + static_cast<int64_t>(z - ptr) * 128);
    cOffset[idx] = static_cast<uint32_t>(static_cast<int64_t>(x) * 128 * n * 128 + static_cast<int64_t>(x) * 128 +
                                         static_cast<int64_t>(z - ptr) * 128);
    idx++;
}

// 写入 count 个 (y, x, z) tile 的 gemm 地址对（z 固定，随 ptr 递减取偏移）
void FillTileOffsets(uint32_t* aOffset, uint32_t* bOffset, uint32_t* cOffset, int& idx, int n, int y, int x, int z,
                     int count)
{
    for (int ptr = 1; ptr <= count; ++ptr)
    {
        StoreTileOffsets(aOffset, bOffset, cOffset, idx, n, y, x, z, ptr);
    }
}

// 优先队列中属于当前列次对角（top.x == i + 1）的 tile 全量配给当前列
void DrainMatchedTiles(std::priority_queue<L, std::vector<L>, std::greater<L>>& pq, int i, int& wl, int n,
                       uint32_t* aOffset, uint32_t* bOffset, uint32_t* cOffset, int& idx, uint32_t& offsetIdx)
{
    while (!pq.empty())
    {
        L top = pq.top();
        if (top.x != i + 1)
        {
            break;
        }
        wl -= top.z;
        FillTileOffsets(aOffset, bOffset, cOffset, idx, n, top.y, top.x, top.z, top.z);
        offsetIdx += top.z;
        pq.pop();
    }
}

// 当前列配额未用完时，从队列继续取 tile 分摊；跨界 tile 剩余部分回队
void DistributeRemainingTiles(std::priority_queue<L, std::vector<L>, std::greater<L>>& pq, int& wl, int n,
                              uint32_t* aOffset, uint32_t* bOffset, uint32_t* cOffset, int& idx, uint32_t& offsetIdx)
{
    while (wl > 0 && !pq.empty())
    {
        L top = pq.top();
        pq.pop();
        if (top.z < wl)
        {
            wl -= top.z;
            FillTileOffsets(aOffset, bOffset, cOffset, idx, n, top.y, top.x, top.z, top.z);
            offsetIdx += top.z;
        }
        else
        {
            FillTileOffsets(aOffset, bOffset, cOffset, idx, n, top.y, top.x, top.z, wl);
            top.z -= wl;
            offsetIdx += wl;
            wl = 0;
            if (top.z > 0)
            {
                pq.push(top);
            }
        }
    }
}

// 逐列调度：列 i 的 tile 入队，先配给次对角 tile，再分摊剩余 workload
void ScheduleGemmTiles(int n, std::vector<int>& workload, uint32_t* aOffset, uint32_t* bOffset, uint32_t* cOffset,
                       uint32_t* OffsetIdx, int& idx)
{
    std::priority_queue<L, std::vector<L>, std::greater<L>> pq;
    for (int i = 1; i <= n - 2; ++i)
    {
        OffsetIdx[i - 1] = 0;
        PushColumnTiles(pq, n, i);
        int& wl = workload[i - 1];
        DrainMatchedTiles(pq, i, wl, n, aOffset, bOffset, cOffset, idx, OffsetIdx[i - 1]);
        DistributeRemainingTiles(pq, wl, n, aOffset, bOffset, cOffset, idx, OffsetIdx[i - 1]);
    }
}
}  // namespace

void GenerateGemmWork(int n, int blockNum, uint8_t* _aOffset, uint8_t* _bOffset, uint8_t* _cOffset, uint8_t* _OffsetIdx)
{
    auto aOffset = reinterpret_cast<uint32_t*>(_aOffset);
    auto bOffset = reinterpret_cast<uint32_t*>(_bOffset);
    auto cOffset = reinterpret_cast<uint32_t*>(_cOffset);
    auto OffsetIdx = reinterpret_cast<uint32_t*>(_OffsetIdx);
    // k < 3（N <= 256）或核数异常时直接返回，四张表保持未写入状态，上传前由调用方清零兜底
    if (n < 3 || blockNum <= 0)
    {
        return;
    }
    int idx = 0;
    const int len = n - 2;
    std::vector<int> b(len, 0);
    std::vector<int> workload(len, 0);
    ComputeTileQuota(n, blockNum, b, workload);
    ScheduleGemmTiles(n, workload, aOffset, bOffset, cOffset, OffsetIdx, idx);
}
struct CpotrfBuffers
{
    uint8_t* aHost = nullptr;
    uint8_t* aDevice = nullptr;
    uint8_t* aDeviceWork = nullptr;
    uint8_t* offsetHost1 = nullptr;
    uint8_t* offsetHost2 = nullptr;
    uint8_t* offsetHost3 = nullptr;
    uint8_t* offsetDevice1 = nullptr;
    uint8_t* offsetDevice2 = nullptr;
    uint8_t* offsetDevice3 = nullptr;
    uint8_t* offsetIdxHost = nullptr;
    uint8_t* offsetIdxDevice = nullptr;
    uint8_t* gatherHost = nullptr;
    uint8_t* gatherDevice = nullptr;
    uint8_t* sync = nullptr;
};

// 释放所有已申请的 HBM / host 内存；未申请的指针为空，可安全跳过。
// 释放后立即置空（issue #186 同款口径）：多处失败路径共用该清理，
// 幂等语义防止同一指针被二次 aclrtFree。
void ReleaseCpotrfBuffers(CpotrfBuffers& buf)
{
    FreeAndReset(buf.offsetDevice1, aclrtFree);
    FreeAndReset(buf.offsetDevice2, aclrtFree);
    FreeAndReset(buf.offsetDevice3, aclrtFree);
    FreeAndReset(buf.offsetIdxDevice, aclrtFree);
    FreeAndReset(buf.offsetHost1, aclrtFreeHost);
    FreeAndReset(buf.offsetHost2, aclrtFreeHost);
    FreeAndReset(buf.offsetHost3, aclrtFreeHost);
    FreeAndReset(buf.offsetIdxHost, aclrtFreeHost);
    FreeAndReset(buf.gatherDevice, aclrtFree);
    FreeAndReset(buf.gatherHost, aclrtFreeHost);
    FreeAndReset(buf.aDevice, aclrtFree);
    FreeAndReset(buf.aDeviceWork, aclrtFree);
}

// A 矩阵上载：HBM 侧按 2*N 个 float 行宽，host 侧按 2*lda 个 float 行宽
aclError AllocAndUploadMatrix(CpotrfBuffers& buf, int64_t N, int64_t lda, size_t workSize, aclrtStream stream)
{
    CHECK_ACLRT(aclrtMalloc((void**)&buf.aDevice, 2 * N * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy2dAsync(buf.aDevice, 2 * N * sizeof(float), buf.aHost, 2 * lda * sizeof(float),
                                   2 * N * sizeof(float), N, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.aDeviceWork, workSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseCpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// GEMM 发单表：host 侧算好后整块搬到 device
aclError AllocAndUploadGemmWork(CpotrfBuffers& buf, int k, size_t offsetSize, uint32_t blockDim, aclrtStream stream)
{
    CHECK_ACLRT(aclrtMallocHost((void**)(&buf.offsetHost1), offsetSize), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void**)(&buf.offsetHost2), offsetSize), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void**)(&buf.offsetHost3), offsetSize), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void**)(&buf.offsetIdxHost), k * sizeof(int)), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.offsetDevice1, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.offsetDevice2, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.offsetDevice3, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.offsetIdxDevice, k * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    // GenerateGemmWork 对 k < 3（即 N <= 256）直接返回，四张表保持未写入状态；上传前必须清零，
    // 否则 kernel 读到堆垃圾 workNum 后会对任意 GM 地址发起 GEMM（issue #263，对齐 spotrf_host 的 memset_s 防护）。
    if (memset_s(buf.offsetHost1, offsetSize, 0, offsetSize) != EOK ||
        memset_s(buf.offsetHost2, offsetSize, 0, offsetSize) != EOK ||
        memset_s(buf.offsetHost3, offsetSize, 0, offsetSize) != EOK ||
        memset_s(buf.offsetIdxHost, k * sizeof(int), 0, k * sizeof(int)) != EOK)
    {
        ReleaseCpotrfBuffers(buf);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    GenerateGemmWork(k, blockDim, buf.offsetHost1, buf.offsetHost2, buf.offsetHost3, buf.offsetIdxHost);
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.offsetDevice1, offsetSize, buf.offsetHost1, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.offsetDevice2, offsetSize, buf.offsetHost2, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.offsetDevice3, offsetSize, buf.offsetHost3, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpyAsync(buf.offsetIdxDevice, k * sizeof(int), buf.offsetIdxHost, k * sizeof(int),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseCpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// gather 索引表：决定实虚拆分/合并时的地址跳转
aclError AllocAndUploadGather(CpotrfBuffers& buf, size_t gatherSize, bool lower, int64_t N, aclrtStream stream)
{
    CHECK_ACLRT(aclrtMallocHost((void**)(&buf.gatherHost), gatherSize), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.gatherDevice, gatherSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    GenerateTiling(lower, static_cast<int>(N), static_cast<int>(N), buf.gatherHost);
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.gatherDevice, gatherSize, buf.gatherHost, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseCpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// 下发 kernel，等待完成后把结果按 lda 行宽搬回 host。
// 失败路径的释放语义（issue #350，对齐 cholesky_host 的 issue #325 口径）：
//   - kernel 下发失败：先等已入队任务落定，落定后释放；
//   - 流同步失败：kernel 可能仍在读写这些缓冲，立即释放构成 use-after-free，
//     故意滞留（有界泄漏优于 UAF），不做释放；
//   - 同步成功后的回拷失败：内核已落定，立即释放安全。
aclError LaunchCpotrf(CpotrfBuffers& buf, bool transpose, int64_t N, int64_t lda, uint32_t blockDim, aclrtStream stream)
{
    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void**)(&buf.sync)), ReleaseCpotrfBuffers(buf));
    cpotrf_kernel_do(buf.sync, transpose, static_cast<int>(N), static_cast<int>(N), buf.aDevice, buf.aDeviceWork,
                     buf.offsetDevice1, buf.offsetDevice2, buf.offsetDevice3, buf.offsetIdxDevice, buf.gatherDevice,
                     blockDim, stream);
    // Kernel launches are asynchronous; do not report success when enqueueing failed.
    const aclError enqueueStatus = aclrtPeekAtLastError(ACL_RT_THREAD_LEVEL);
    if (enqueueStatus != ACL_SUCCESS)
    {
        // 同步等待在途任务落定后再释放；同步本身也失败时按下方口径滞留。
        if (aclrtSynchronizeStream(stream) == ACL_SUCCESS)
        {
            ReleaseCpotrfBuffers(buf);
        }
        return enqueueStatus;
    }
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ((void)0));
    CHECK_ACLRT(aclrtMemcpy2dAsync(buf.aHost, 2 * lda * sizeof(float), buf.aDevice, 2 * N * sizeof(float),
                                   2 * N * sizeof(float), N, ACL_MEMCPY_DEVICE_TO_HOST, stream),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ((void)0));
    return ACL_SUCCESS;
}

aclError aclsolverCpotrf(aclsolverHandle_t handle, const char uplo, const int64_t n, std::complex<float>* A,
                         const int64_t lda, int32_t* info)
{
    // 参数契约对齐同族 aclsolverSpotrf（spotrf_host.cpp）：n 取 [1, 8192]（与
    // aclsolverSgetrf 文档口径一致）、lda <= 16384（kernel 行间隙 uint16 计数约束）。
    // n 上界同时保证 GEMM 发单表偏移恒在 32 位内（issue #308）。
    SOLVER_ECHECK((uplo == 'L' || uplo == 'U') && n > 0 && n <= 8192 && A != nullptr && lda >= n && lda <= 16384 &&
                      info != nullptr,
                  "aclsolverCpotrf invalid param: uplo != 'U'/'L', n out of [1, 8192], lda < n, lda > 16384, "
                  "or A, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        // GetStream 失败时向上报错而非静默改用默认流，与 950 Device 路径及
        // getrs/cheevj 家族的修复口径一致（issue #326）。
        const aclsolverStatus_t streamRet = aclsolverGetStream(handle, &stream);
        SOLVER_ECHECK(streamRet == ACLSOLVER_STATUS_SUCCESS, "aclsolverCpotrf: failed to get stream from handle.",
                      streamRet);
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr)
        numBlocks = ascendcPlatform->GetCoreNumAic();
    else
        return ACL_ERROR_RT_INTERNAL_ERROR;
    SOLVER_ECHECK(numBlocks > 0, "No available AIcore found on current device.", ACL_ERROR_RT_FEATURE_NOT_SUPPORT);

    int64_t N = n, padN = ((N + 127) / 128) * 128;
    size_t aMatrixWorkSize = 4 * padN * padN * sizeof(float);
    size_t gatherSize = 8192 * sizeof(uint32_t) * 2;
    int k = static_cast<int>((N + 128 - 1) / 128);
    size_t offsetSize = k * (k - 1) * (k - 2) * sizeof(int) / 6;
    if (k < 3)
    {
        offsetSize = k * sizeof(int);
    }

    CpotrfBuffers buf;
    buf.aHost = reinterpret_cast<uint8_t*>(A);

    CHECK_ACLRT(AllocAndUploadMatrix(buf, N, lda, aMatrixWorkSize, stream), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(AllocAndUploadGemmWork(buf, k, offsetSize, numBlocks, stream), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(AllocAndUploadGather(buf, gatherSize, uplo == 'L', N, stream), ReleaseCpotrfBuffers(buf));
    // LaunchCpotrf 自管失败路径的释放（含同步失败故意滞留的口径），
    // 此处不再兜底释放，避免同步失败分支的滞留语义被调用方推翻（issue #350）。
    CHECK_ACLRT(LaunchCpotrf(buf, uplo == 'L', N, lda, numBlocks, stream), ((void)0));

    ReleaseCpotrfBuffers(buf);
    return ACL_SUCCESS;
}
