/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <vector>

#include "../utils/assert.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"
#include "kernel_tiling/kernel_tiling.h"
#include "spotrf_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"

#define GM_ADDR uint8_t *

using namespace std::chrono;
using namespace std;

#define debug(x) std::cerr << #x << ": " << x << std::endl
constexpr int BLOCKDIM_MAX = 20;

namespace
{
void GenerateGather(int blockM, int blockN, uint8_t *gatherBuf)
{
    try
    {
        auto buf = reinterpret_cast<uint32_t *>(gatherBuf);
        int idx = 0;
        for (int i = 0; i < blockM; ++i)      // 128
            for (int j = 0; j < blockN; ++j)  // 64
                buf[idx++] = (j * blockM + i) * 4;
    }
    catch (const std::exception &e)
    {
        std::cout << e.what() << std::endl;
        exit(1);
    }
}
}  // namespace

// 数据结构 L，包含三个 int 变量 x, y, z
struct L
{
    int x, y, z;
    // 优先队列为小顶堆：x 越小优先级越高，x 相同则 y 越小优先级越高
    // 使用 std::greater<L> 需要定义 operator>
    bool operator>(const L &other) const
    {
        if (x != other.x) return x > other.x;
        return y > other.y;
    }
};

// 计算各对角块行 i 的 GEMM 配额 b[i] 与归一化工作量 workload[i]（原步骤 1/2 提取）
static void FillWorkQuota(int n, int blockNum, std::vector<int> &b, std::vector<int> &workload)
{
    const int len = n - 2;
    b.assign(len, 0);
    workload.assign(len, 0);
    int total = 0;
    int denominator = blockNum;
    // b[i] = ceil((n - i) / (2p))，这里 i 从 2 到 n-1
    for (int i = 1; i <= n - 2; ++i)
    {
        int numerator = n - i;
        // 向上取整
        int val = (numerator + denominator - 1) / denominator + 1;
        b[i - 1] = val;
        total += val;
    }
    if (total <= 0)
    {
        return;
    }
    for (int i = 1; i <= len; ++i)
    {
        int num = static_cast<int>(b[i - 1]) * n * (n - 1) * (n - 2);
        int denom = 6LL * total;
        workload[i - 1] = static_cast<int>((num + denom - 1) / denom);
    }
}

// 写入一条 GEMM 分块的 a/b/c 偏移表；原三处逐行重复循环的合并体（CodeCheck 重复代码整改）
static void EmitGemmTile(uint32_t *aOffset, uint32_t *bOffset, uint32_t *cOffset, int &idx, int n, int blocksize, int y,
                         int x, int z, int count)
{
    for (int ptr = 1; ptr <= count; ++ptr)
    {
        aOffset[idx] = y * blocksize * n * blocksize + x * blocksize;
        bOffset[idx] = aOffset[idx] + (z - ptr) * blocksize;
        cOffset[idx] = x * blocksize * n * blocksize + x * blocksize + (z - ptr) * blocksize;
        idx++;
    }
}

// 消费优先队列：按 workload 配额切分 GEMM 分块并写偏移表（原两个 while 提取）
static void DrainPendingTiles(int i, int &wl, std::priority_queue<L, std::vector<L>, std::greater<L>> &pq,
                              uint32_t *aOffset, uint32_t *bOffset, uint32_t *cOffset, int &idx, int n, int blocksize,
                              uint32_t *OffsetIdx)
{
    while (!pq.empty())
    {
        L top = pq.top();
        if (top.x != i + 1)
        {
            break;
        }
        // 若 x == i+1，workload[i] 减去 z 并删除该结构
        wl -= top.z;
        EmitGemmTile(aOffset, bOffset, cOffset, idx, n, blocksize, top.y, top.x, top.z, top.z);
        OffsetIdx[i - 1] += top.z;
        pq.pop();
    }
    while (wl > 0 && !pq.empty())
    {
        L top = pq.top();
        pq.pop();
        if (top.x <= i)
        {
            printf("i: %d,x: %d, y: %d, z: %d\n", i, top.x, top.y, top.z);
        }
        if (top.z < wl)
        {
            // z < workload[i]，workload[i] 减去 z，删除结构
            wl -= top.z;
            EmitGemmTile(aOffset, bOffset, cOffset, idx, n, blocksize, top.y, top.x, top.z, top.z);
            OffsetIdx[i - 1] += top.z;
        }
        else
        {
            // z >= workload[i]，z 减去 workload[i]，workload[i] 置为 0
            EmitGemmTile(aOffset, bOffset, cOffset, idx, n, blocksize, top.y, top.x, top.z, wl);
            top.z -= wl;
            OffsetIdx[i - 1] += wl;
            wl = 0;
            if (top.z > 0)
            {
                pq.push(top);  // 剩余工作量放回队列
            }
        }
    }
}

void GenerateGemmWork(int n, int blockNum, uint32_t *aOffset, uint32_t *bOffset, uint32_t *cOffset, uint32_t *OffsetIdx)
{
    if (n < 3)
    {
        // 无有效操作
        return;
    }
    int idx = 0;
    int blocksize = 128;

    // 1/2. 计算各对角块行的 GEMM 配额 b[i] 与归一化工作量 workload[i]
    std::vector<int> b;
    std::vector<int> workload;
    FillWorkQuota(n, blockNum, b, workload);

    // 3. 优先队列，存储 L 结构
    std::priority_queue<L, std::vector<L>, std::greater<L>> pq;

    // 4. 外层循环 i = 1 .. n-2
    for (int i = 1; i <= n - 2; ++i)
    {
        OffsetIdx[i - 1] = 0;
        // 4.1 创建 n-i 个 L，参数为 (t, i-1)，t = i+1 .. n
        for (int t = i + 1; t <= n - 1; ++t)
        {
            L item;
            item.x = t;
            item.y = i - 1;
            item.z = n - t;
            pq.push(item);
        }
        DrainPendingTiles(i, workload[i - 1], pq, aOffset, bOffset, cOffset, idx, n, blocksize, OffsetIdx);
    }
}

// spotrf 全部 host/device 缓冲；未申请的指针为空，ReleaseSpotrfBuffers 可安全跳过
struct SpotrfBuffers
{
    uint8_t *aHost = nullptr;        // 用户输入矩阵（host 侧首地址）
    uint8_t *aDevice = nullptr;      // 上载后的输入矩阵
    uint8_t *aDeviceWork = nullptr;  // kernel 工作区
    uint32_t *offsetHost1 = nullptr;
    uint32_t *offsetHost2 = nullptr;
    uint32_t *offsetHost3 = nullptr;
    uint32_t *offsetDevice1 = nullptr;
    uint32_t *offsetDevice2 = nullptr;
    uint32_t *offsetDevice3 = nullptr;
    uint32_t *offsetIdxHost = nullptr;
    uint32_t *offsetIdxDevice = nullptr;
    uint8_t *gatherbuf = nullptr;        // gather 索引表（host）
    uint8_t *gatherbufDevice = nullptr;  // gather 索引表（device）
    uint8_t *sync = nullptr;             // 硬件同步地址
};

// 释放所有已申请的 HBM / host 内存（原 cleanup lambda 提取；补充了原实现遗漏的 aDeviceWork 释放）
static void ReleaseSpotrfBuffers(SpotrfBuffers &buf)
{
    if (buf.aDevice) aclrtFree(buf.aDevice);
    if (buf.aDeviceWork) aclrtFree(buf.aDeviceWork);
    if (buf.offsetDevice1) aclrtFree(buf.offsetDevice1);
    if (buf.offsetDevice2) aclrtFree(buf.offsetDevice2);
    if (buf.offsetDevice3) aclrtFree(buf.offsetDevice3);
    if (buf.offsetHost1) aclrtFreeHost(buf.offsetHost1);
    if (buf.offsetHost2) aclrtFreeHost(buf.offsetHost2);
    if (buf.offsetHost3) aclrtFreeHost(buf.offsetHost3);
    if (buf.offsetIdxHost) aclrtFreeHost(buf.offsetIdxHost);
    if (buf.offsetIdxDevice) aclrtFree(buf.offsetIdxDevice);
    if (buf.gatherbuf) aclrtFreeHost(buf.gatherbuf);
    if (buf.gatherbufDevice) aclrtFree(buf.gatherbufDevice);
}

// A 矩阵上载：host → device，另申请等尺寸 kernel 工作区
static aclError AllocAndUploadMatrix(SpotrfBuffers &buf, int n, int lda, size_t workSize)
{
    size_t aMatrixFileSize = lda * n * sizeof(float);
    CHECK_ACLRT(aclrtMalloc((void **)&buf.aDevice, workSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy(buf.aDevice, aMatrixFileSize, buf.aHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE),
                ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.aDeviceWork, workSize, ACL_MEM_MALLOC_HUGE_FIRST), ReleaseSpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// gather 索引表：host 侧生成后整块搬到 device
static aclError AllocAndUploadGather(SpotrfBuffers &buf, aclrtStream stream)
{
    CHECK_ACLRT(aclrtMallocHost((void **)(&buf.gatherbuf), 128 * 128 * sizeof(uint32_t)), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)(&buf.gatherbufDevice), 128 * 128 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSpotrfBuffers(buf));
    GenerateGather(64, 128, buf.gatherbuf);
    CHECK_ACLRT(aclrtMemcpyAsync(buf.gatherbufDevice, 128 * 128 * sizeof(uint32_t), buf.gatherbuf,
                                 128 * 128 * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseSpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// GEMM 发单表：host 侧生成三组偏移与索引表后搬到 device
static aclError AllocAndUploadGemmWork(SpotrfBuffers &buf, int k, size_t offsetSize, int blockNum, aclrtStream stream)
{
    CHECK_ACLRT(aclrtMallocHost((void **)(&buf.offsetHost1), offsetSize), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void **)(&buf.offsetHost2), offsetSize), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void **)(&buf.offsetHost3), offsetSize), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMallocHost((void **)(&buf.offsetIdxHost), k * sizeof(int)), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.offsetDevice1, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.offsetDevice2, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.offsetDevice3, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void **)&buf.offsetIdxDevice, k * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseSpotrfBuffers(buf));
    bool memsetOk = memset_s(buf.offsetHost1, offsetSize, 0, offsetSize) == EOK &&
                    memset_s(buf.offsetHost2, offsetSize, 0, offsetSize) == EOK &&
                    memset_s(buf.offsetHost3, offsetSize, 0, offsetSize) == EOK &&
                    memset_s(buf.offsetIdxHost, k * sizeof(int), 0, k * sizeof(int)) == EOK;
    if (!memsetOk)
    {
        std::cerr << "[ERROR] spotrf memset host offset buffers failed." << std::endl;
        ReleaseSpotrfBuffers(buf);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    GenerateGemmWork(k, blockNum, buf.offsetHost1, buf.offsetHost2, buf.offsetHost3, buf.offsetIdxHost);
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.offsetDevice1, offsetSize, buf.offsetHost1, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.offsetDevice2, offsetSize, buf.offsetHost2, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(
        aclrtMemcpyAsync(buf.offsetDevice3, offsetSize, buf.offsetHost3, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream),
        ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpyAsync(buf.offsetIdxDevice, k * sizeof(int), buf.offsetIdxHost, k * sizeof(int),
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseSpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// 下发 kernel，等待完成后把结果搬回 host
static aclError LaunchSpotrf(SpotrfBuffers &buf, int m, int n, int blockM, int blockN, int lda, bool isTranse,
                             int blockNum, aclrtStream stream)
{
    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&buf.sync), ReleaseSpotrfBuffers(buf));
    spotrf_kernel_do(buf.sync, m, n, blockM, blockN, lda, isTranse, buf.aDevice, buf.aDeviceWork, buf.gatherbufDevice,
                     buf.offsetDevice1, buf.offsetDevice2, buf.offsetDevice3, buf.offsetIdxDevice, blockNum, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ReleaseSpotrfBuffers(buf));
    size_t aMatrixFileSize = lda * n * sizeof(float);
    CHECK_ACLRT(aclrtMemcpy(buf.aHost, aMatrixFileSize, buf.aDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST),
                ReleaseSpotrfBuffers(buf));
    return ACL_SUCCESS;
}

aclError aclsolverSpotrf(aclsolverHandle_t handle, const char uplo, const int64_t m, const int64_t n, float *A,
                         const int64_t lda, int32_t *info)
{
    // Cholesky 分解要求 A 为 n 阶方阵，m 必须等于 n（issue #267）；n 上界与 aclsolverSgetrf
    // 的文档口径（valid range [1, 8192]）保持一致。
    // kernel 内以 uint16_t 承载 DataCopyExtParams 行间隙字节数。uplo == 'U' 时成员列数逐轮
    // 递减，行间隙上界出现在成员列收敛到 1 时，即 (lda - 1) * 4 字节 <= 65535，故约束
    // lda 本身 <= 16384，而非 lda - n（issue #266 检视），超出时 srcGap 会回绕。
    SOLVER_ECHECK(m == n && n > 0 && n <= 8192 && A != nullptr && lda >= n && lda <= 16384 && info != nullptr &&
                      (uplo == 'U' || uplo == 'L'),
                  "aclsolverSpotrf invalid param: m != n, uplo != 'U'/'L', n out of [1, 8192], lda < n, "
                  "lda > 16384, or A, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
    }
    bool isTranse = false;
    if (uplo == 'L')
    {
        isTranse = true;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr)
    {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    int N = static_cast<int>(n);
    int blockN = 32;
    int blockM = 32;
    int blocksize = 128;
    int k = (N + blocksize - 1) / blocksize;
    size_t offsetSize = k * (k - 1) * (k - 2) * sizeof(int) / 6;
    if (k < 3)
    {
        offsetSize = k * sizeof(int);
    }
    size_t aMatrixWorkSize =
        ((N + blocksize - 1) / blocksize * blocksize) * ((lda + blocksize - 1) / blocksize * blocksize) * sizeof(float);

    SpotrfBuffers buf;
    buf.aHost = reinterpret_cast<uint8_t *>(A);
    CHECK_ACLRT(AllocAndUploadMatrix(buf, N, static_cast<int>(lda), aMatrixWorkSize), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(AllocAndUploadGather(buf, stream), ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(AllocAndUploadGemmWork(buf, k, offsetSize, static_cast<int>(numBlocks), stream),
                ReleaseSpotrfBuffers(buf));
    CHECK_ACLRT(LaunchSpotrf(buf, static_cast<int>(m), N, blockM, blockN, static_cast<int>(lda), isTranse,
                             static_cast<int>(numBlocks), stream),
                ReleaseSpotrfBuffers(buf));

    ReleaseSpotrfBuffers(buf);
    return ACL_SUCCESS;
}
