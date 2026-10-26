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
#include "../utils/gm_addr.h"
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

struct L
{
    int x, y, z;
    bool operator>(const L& other) const
    {
        if (x != other.x) return x > other.x;
        return y > other.y;
    }
};

void GenerateGemmWork(int n, int blockNum, uint8_t* _aOffset, uint8_t* _bOffset, uint8_t* _cOffset, uint8_t* _OffsetIdx)
{
    auto aOffset = reinterpret_cast<uint32_t*>(_aOffset);
    auto bOffset = reinterpret_cast<uint32_t*>(_bOffset);
    auto cOffset = reinterpret_cast<uint32_t*>(_cOffset);
    auto OffsetIdx = reinterpret_cast<uint32_t*>(_OffsetIdx);

    if (n < 3)
    {
        return;
    }
    int idx = 0;
    const int len = n - 2;
    std::vector<int> b(len, 0);
    int total = 0;

    int denominator = blockNum;
    for (int i = 1; i <= n - 2; ++i)
    {
        int numerator = n - i;
        int val = (numerator + denominator - 1) / denominator;
        b[i - 1] = val;
        total += val;
    }

    std::vector<int> workload(len, 0);
    if (total > 0)
    {
        for (int i = 1; i <= len; ++i)
        {
            int64_t num = static_cast<int64_t>(b[i - 1]) * n * (n - 1) * (n - 2);
            int denom = 6LL * total;
            workload[i - 1] = static_cast<int>((num + denom - 1) / denom);
        }
    }

    std::priority_queue<L, std::vector<L>, std::greater<L>> pq;

    total = 0;
    for (int i = 1; i <= n - 2; ++i)
    {
        OffsetIdx[i - 1] = 0;
        for (int t = i + 1; t <= n - 1; ++t)
        {
            L item;
            item.x = t;
            item.y = i - 1;
            item.z = n - t;
            pq.push(item);
            total += n - t;
        }

        int& wl = workload[i - 1];
        while (!pq.empty())
        {
            L top = pq.top();

            if (top.x == i + 1)
            {
                wl -= top.z;
                for (int ptr = 1; ptr <= top.z; ++ptr)
                {
                    aOffset[idx] = top.y * 128 * n * 128 + top.x * 128;
                    bOffset[idx] = aOffset[idx] + (top.z - ptr) * 128;
                    cOffset[idx] = top.x * 128 * n * 128 + top.x * 128 + (top.z - ptr) * 128;
                    idx++;
                }
                OffsetIdx[i - 1] += top.z;
                pq.pop();
            }
            else
            {
                break;
            }
        }
        while (wl > 0 && !pq.empty())
        {
            L top = pq.top();
            pq.pop();

            if (top.z < wl)
            {
                wl -= top.z;
                for (int ptr = 1; ptr <= top.z; ++ptr)
                {
                    aOffset[idx] = top.y * 128 * n * 128 + top.x * 128;
                    bOffset[idx] = aOffset[idx] + (top.z - ptr) * 128;
                    cOffset[idx] = top.x * 128 * n * 128 + top.x * 128 + (top.z - ptr) * 128;
                    idx++;
                }
                OffsetIdx[i - 1] += top.z;
            }
            else
            {
                for (int ptr = 1; ptr <= wl; ++ptr)
                {
                    aOffset[idx] = top.y * 128 * n * 128 + top.x * 128;
                    bOffset[idx] = aOffset[idx] + (top.z - ptr) * 128;
                    cOffset[idx] = top.x * 128 * n * 128 + top.x * 128 + (top.z - ptr) * 128;
                    idx++;
                }
                top.z -= wl;
                OffsetIdx[i - 1] += wl;
                wl = 0;
                if (top.z > 0)
                {
                    pq.push(top);
                }
            }
        }
    }
    return;
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

// 释放所有已申请的 HBM / host 内存；未申请的指针为空，可安全跳过
void ReleaseCpotrfBuffers(CpotrfBuffers& buf)
{
    if (buf.offsetDevice1) aclrtFree(buf.offsetDevice1);
    if (buf.offsetDevice2) aclrtFree(buf.offsetDevice2);
    if (buf.offsetDevice3) aclrtFree(buf.offsetDevice3);
    if (buf.offsetIdxDevice) aclrtFree(buf.offsetIdxDevice);
    if (buf.offsetHost1) aclrtFreeHost(buf.offsetHost1);
    if (buf.offsetHost2) aclrtFreeHost(buf.offsetHost2);
    if (buf.offsetHost3) aclrtFreeHost(buf.offsetHost3);
    if (buf.offsetIdxHost) aclrtFreeHost(buf.offsetIdxHost);
    if (buf.gatherDevice) aclrtFree(buf.gatherDevice);
    if (buf.gatherHost) aclrtFreeHost(buf.gatherHost);
    if (buf.aDevice) aclrtFree(buf.aDevice);
    if (buf.aDeviceWork) aclrtFree(buf.aDeviceWork);
}

// A 矩阵上载：HBM 侧按 2*N 个 float 行宽，host 侧按 2*lda 个 float 行宽
aclError AllocAndUploadMatrix(CpotrfBuffers& buf, int64_t N, int64_t lda, size_t workSize, aclrtStream stream)
{
    CHECK_ACLRT(aclrtMalloc((void**)&buf.aDevice, 2 * N * N * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy2dAsync(buf.aDevice, 2 * N * sizeof(float), buf.aHost, 2 * lda * sizeof(float),
                                   2 * N * sizeof(float), N, ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMalloc((void**)&buf.aDeviceWork, workSize, ACL_MEM_MALLOC_HUGE_FIRST),
                ReleaseCpotrfBuffers(buf));
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
    GenerateGemmWork(k, blockDim, buf.offsetHost1, buf.offsetHost2, buf.offsetHost3, buf.offsetIdxHost);
    CHECK_ACLRT(aclrtMemcpyAsync(buf.offsetDevice1, offsetSize, buf.offsetHost1, offsetSize,
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpyAsync(buf.offsetDevice2, offsetSize, buf.offsetHost2, offsetSize,
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpyAsync(buf.offsetDevice3, offsetSize, buf.offsetHost3, offsetSize,
                                 ACL_MEMCPY_HOST_TO_DEVICE, stream),
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
    CHECK_ACLRT(aclrtMemcpyAsync(buf.gatherDevice, gatherSize, buf.gatherHost, gatherSize, ACL_MEMCPY_HOST_TO_DEVICE,
                                 stream),
                ReleaseCpotrfBuffers(buf));
    return ACL_SUCCESS;
}

// 下发 kernel，等待完成后把结果按 lda 行宽搬回 host
aclError LaunchCpotrf(CpotrfBuffers& buf, bool transpose, int64_t N, int64_t lda, uint32_t blockDim,
                      aclrtStream stream)
{
    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void**)(&buf.sync)), ReleaseCpotrfBuffers(buf));
    cpotrf_kernel_do(buf.sync, transpose, static_cast<int>(N), static_cast<int>(N), buf.aDevice, buf.aDeviceWork,
                     buf.offsetDevice1, buf.offsetDevice2, buf.offsetDevice3, buf.offsetIdxDevice, buf.gatherDevice,
                     blockDim, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtMemcpy2dAsync(buf.aHost, 2 * lda * sizeof(float), buf.aDevice, 2 * N * sizeof(float),
                                   2 * N * sizeof(float), N, ACL_MEMCPY_DEVICE_TO_HOST, stream),
                ReleaseCpotrfBuffers(buf));
    CHECK_ACLRT(aclrtSynchronizeStream(stream), ReleaseCpotrfBuffers(buf));
    return ACL_SUCCESS;
}

aclError aclsolverCpotrf(aclsolverHandle_t handle, const char uplo, const int64_t n, std::complex<float>* A,
                         const int64_t lda, int32_t* info)
{
    SOLVER_ECHECK((uplo == 'L' || uplo == 'U') && n > 0 && A != nullptr && lda >= n && info != nullptr,
                  "aclsolverCpotrf invalid param: uplo != 'U'/'L', n, lda < n, or A, info is nullptr.",
                  ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= INT32_MAX, "aclsolverCpotrf invalid param: n exceeds int32 range.", ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr)
    {
        aclsolverGetStream(handle, &stream);
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
    CHECK_ACLRT(LaunchCpotrf(buf, uplo == 'L', N, lda, numBlocks, stream), ReleaseCpotrfBuffers(buf));

    ReleaseCpotrfBuffers(buf);
    return ACL_SUCCESS;
}
