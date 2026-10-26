/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <chrono>
#include <limits>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <queue>
#include <algorithm>
#include "cann_ops_solver.h"

#include <securec.h>

#include "../utils/assert.h"
#include "kernel_tiling/kernel_tiling.h"
#include "spotrf_launchers.hpp"
#include "tiling/platform/platform_ascendc.h"


#include "acl/acl.h"

#define GM_ADDR uint8_t*

using namespace std::chrono;
using namespace std;

#define debug(x) std::cerr << #x << ": " << x << std::endl
constexpr int BLOCKDIM_MAX = 20;

namespace {
void GenerateGather(int blockM, int blockN, uint8_t *gatherBuf) {
    try {
        auto buf = reinterpret_cast<uint32_t *>(gatherBuf);
        int idx = 0;
        for (int i = 0; i < blockM; ++i) //128
            for (int j = 0; j < blockN; ++j) //64
                buf[idx++] = (j * blockM + i) * 4;
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        exit(1);
    }
}
}  // namespace

// 数据结构 L，包含三个 int 变量 x, y, z
struct L {
    int x, y, z;
    // 优先队列为小顶堆：x 越小优先级越高，x 相同则 y 越小优先级越高
    // 使用 std::greater<L> 需要定义 operator>
    bool operator>(const L& other) const {
        if (x != other.x) return x > other.x;
        return y > other.y;
    }
};

void GenerateGemmWork(int n,int blockNum, uint32_t *aOffset, uint32_t *bOffset, uint32_t *cOffset, uint32_t *OffsetIdx){
    if (n < 3) {
        // 无有效操作
        return;
    }
    int idx = 0;
    const int len = n - 2;          // 数组 b 与 workload 的长度
    std::vector<int> b(len, 0);
    int total = 0;
    int blocksize = 128; 

    // 1. 计算数组 b, b[i] = ceil((n - i) / (2p))，这里 i 从 2 到 n-1
    int denominator = blockNum;
    for (int i = 1; i <= n - 2; ++i) {
        int numerator = n - i;
        // 向上取整
        int val = (numerator + denominator - 1) / denominator + 1;
        b[i - 1] = val;
        total += val;
    }

    // 2. 计算数组 workload
    std::vector<int> workload(len, 0);
    if (total > 0) {
        for (int i = 1; i <= len; ++i) {
            int num = static_cast<int>(b[i-1]) * n * (n - 1) * (n - 2);
            int denom = 6LL * total;
            workload[i-1] = static_cast<int>((num + denom - 1) / denom);
        }
    }

    // 3. 优先队列，存储 L 结构
    std::priority_queue<L, std::vector<L>, std::greater<L>> pq;

    total = 0;
    // 4. 外层循环 i = 1 .. n-2
    for (int i = 1; i <= n - 2; ++i) {//i = 1 to 14
        OffsetIdx[i-1] = 0;
        // 4.1 创建 n-i 个 L，参数为 (t, i-1)，t = i+1 .. n
        for (int t = i + 1; t <= n-1; ++t) {
            L item;
            item.x = t;
            item.y = i - 1;
            item.z = n - t;   
            pq.push(item);
            total += n-t;
        }

        int& wl = workload[i - 1];
        while (!pq.empty()) {
            L top = pq.top();
            
            if (top.x == i + 1) {
                // 若 x == i+1，workload[i] 减去 z 并删除该结构
                wl -= top.z;
                for(int ptr = 1;ptr <= top.z;++ptr){
                    aOffset[idx] = top.y * blocksize * n * blocksize + top.x * blocksize;
                    bOffset[idx] = aOffset[idx] + (top.z - ptr) * blocksize;
                    cOffset[idx] = top.x * blocksize * n * blocksize + top.x * blocksize + (top.z - ptr) * blocksize;
                    idx++;
                }
                OffsetIdx[i-1] += top.z;
                pq.pop();
            }else{
                break;
            }   
        }
        while (wl > 0 && !pq.empty()) {
            L top = pq.top();
            pq.pop();
            if(top.x <= i){
                printf("i: %d,x: %d, y: %d, z: %d\n",i,top.x,top.y,top.z);
            }

            if (top.z < wl) {
                // z < workload[i]，workload[i] 减去 z，删除结构
                wl -= top.z;
                for(int ptr = 1;ptr <= top.z;++ptr){
                    aOffset[idx] = top.y * blocksize * n * blocksize + top.x * blocksize;
                    bOffset[idx] = aOffset[idx] + (top.z - ptr) * blocksize;
                    cOffset[idx] = top.x * blocksize * n * blocksize + top.x * blocksize + (top.z - ptr) * blocksize;

                    idx++;
                }
                OffsetIdx[i-1] += top.z;
            } else {
                // z >= workload[i]，z 减去 workload[i]，workload[i] 置为 0
                for(int ptr = 1;ptr <= wl;++ptr){
                    aOffset[idx] = top.y * blocksize * n * blocksize + top.x * blocksize;
                    bOffset[idx] = aOffset[idx] + (top.z - ptr) * blocksize;
                    cOffset[idx] = top.x * blocksize * n * blocksize + top.x * blocksize + (top.z - ptr) * blocksize;

                    idx++;
                }
                top.z -= wl;
                OffsetIdx[i-1] += wl;
                wl = 0;
                if (top.z > 0) {
                    pq.push(top);   // 剩余工作量放回队列
                }
            }
        }
    }
    return;
}



aclError aclsolverSpotrf(aclsolverHandle_t handle, const char uplo, const int64_t m, const int64_t n, float *A, const int64_t lda,
                         int32_t *info) {
    SOLVER_ECHECK(n > 0 && A != nullptr && lda >= n && info != nullptr && (uplo == 'U' || uplo == 'L'),
 	                   "aclsolverSpotrf invalid param: uplo != 'U'/'L', n, lda < n, or A, info is nullptr.",
 	                   ACL_ERROR_INVALID_PARAM);
    aclrtStream stream = nullptr;
    if (handle != nullptr) {
        aclsolverGetStream(handle, &stream);
    }
    bool isTranse = false;
    if(uplo == 'L'){
        isTranse = true;
    }
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr) {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

	int M, N, blockDim, blockN, blockM;
    auto outputAFile = "../output/A_gm.bin";

    uint8_t* aMatrixHost = reinterpret_cast<uint8_t*>(A);
 	uint8_t* aMatrixDevice = nullptr;
 	uint8_t* aMatrixDeviceWork = nullptr;
 	uint32_t *offsetHost1 = nullptr, *offsetHost2 = nullptr, *offsetHost3 = nullptr;
 	uint32_t *offsetDevice1 = nullptr, *offsetDevice2 = nullptr, *offsetDevice3 = nullptr;
 	uint32_t *offsetIdxHost = nullptr;
 	uint32_t *offsetIdxDevice = nullptr;
 	uint8_t *gatherHost = nullptr;
 	uint8_t *gatherDevice = nullptr;
 	uint8_t *sync = nullptr;
    uint8_t* gatherbuf = nullptr;
    uint8_t* gatherbufDevice = nullptr;

    auto cleanup = [&]() {
        if (aMatrixDevice) aclrtFree(aMatrixDevice);
        if (offsetDevice1) aclrtFree(offsetDevice1);
        if (offsetDevice2) aclrtFree(offsetDevice2);
        if (offsetDevice3) aclrtFree(offsetDevice3);
        if (offsetHost1) aclrtFreeHost(offsetHost1);
        if (offsetHost2) aclrtFreeHost(offsetHost2);
        if (offsetHost3) aclrtFreeHost(offsetHost3);
        if (gatherDevice) aclrtFree(gatherDevice);
        if (gatherHost) aclrtFreeHost(gatherHost);
        if (offsetIdxHost) aclrtFreeHost(offsetIdxHost);
        if (offsetIdxDevice) aclrtFree(offsetIdxDevice);
        if (gatherbuf) aclrtFreeHost(gatherbuf);
        if (gatherbufDevice) aclrtFree(gatherbufDevice);
        return ACL_SUCCESS;
    };

    M = m;
    N = n;
    blockDim = numBlocks;

    blockN = 32;
    blockM = 32;
    int blocksize = 128;

    size_t aMatrixFileSize = lda * N * sizeof(float);
    size_t aMatrixWorkSize = ((N + blocksize - 1) / blocksize * blocksize) * ((lda + blocksize - 1) / blocksize * blocksize) * sizeof(float);

    CHECK_ACLRT(aclrtMalloc((void**)&aMatrixDevice, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMemcpy(aMatrixDevice, aMatrixFileSize, aMatrixHost, aMatrixFileSize, ACL_MEMCPY_HOST_TO_DEVICE), cleanup());

    
    CHECK_ACLRT(aclrtMalloc((void**)&aMatrixDeviceWork, aMatrixWorkSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());

    CHECK_ACLRT(aclrtMallocHost((void**)(&gatherbuf), 128 * 128 * sizeof(uint32_t)), cleanup());
    CHECK_ACLRT(aclrtMalloc((void**)(&gatherbufDevice), 128 * 128 * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    GenerateGather(64, 128, gatherbuf);
    CHECK_ACLRT(aclrtMemcpyAsync(gatherbufDevice, 128 * 128 * sizeof(uint32_t), gatherbuf, 128 * 128 * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());

    int k = (N + blocksize - 1) / blocksize;
    size_t offsetSize = k * (k-1) * (k-2) * sizeof(int) / 6;
    if(k < 3){
        offsetSize = k * sizeof(int);
    }

    CHECK_ACLRT(aclrtMallocHost((void**)(&offsetHost1), offsetSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void**)(&offsetHost2), offsetSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void**)(&offsetHost3), offsetSize), cleanup());
    CHECK_ACLRT(aclrtMallocHost((void**)(&offsetIdxHost), k * sizeof(int)), cleanup());
    CHECK_ACLRT(aclrtMalloc((void**)&offsetDevice1, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void**)&offsetDevice2, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void**)&offsetDevice3, offsetSize, ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    CHECK_ACLRT(aclrtMalloc((void**)&offsetIdxDevice, k * sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST), cleanup());
    SOLVER_ECHECK(memset_s(offsetHost1, offsetSize, 0, offsetSize) == EOK,
                  "spotrf memset_s offsetHost1 failed.", cleanup(); return ACL_ERROR_INTERNAL_ERROR);
    SOLVER_ECHECK(memset_s(offsetHost2, offsetSize, 0, offsetSize) == EOK,
                  "spotrf memset_s offsetHost2 failed.", cleanup(); return ACL_ERROR_INTERNAL_ERROR);
    SOLVER_ECHECK(memset_s(offsetHost3, offsetSize, 0, offsetSize) == EOK,
                  "spotrf memset_s offsetHost3 failed.", cleanup(); return ACL_ERROR_INTERNAL_ERROR);
    SOLVER_ECHECK(memset_s(offsetIdxHost, k * sizeof(int), 0, k * sizeof(int)) == EOK,
                  "spotrf memset_s offsetIdxHost failed.", cleanup(); return ACL_ERROR_INTERNAL_ERROR);
    GenerateGemmWork(k, blockDim, offsetHost1, offsetHost2, offsetHost3, offsetIdxHost);
    CHECK_ACLRT(aclrtMemcpyAsync(offsetDevice1, offsetSize, offsetHost1, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(offsetDevice2, offsetSize, offsetHost2, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(offsetDevice3, offsetSize, offsetHost3, offsetSize, ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());
    CHECK_ACLRT(aclrtMemcpyAsync(offsetIdxDevice, k * sizeof(int), offsetIdxHost, k * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE, stream), cleanup());

    CHECK_ACLRT(aclrtGetHardwareSyncAddr((void **)&sync), cleanup());

    spotrf_kernel_do(sync, M, N, blockM, blockN, lda, isTranse, aMatrixDevice, aMatrixDeviceWork, gatherbufDevice, 
                     offsetDevice1, offsetDevice2, offsetDevice3, offsetIdxDevice, blockDim, stream);
    CHECK_ACLRT(aclrtSynchronizeStream(stream), cleanup());

    CHECK_ACLRT(aclrtMemcpy(aMatrixHost, aMatrixFileSize, aMatrixDevice, aMatrixFileSize, ACL_MEMCPY_DEVICE_TO_HOST), cleanup());

    cleanup();
    return ACL_SUCCESS;
}
