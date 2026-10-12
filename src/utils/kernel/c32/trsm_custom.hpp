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
 * \file trsm_custom.hpp
 * \brief
 */

#ifndef _TRSM_CUSTOM_HPP_
#define _TRSM_CUSTOM_HPP_

// 自带 LU_CEIL_ALIGN 定义（issue #178）：本文件使用了该宏但此前依赖包含者先包含
// lu_custom.hpp 才可编译；#ifndef 守卫使其可与其他 5 处定义安全共存
#ifndef LU_CEIL_ALIGN
#define LU_CEIL_ALIGN(x, y) (((x) + (y) - 1) / (y) * (y))
#endif

#include <lib/matrix/matmul/matmul.h>

#include <cassert>
#include <cstdint>

#include "gemm.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

#define MAX(a, b) ((a) >= (b) ? (a) : (b))
#define MIN(a, b) ((a) <= (b) ? (a) : (b))

template <typename T>
class SolveTrsmCustom
{
    GlobalTensor<T> aGlobal, lGlobal, xGlobal;
    TQue<QuePosition::VECIN, 1> triangularQueue;
    TQue<QuePosition::VECIN, 1> MatAQueue;
    TQue<QuePosition::VECOUT, 1> augQueue;
    uint64_t elementsPerBlock;
    int M;
    int resized_M;
    int N;
    int block_m;
    int block_n;

   public:
    __aicore__ inline SolveTrsmCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> &aGlobal,
                                GlobalTensor<T> &lGlobal, GlobalTensor<T> &xGlobal, int M, int resized_M, int N,
                                int block_m, int block_n)
    {
        this->M = M;
        this->resized_M = resized_M;
        this->N = N;
        this->block_m = block_m;
        this->block_n = block_n;

        this->aGlobal = aGlobal;
        this->lGlobal = lGlobal;
        this->xGlobal = xGlobal;

        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(triangularQueue, 1, block_m * block_m * sizeof(T));
        pipe->InitBuffer(MatAQueue, 1, block_m * block_n * sizeof(T));
        pipe->InitBuffer(augQueue, 1, block_m * block_n * sizeof(T));
    }
    __aicore__ inline void Process(int offset_m)
    {
        uint64_t matA_block_offset = offset_m * N;
        uint64_t matL_block_offset = offset_m * resized_M + offset_m;

        // CopyIn
        LocalTensor<T> triangularLocal = triangularQueue.AllocTensor<T>();
        LocalTensor<T> matALocal = MatAQueue.AllocTensor<T>();

        // copy L
        DataCopyParams copyParamsMatL{static_cast<uint16_t>(block_m), static_cast<uint16_t>(block_m / elementsPerBlock),
                                      static_cast<uint16_t>((resized_M - block_m) / elementsPerBlock), 0};
        DataCopy(triangularLocal, lGlobal[matL_block_offset], copyParamsMatL);

        // copy A
        DataCopyParams copyParamsMatA{static_cast<uint16_t>(block_m), static_cast<uint16_t>(block_n / elementsPerBlock),
                                      static_cast<uint16_t>((N - block_n) / elementsPerBlock), 0};
        DataCopy(matALocal, aGlobal[matA_block_offset], copyParamsMatA);

        triangularQueue.EnQue(triangularLocal);
        MatAQueue.EnQue(matALocal);
        LocalTensor<T> triangularLocal_1 = triangularQueue.DeQue<T>();
        LocalTensor<T> matALocal_1 = MatAQueue.DeQue<T>();
        LocalTensor<T> augLocal_1 = augQueue.AllocTensor<T>();

        for (uint32_t rowIdx = 0; rowIdx < block_m; ++rowIdx)
        {
            for (uint32_t tempRowIdx = 0; tempRowIdx < rowIdx; ++tempRowIdx)
            {
                T colScalar = triangularLocal_1.GetValue(rowIdx * block_m + tempRowIdx);

                Axpy(matALocal_1[rowIdx * block_n], augLocal_1[tempRowIdx * block_n], colScalar, block_n);
                PipeBarrier<PIPE_V>();
            }
            Muls(augLocal_1[block_n * rowIdx], matALocal_1[block_n * rowIdx], T(-1), block_n);
        }

        augQueue.EnQue(augLocal_1);
        triangularQueue.FreeTensor(triangularLocal_1);
        MatAQueue.FreeTensor(matALocal_1);

        // CopyOut
        LocalTensor<T> retLocal = augQueue.DeQue<T>();

        DataCopyParams copyParamsMatOut{static_cast<uint16_t>(block_m),
                                        static_cast<uint16_t>(block_n / elementsPerBlock), 0,
                                        static_cast<uint16_t>((N - block_n) / elementsPerBlock)};
        DataCopy(xGlobal[matA_block_offset], retLocal, copyParamsMatOut);
        augQueue.FreeTensor(retLocal);
    }
};

// Assert 16 | M, p * 16 | N
__aicore__ inline void custom_trsm(TBufPool<TPosition::VECCALC, 16> &tbufPool, MatmulCustom<float> &opmm, int M, int N,
                                   int strideN, int block_m, GM_ADDR A, GM_ADDR L)
{
    int baseM = 128;
    int lim = 512;
    int big_block_count = lim / baseM;
    lim /= block_m;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int cnt = M / block_m;
    int coreIdx = blockIdx;
#ifdef __DAV_C220_VEC__
    coreIdx >>= 1;
#endif

    GlobalTensor<float> aGlobal;
    GlobalTensor<float> lGlobal;

    lGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(L));

    int block_n = LU_CEIL_ALIGN((N + blockNum - 1) / blockNum, 64);
    int trsmOffsetN = block_n * coreIdx;
    int realBlockN = min(N - trsmOffsetN, block_n);

    if (realBlockN <= 0) return;

#ifdef __DAV_C220_VEC__
    blockNum <<= 1;

    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A + trsmOffsetN * sizeof(float)));

    int doubleAIV = realBlockN > 64;
    SolveTrsmCustom<float> optrsm;
    auto trsmAGlobal = aGlobal[(blockIdx & 1) * (realBlockN / 2)];
    if (doubleAIV)
        optrsm.Init(&tbufPool, trsmAGlobal, lGlobal, trsmAGlobal, M, strideN, strideN, block_m, realBlockN / 2);
    else if (~blockIdx & 1)
        optrsm.Init(&tbufPool, trsmAGlobal, lGlobal, trsmAGlobal, M, strideN, strideN, block_m, realBlockN);
#endif

#ifdef __DAV_C220_CUBE__
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A + trsmOffsetN * sizeof(float)));
    opmm.SetMatrix(lGlobal, aGlobal, aGlobal, strideN, strideN);
    SetAtomicAdd<float>();
#endif
#ifdef __DAV_C220_VEC__
    if (blockIdx % 2 <= doubleAIV) optrsm.Process(0);
    if (cnt > 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
#endif

    int big_block_m = 0;
    int big_block_slice_m;
    int big_block_offsetA, big_block_offsetB, big_block_offsetC;
    for (int i = 1, offset_m = block_m, big_block_offset_m = 0; i < cnt; ++i, offset_m += block_m)
    {
#ifdef __DAV_C220_CUBE__
        int r = i % lim;
        if (r)
        {
            int offsetA = offset_m * strideN + offset_m - block_m;
            int offsetB = (offset_m - block_m) * strideN;
            int offsetC = offset_m * strideN;
            CrossCoreWaitFlag(0x0);
            opmm.Process(offsetA, offsetB, offsetC, block_m, realBlockN, block_m);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);

            if (r == lim - 1) continue;
            if (offset_m + block_m >= M) continue;
            offsetA = (offset_m + block_m) * strideN + big_block_offset_m;
            offsetB = big_block_offset_m * strideN;
            offsetC += block_m * strideN;
            opmm.Process(offsetA, offsetB, offsetC, block_m, realBlockN, block_m * r);

            if (big_block_m == 0 || r >= big_block_count - 1) continue;
            big_block_offsetA += baseM * strideN;
            big_block_offsetC += baseM * strideN;
            int big_block_tmp = MIN(M - big_block_offsetA / strideN, baseM);
            if (big_block_tmp <= 0) continue;
            opmm.Process(big_block_offsetA, big_block_offsetB, big_block_offsetC, big_block_tmp, realBlockN,
                         big_block_m);
        }
        else
        {
            big_block_m = (i & -i) * block_m;
            big_block_count = big_block_m / baseM;
            big_block_offset_m += lim * block_m;
            int L_offset_n = offset_m - big_block_m;
            big_block_offsetA = offset_m * strideN + L_offset_n;
            big_block_offsetB = L_offset_n * strideN;
            big_block_offsetC = offset_m * strideN;
            CrossCoreWaitFlag(0x0);
            opmm.Process(big_block_offsetA, big_block_offsetB, big_block_offsetC,
                         MIN(M - big_block_offsetA / strideN, baseM), realBlockN, big_block_m);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);
            big_block_offsetA += baseM * strideN;
            big_block_offsetC += baseM * strideN;
            int big_block_tmp = MIN(M - big_block_offsetA / strideN, baseM);
            if (big_block_tmp <= 0) continue;
            opmm.Process(big_block_offsetA, big_block_offsetB, big_block_offsetC, big_block_tmp, realBlockN,
                         big_block_m);
        }
#elif __DAV_C220_VEC__
        CrossCoreWaitFlag(0x1);
        if (blockIdx % 2 <= doubleAIV) optrsm.Process(offset_m);
        if (i != cnt - 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
#endif
    }
#ifdef __DAV_C220_VEC__
    tbufPool.Reset();
#endif
}

#endif  // _TRSM_CUSTOM_HPP_
