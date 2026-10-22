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
    GlobalTensor<T> aRealGlobal, lRealGlobal;
    GlobalTensor<T> aImagGlobal, lImagGlobal;
    GlobalTensor<T> aImagNegGlobal;
    TQue<QuePosition::VECIN, 1> triangularQueueReal, triangularQueueImag;
    TQue<QuePosition::VECIN, 1> MatAQueueReal, MatAQueueImag;
    TQue<QuePosition::VECOUT, 1> augQueueReal, augQueueImag, augQueueImagNeg;
    TBuf<TPosition::VECCALC> calcBuf;
    LocalTensor<T> calcLocal;
    static constexpr int maxN = 256;
    uint64_t elementsPerBlock;
    int M;
    int N;
    int blockM;
    int blockN;

   public:
    __aicore__ inline SolveTrsmCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, GlobalTensor<T> &aRealGlobal,
                                GlobalTensor<T> &aImagGlobal, GlobalTensor<T> &aImagNegGlobal,
                                GlobalTensor<T> &lRealGlobal, GlobalTensor<T> &lImagGlobal, int M, int N, int blockM,
                                int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockM = blockM;
        this->blockN = blockN;

        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->aImagNegGlobal = aImagNegGlobal;
        this->lRealGlobal = lRealGlobal;
        this->lImagGlobal = lImagGlobal;

        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(triangularQueueReal, 1, blockM * blockM * sizeof(T));
        pipe->InitBuffer(triangularQueueImag, 1, blockM * blockM * sizeof(T));
        pipe->InitBuffer(MatAQueueReal, 1, blockM * maxN * sizeof(T));
        pipe->InitBuffer(MatAQueueImag, 1, blockM * maxN * sizeof(T));
        pipe->InitBuffer(augQueueReal, 1, blockM * maxN * sizeof(T));
        pipe->InitBuffer(augQueueImag, 1, blockM * maxN * sizeof(T));
        pipe->InitBuffer(augQueueImagNeg, 1, blockM * maxN * sizeof(T));

        pipe->InitBuffer(calcBuf, blockM * 32);

        calcLocal = calcBuf.Get<T>();
    }
    __aicore__ inline void Process(int offset_m)
    {
        uint64_t matABlockOffset = offset_m * N;
        uint64_t matLBlockOffset = offset_m * M + offset_m;
        LocalTensor<T> triangularLocalReal = triangularQueueReal.AllocTensor<T>();
        LocalTensor<T> triangularLocalImag = triangularQueueImag.AllocTensor<T>();
        LocalTensor<T> matALocalReal = MatAQueueReal.AllocTensor<T>();
        LocalTensor<T> matALocalImag = MatAQueueImag.AllocTensor<T>();

        DataCopyParams copyParamsMatL{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockM / elementsPerBlock),
                                      static_cast<uint16_t>((M - blockM) / elementsPerBlock), 0};
        DataCopy(triangularLocalReal, lRealGlobal[matLBlockOffset], copyParamsMatL);
        DataCopy(triangularLocalImag, lImagGlobal[matLBlockOffset], copyParamsMatL);

        DataCopyParams copyParamsMatA{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                      static_cast<uint16_t>((N - blockN) / elementsPerBlock), 0};
        DataCopy(matALocalReal, aRealGlobal[matABlockOffset], copyParamsMatA);
        DataCopy(matALocalImag, aImagGlobal[matABlockOffset], copyParamsMatA);

        triangularQueueReal.EnQue(triangularLocalReal);
        triangularQueueImag.EnQue(triangularLocalImag);
        MatAQueueReal.EnQue(matALocalReal);
        MatAQueueImag.EnQue(matALocalImag);

        triangularLocalReal = triangularQueueReal.DeQue<T>();
        triangularLocalImag = triangularQueueImag.DeQue<T>();
        matALocalReal = MatAQueueReal.DeQue<T>();
        matALocalImag = MatAQueueImag.DeQue<T>();
        LocalTensor<T> augLocalReal = augQueueReal.AllocTensor<T>();
        LocalTensor<T> augLocalImag = augQueueImag.AllocTensor<T>();
        LocalTensor<T> augLocalImagNeg = augQueueImagNeg.AllocTensor<T>();

        PipeBarrier<PIPE_ALL>();

        for (int rowIdx = 0; rowIdx < blockM; ++rowIdx)
        {
            int offset = blockN * rowIdx;
            for (int tempRowIdx = 0; tempRowIdx < rowIdx; ++tempRowIdx)
            {
                int offsetS = blockM * rowIdx + tempRowIdx;
                int offsetT = blockN * tempRowIdx;
                T sReal = triangularLocalReal.GetValue(offsetS);
                T sImag = triangularLocalImag.GetValue(offsetS);
                PipeBarrier<PIPE_ALL>();
                Axpy(matALocalReal[offset], augLocalReal[offsetT], sReal, blockN);
                PipeBarrier<PIPE_V>();
                Axpy(matALocalReal[offset], augLocalImag[offsetT], sImag, blockN);
                PipeBarrier<PIPE_V>();
                Axpy(matALocalImag[offset], augLocalReal[offsetT], sImag, blockN);
                PipeBarrier<PIPE_V>();
                Axpy(matALocalImag[offset], augLocalImagNeg[offsetT], sReal, blockN);
                PipeBarrier<PIPE_V>();
            }
            Muls(augLocalReal[offset], matALocalReal[offset], T(-1), blockN);
            PipeBarrier<PIPE_V>();
            Muls(augLocalImag[offset], matALocalImag[offset], T(1), blockN);
            PipeBarrier<PIPE_V>();
            Muls(augLocalImagNeg[offset], matALocalImag[offset], T(-1), blockN);
            PipeBarrier<PIPE_V>();
        }
        augQueueReal.EnQue(augLocalReal);
        augQueueImag.EnQue(augLocalImag);
        augQueueImagNeg.EnQue(augLocalImagNeg);
        triangularQueueReal.FreeTensor(triangularLocalReal);
        triangularQueueImag.FreeTensor(triangularLocalImag);
        MatAQueueReal.FreeTensor(matALocalReal);
        MatAQueueImag.FreeTensor(matALocalImag);

        augLocalReal = augQueueReal.DeQue<T>();
        augLocalImag = augQueueImag.DeQue<T>();
        augLocalImagNeg = augQueueImagNeg.DeQue<T>();

        DataCopyParams copyParamsMatOut{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                        0, static_cast<uint16_t>((N - blockN) / elementsPerBlock)};

        DataCopy(aRealGlobal[matABlockOffset], augLocalReal, copyParamsMatOut);
        DataCopy(aImagGlobal[matABlockOffset], augLocalImagNeg, copyParamsMatOut);
        DataCopy(aImagNegGlobal[matABlockOffset], augLocalImag, copyParamsMatOut);
        augQueueReal.FreeTensor(augLocalReal);
        augQueueImag.FreeTensor(augLocalImag);
        augQueueImagNeg.FreeTensor(augLocalImagNeg);
    }
};
__aicore__ inline void custom_trsm(TBufPool<TPosition::VECCALC, 16> &tbufPool, CMatmulCustom<float> &opmm, int M, int N,
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
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    coreIdx >>= 1;
#endif

    GlobalTensor<float> aGlobalReal;
    GlobalTensor<float> aGlobalImag;
    GlobalTensor<float> aGlobalImagNeg;
    GlobalTensor<float> lGlobalReal;
    GlobalTensor<float> lGlobalImag;

    lGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(L));
    lGlobalImag = lGlobalReal[M * strideN];

    int block_n = LU_CEIL_ALIGN((N + blockNum - 1) / blockNum, 64);
    int trsmOffsetN = block_n * coreIdx;
    int realBlockN = min(N - trsmOffsetN, block_n);
    if (realBlockN <= 0)
    {
        return;
    }

#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    blockNum <<= 1;

    aGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A + trsmOffsetN * sizeof(float)));

    int doubleAIV = realBlockN > 64;
    SolveTrsmCustom<float> optrsm;
    auto trsmAGlobalReal = aGlobalReal[(blockIdx & 1) * (realBlockN / 2)];
    auto trsmAGlobalImag = aGlobalReal[(blockIdx & 1) * (realBlockN / 2) + M * strideN];
    auto trsmAGlobalImagNeg = aGlobalReal[(blockIdx & 1) * (realBlockN / 2) + M * strideN * 2];
    if (doubleAIV)
        optrsm.Init(&tbufPool, trsmAGlobalReal, trsmAGlobalImag, trsmAGlobalImagNeg, lGlobalReal, lGlobalImag, strideN,
                    strideN, block_m, realBlockN / 2);
    else if (~blockIdx & 1)
        optrsm.Init(&tbufPool, trsmAGlobalReal, trsmAGlobalImag, trsmAGlobalImagNeg, lGlobalReal, lGlobalImag, strideN,
                    strideN, block_m, realBlockN);
#endif

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    aGlobalReal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A + trsmOffsetN * sizeof(float)));
    aGlobalImag = aGlobalReal[M * strideN];
    aGlobalImagNeg = aGlobalReal[M * strideN * 2];
    opmm.SetMatrix(lGlobalReal, lGlobalImag, aGlobalReal, aGlobalImag, aGlobalImagNeg, aGlobalReal, aGlobalImag,
                   strideN, strideN);
    SetAtomicAdd<float>();
#endif
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    if (blockIdx % 2 <= doubleAIV) optrsm.Process(0);
    if (cnt > 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
#endif

    int big_block_m = 0;
    int big_block_slice_m;
    int big_block_offsetA;
    int big_block_offsetB;
    int big_block_offsetC;
    for (int i = 1, offset_m = block_m, big_block_offset_m = 0; i < cnt; ++i, offset_m += block_m)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
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
#elif defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        CrossCoreWaitFlag(0x1);
        if (blockIdx % 2 <= doubleAIV) optrsm.Process(offset_m);
        if (i != cnt - 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
#endif
    }
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    tbufPool.Reset();
#endif
}

#endif  // _TRSM_CUSTOM_HPP_
