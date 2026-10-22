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
 * \file trsm.cpp
 * \brief
 */

#ifndef _TRSM_HPP_
#define _TRSM_HPP_

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "gemm.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class SolveTrsm
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
    __aicore__ inline SolveTrsm() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, int blockM)
    {
        this->blockM = blockM;

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
    __aicore__ inline void SetMatrix(GlobalTensor<T> &aRealGlobal, GlobalTensor<T> &aImagGlobal,
                                     GlobalTensor<T> &aImagNegGlobal, GlobalTensor<T> &lRealGlobal,
                                     GlobalTensor<T> &lImagGlobal, int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aRealGlobal = aRealGlobal;
        this->aImagGlobal = aImagGlobal;
        this->aImagNegGlobal = aImagNegGlobal;
        this->lRealGlobal = lRealGlobal;
        this->lImagGlobal = lImagGlobal;
    }
    __aicore__ inline void Process(int offset_m)
    {
        uint64_t matABlockOffset = offset_m * N;
        uint64_t matLBlockOffset = offset_m * M + offset_m;
        // CopyIn
        LocalTensor<T> triangularLocalReal = triangularQueueReal.AllocTensor<T>();
        LocalTensor<T> triangularLocalImag = triangularQueueImag.AllocTensor<T>();
        LocalTensor<T> matALocalReal = MatAQueueReal.AllocTensor<T>();
        LocalTensor<T> matALocalImag = MatAQueueImag.AllocTensor<T>();

        // copy L
        DataCopyParams copyParamsMatL{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockM / elementsPerBlock),
                                      static_cast<uint16_t>((M - blockM) / elementsPerBlock), 0};
        DataCopy(triangularLocalReal, lRealGlobal[matLBlockOffset], copyParamsMatL);
        DataCopy(triangularLocalImag, lImagGlobal[matLBlockOffset], copyParamsMatL);

        // copy A
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
                PipeBarrier<PIPE_ALL>();
            }
            Muls(augLocalReal[offset], matALocalReal[offset], T(-1), blockN);
            PipeBarrier<PIPE_V>();
            Muls(augLocalImag[offset], matALocalImag[offset], T(1), blockN);
            PipeBarrier<PIPE_V>();
            Muls(augLocalImagNeg[offset], matALocalImag[offset], T(-1), blockN);
            PipeBarrier<PIPE_ALL>();
        }

        augQueueReal.EnQue(augLocalReal);
        augQueueImag.EnQue(augLocalImag);
        augQueueImagNeg.EnQue(augLocalImagNeg);
        triangularQueueReal.FreeTensor(triangularLocalReal);
        triangularQueueImag.FreeTensor(triangularLocalImag);
        MatAQueueReal.FreeTensor(matALocalReal);
        MatAQueueImag.FreeTensor(matALocalImag);

        // CopyOut
        augLocalReal = augQueueReal.DeQue<T>();
        augLocalImag = augQueueImag.DeQue<T>();
        augLocalImagNeg = augQueueImagNeg.DeQue<T>();

        DataCopyParams copyParamsMatOut{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                        0, static_cast<uint16_t>((N - blockN) / elementsPerBlock)};

        DataCopy(aRealGlobal[matABlockOffset], augLocalReal, copyParamsMatOut);
        DataCopy(aImagGlobal[matABlockOffset], augLocalImag, copyParamsMatOut);
        DataCopy(aImagNegGlobal[matABlockOffset], augLocalImagNeg, copyParamsMatOut);
        augQueueReal.FreeTensor(augLocalReal);
        augQueueImag.FreeTensor(augLocalImag);
        augQueueImagNeg.FreeTensor(augLocalImagNeg);
    }
};

template <typename T>
class NegateCustom
{
    TQue<QuePosition::VECIN, 1> inQueueSrc;
    TQue<QuePosition::VECOUT, 1> outQueueDst;
    GlobalTensor<T> aGlobal;
    int N;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 8192;

   public:
    __aicore__ inline NegateCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(inQueueSrc, 1, TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueueDst, 1, TILE_LENGTH * sizeof(T));
        this->elementsPerBlock = 32 / sizeof(T);
    }
    __aicore__ inline void SetMatrix(const GlobalTensor<T> &aGlobal, int N)
    {
        this->aGlobal = aGlobal;
        this->N = N;
    }
    __aicore__ inline void Process(int blockM, int blockN)
    {
        int linesPerIter = TILE_LENGTH / blockN;
        for (int i = 0, offset = 0; i < blockM; i += linesPerIter, offset += linesPerIter * N)
        {
            int t = min(linesPerIter, blockM - i);
            {
                LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
                DataCopyParams copyInParams{static_cast<uint16_t>(t), static_cast<uint16_t>(blockN / elementsPerBlock),
                                            static_cast<uint16_t>((N - blockN) / elementsPerBlock), 0};
                DataCopy(srcLocal, aGlobal[offset], copyInParams);
                inQueueSrc.EnQue(srcLocal);
            }
            {
                LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
                LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
                Muls(dstLocal, srcLocal, T(-1), t * blockN);
                inQueueSrc.FreeTensor(srcLocal);
                outQueueDst.EnQue(dstLocal);
            }
            {
                LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
                DataCopyParams copyOutParams{static_cast<uint16_t>(t), static_cast<uint16_t>(blockN / elementsPerBlock),
                                             0, static_cast<uint16_t>((N - blockN) / elementsPerBlock)};
                DataCopy(aGlobal[offset], dstLocal, copyOutParams);
                outQueueDst.FreeTensor(dstLocal);
            }
        }
    }

   private:
    __aicore__ inline void CalcRow(int st, int length)
    {
        CopyIn(st, length);
        Compute(length);
        CopyOut(st, length);
    }
    __aicore__ inline void CopyIn(int offset, int length)
    {
        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopy(srcLocal, aGlobal[offset], length);
        inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int length)
    {
        LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
        LocalTensor<T> dstLocal = outQueueDst.AllocTensor<T>();
        Muls(dstLocal, srcLocal, (T)-1, length);
        outQueueDst.EnQue<T>(dstLocal);
        inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut(int offset, int length)
    {
        LocalTensor<T> dstLocal = outQueueDst.DeQue<T>();
        DataCopy(aGlobal[offset], dstLocal, length);
        outQueueDst.FreeTensor(dstLocal);
    }
};

template <typename T>
__aicore__ inline void solveTrsm(SolveTrsm<T> &optrsm, CMatmulCustom<T> &opmm, int M, int N, int strideN, int blockM,
                                 GlobalTensor<T> aGlobalReal, GlobalTensor<T> aGlobalImag,
                                 GlobalTensor<T> aGlobalImagNeg, GlobalTensor<T> lGlobalReal,
                                 GlobalTensor<T> lGlobalImag)
{
    int blockIdx = GetBlockIdx();
    int baseM = 128;
    int bigBlockCount;
    int lim = 512 / blockM;
    int cnt = M / blockM;
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    int halfN = (N / 2 + 15) / 16 * 16;
    auto trsmAGlobalReal = aGlobalReal[(blockIdx & 1) * halfN];
    auto trsmAGlobalImag = aGlobalImag[(blockIdx & 1) * halfN];
    auto trsmAGlobalImagNeg = aGlobalImagNeg[(blockIdx & 1) * halfN];
    optrsm.SetMatrix(trsmAGlobalReal, trsmAGlobalImag, trsmAGlobalImagNeg, lGlobalReal, lGlobalImag, strideN, strideN,
                     blockIdx & 1 ? N - halfN : halfN);
#endif
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    opmm.SetMatrix(lGlobalReal, lGlobalImag, aGlobalReal, aGlobalImagNeg, aGlobalImag, aGlobalReal, aGlobalImag,
                   strideN, strideN);
#endif
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    optrsm.Process(0);
    if (cnt > 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
#endif
    int bigBlockM = 0;
    int bigBlockOffsetA, bigBlockOffsetB, bigBlockOffsetC;
    for (int i = 1, offsetM = blockM, bigBlockOffsetM = 0; i < cnt; ++i, offsetM += blockM)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        int r = i % lim;
        if (r)
        {
            int offsetA = offsetM * strideN + offsetM - blockM;
            int offsetB = (offsetM - blockM) * strideN;
            int offsetC = offsetM * strideN;
            CrossCoreWaitFlag(0x0);
            opmm.Process(offsetA, offsetB, offsetC, blockM, N, blockM);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);
            if (r == lim - 1) continue;
            if (offsetM + blockM >= M) continue;
            offsetA = (offsetM + blockM) * strideN + bigBlockOffsetM;
            offsetB = bigBlockOffsetM * strideN;
            offsetC += blockM * strideN;
            opmm.Process(offsetA, offsetB, offsetC, blockM, N, blockM * r);

            if (bigBlockM == 0 || r >= bigBlockCount - 1) continue;
            bigBlockOffsetA += baseM * strideN;
            bigBlockOffsetC += baseM * strideN;
            int bigBlockTmp = min(M - bigBlockOffsetA / strideN, baseM);
            if (bigBlockTmp <= 0) continue;
            opmm.Process(bigBlockOffsetA, bigBlockOffsetB, bigBlockOffsetC, bigBlockTmp, N, bigBlockM);
        }
        else
        {
            bigBlockM = (i & -i) * blockM;
            bigBlockCount = bigBlockM / baseM;
            bigBlockOffsetM += lim * blockM;
            int LOffsetN = offsetM - bigBlockM;
            bigBlockOffsetA = offsetM * strideN + LOffsetN;
            bigBlockOffsetB = LOffsetN * strideN;
            bigBlockOffsetC = offsetM * strideN;
            CrossCoreWaitFlag(0x0);
            opmm.Process(bigBlockOffsetA, bigBlockOffsetB, bigBlockOffsetC, min(M - bigBlockOffsetA / strideN, baseM),
                         N, bigBlockM);
            CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);
            bigBlockOffsetA += baseM * strideN;
            bigBlockOffsetC += baseM * strideN;
            int bigBlockTmp = min(M - bigBlockOffsetA / strideN, baseM);
            if (bigBlockTmp <= 0) continue;
            opmm.Process(bigBlockOffsetA, bigBlockOffsetB, bigBlockOffsetC, bigBlockTmp, N, bigBlockM);
        }
#elif defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        CrossCoreWaitFlag(0x1);
        optrsm.Process(offsetM);
        if (i != cnt - 1) CrossCoreSetFlag<0x2, PIPE_MTE3>(0x0);
#endif
    }
}

#endif  // _TRSM_HPP_
