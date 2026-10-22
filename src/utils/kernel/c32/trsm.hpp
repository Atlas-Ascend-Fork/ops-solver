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
 * \file trsm.hpp
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
    GlobalTensor<T> aGlobal, lGlobal, xGlobal;
    TQue<QuePosition::VECIN, 1> triangularQueue;
    TQue<QuePosition::VECIN, 1> MatAQueue;
    TQue<QuePosition::VECOUT, 1> augQueue;
    static constexpr int maxN = 256;
    uint64_t elementsPerBlock;
    int M, N;
    int blockM, blockN;

   public:
    __aicore__ inline SolveTrsm() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, int blockM)
    {
        this->blockM = blockM;
        elementsPerBlock = 32 / sizeof(T);

        pipe->InitBuffer(triangularQueue, 1, blockM * blockM * sizeof(T));
        pipe->InitBuffer(MatAQueue, 1, blockM * maxN * sizeof(T));
        pipe->InitBuffer(augQueue, 1, blockM * maxN * sizeof(T));
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> &aGlobal, GlobalTensor<T> &lGlobal, GlobalTensor<T> &xGlobal,
                                     int M, int N, int blockN)
    {
        this->M = M;
        this->N = N;
        this->blockN = blockN;

        this->aGlobal = aGlobal;
        this->lGlobal = lGlobal;
        this->xGlobal = xGlobal;
    }
    __aicore__ inline void Process(int offsetM)
    {
        uint64_t matABlockOffset = offsetM * N;
        uint64_t matLBlockOffset = offsetM * N + offsetM;

        // CopyIn
        LocalTensor<T> triangularLocal = triangularQueue.AllocTensor<T>();
        LocalTensor<T> matALocal = MatAQueue.AllocTensor<T>();

        // copy L
        DataCopyParams copyParamsMatL{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockM / elementsPerBlock),
                                      static_cast<uint16_t>((M - blockM) / elementsPerBlock), 0};
        DataCopy(triangularLocal, lGlobal[matLBlockOffset], copyParamsMatL);

        // copy A
        DataCopyParams copyParamsMatA{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                      static_cast<uint16_t>((N - blockN) / elementsPerBlock), 0};
        DataCopy(matALocal, aGlobal[matABlockOffset], copyParamsMatA);
        triangularQueue.EnQue(triangularLocal);
        MatAQueue.EnQue(matALocal);
        triangularLocal = triangularQueue.DeQue<T>();
        matALocal = MatAQueue.DeQue<T>();
        LocalTensor<T> augLocal = augQueue.AllocTensor<T>();

        for (uint32_t rowIdx = 0; rowIdx < blockM; ++rowIdx)
        {
            for (uint32_t tempRowIdx = 0; tempRowIdx < rowIdx; ++tempRowIdx)
            {
                T colScalar = triangularLocal.GetValue(tempRowIdx + rowIdx * blockM);
                Axpy(matALocal[rowIdx * blockN], augLocal[tempRowIdx * blockN], colScalar, blockN);
                PipeBarrier<PIPE_V>();
            }
            Muls(augLocal[rowIdx * blockN], matALocal[rowIdx * blockN], T(-1), blockN);
        }

        augQueue.EnQue(augLocal);
        triangularQueue.FreeTensor(triangularLocal);
        MatAQueue.FreeTensor(matALocal);

        // CopyOut
        augLocal = augQueue.DeQue<T>();

        DataCopyParams copyParamsMatOut{static_cast<uint16_t>(blockM), static_cast<uint16_t>(blockN / elementsPerBlock),
                                        0, static_cast<uint16_t>((N - blockN) / elementsPerBlock)};
        DataCopy(xGlobal[matABlockOffset], augLocal, copyParamsMatOut);
        augQueue.FreeTensor(augLocal);
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
__aicore__ inline void solveTrsm(SolveTrsm<T> &optrsm, MatmulCustom<T> &opmm, int M, int N, int strideN, int blockM,
                                 GlobalTensor<T> aGlobal, GlobalTensor<T> lGlobal)
{
    int blockIdx = GetBlockIdx();
    int baseM = 128;
    int bigBlockCount;
    int lim = 512 / blockM;
    int cnt = M / blockM;
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    int halfN = (N / 2 + 15) / 16 * 16;
    auto trsmAGlobal = aGlobal[(blockIdx & 1) * halfN];
    optrsm.SetMatrix(trsmAGlobal, lGlobal, trsmAGlobal, strideN, strideN, blockIdx & 1 ? N - halfN : halfN);
#endif
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    opmm.SetMatrix(lGlobal, aGlobal, aGlobal, strideN, strideN);
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
