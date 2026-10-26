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
 * \file merge.hpp
 * \brief 复数矩阵实部/虚部合并的公共实现（MergeRealImag / MergeRealImagWithTrans）。
 */

#ifndef CPOTRF_KERNEL_MERGE_HPP
#define CPOTRF_KERNEL_MERGE_HPP

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

// 合并阶段的公共部分：64x64 分块遍历、gather 索引加载、实虚拼装与回写。
// 派生类只需实现 Process（转置/非转置的偏移与行列互换不同）。
template <typename T, typename Derived>
class MergeRealImagBase
{
protected:
    static const int TILE_LENGTH = 4096;
    static const int m = 64;
    static const int n = 64;
    GlobalTensor<T> aOrgGlobal;
    GlobalTensor<T> aGlobalReal, aGlobalImag;
    GlobalTensor<uint32_t> aGather;
    int orgM, orgN, lda;
    int srcStride;
    int strideBatchSrc, strideBatchDst;
    int elementsPerBlock;
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    TQue<QuePosition::VECIN, 1> gatherQueue;
    LocalTensor<uint32_t> gatherLocal;

    __aicore__ inline void InitBase(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<uint32_t> aGather)
    {
        pipe->InitBuffer(inQueue, 2, 2 * TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(outQueue, 2, 2 * TILE_LENGTH * sizeof(T));
        pipe->InitBuffer(gatherQueue, 1, 2 * TILE_LENGTH * sizeof(uint32_t));
        elementsPerBlock = 32 / sizeof(T);
        this->aGather = aGather;
    }

    // srcStride 为源矩阵行间距（非转置用 strideN，转置用 padStride）
    __aicore__ inline void SetMatrixBase(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                         GlobalTensor<T> aGlobalImag, int orgM, int orgN, int lda, int srcStride,
                                         int strideBatchSrc, int strideBatchDst)
    {
        this->aOrgGlobal = aOrgGlobal;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->orgM = orgM;
        this->orgN = orgN;
        this->lda = lda;
        this->srcStride = srcStride;
        if (strideBatchSrc)
            this->strideBatchSrc = strideBatchSrc;
        else
            this->strideBatchSrc = (orgM + 127) / 128 * 128 * srcStride;
        if (strideBatchDst)
            this->strideBatchDst = strideBatchDst;
        else
            this->strideBatchDst = 2 * orgM * orgN;
    }

    __aicore__ inline void MergeWork(int p, int id, int batch)
    {
        // 调用方传入的 p 是参与合并的核数（blockNum << 1 >= 2），退化入参直接返回，避免除零
        if (p == 0)
        {
            return;
        }
        int cntN = (orgN + n - 1) / n, cntM = (orgM + m - 1) / m;
        int totalNum = batch * cntN * cntM;
        int k = totalNum / p, r = totalNum % p;
        int startIdx = k * id + min(id, r);
        int calcNum = (k + (id < r));
        if (calcNum == 0)
        {
            return;
        }
        LoadGather();
        for (int i = startIdx; i < startIdx + calcNum; ++i)
        {
            int batchIdx = i / cntN / cntM;
            int idM = (i - batchIdx * cntN * cntM) / cntN, idN = i % cntN;
            static_cast<Derived*>(this)->Process(idM, idN, batchIdx);
        }
        gatherQueue.FreeTensor(gatherLocal);
    }

    __aicore__ inline void LoadGather()
    {
        gatherLocal = gatherQueue.AllocTensor<uint32_t>();
        DataCopy(gatherLocal, aGather, 2 * m * n);
        gatherQueue.EnQue(gatherLocal);
        gatherLocal = gatherQueue.DeQue<uint32_t>();
    }

    __aicore__ inline void CopyIn(int offset, int rows, int cols, int srcStride)
    {
        LocalTensor<T> srcLocal = inQueue.AllocTensor<T>();
        int copyDB = (cols + elementsPerBlock - 1) / elementsPerBlock;
        DataCopyParams copyInParams{static_cast<uint16_t>(rows), static_cast<uint16_t>(copyDB),
                                    static_cast<uint16_t>(srcStride / elementsPerBlock - copyDB),
                                    static_cast<uint16_t>(n / elementsPerBlock - copyDB)};
        DataCopy(srcLocal, aGlobalReal[offset], copyInParams);
        DataCopy(srcLocal[TILE_LENGTH], aGlobalImag[offset], copyInParams);
        inQueue.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(int length)
    {
        LocalTensor<T> srcLocal = inQueue.DeQue<T>();
        LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
        Gather(dstLocal, srcLocal, gatherLocal, 0, length);
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(dstLocal);
    }

    // nCols 为回写时的列数基准（非转置为 n，转置为 m）
    __aicore__ inline void CopyOut(int offset, int rows, int cols, int nCols)
    {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        DataCopyExtParams copyOutParams{static_cast<uint16_t>(rows), static_cast<uint32_t>(cols * 2 * sizeof(T)),
                                        static_cast<uint32_t>((nCols - cols) * 2 / elementsPerBlock),
                                        static_cast<uint32_t>((lda - cols) * 2 * sizeof(T)), 0};
        DataCopyPad(aOrgGlobal[offset], dstLocal, copyOutParams);
        outQueue.FreeTensor(dstLocal);
    }
};

// 非转置合并：按 (idM, idN) 顺序处理，行列保持原序
template <typename T>
class MergeRealImag : public MergeRealImagBase<T, MergeRealImag<T>>
{
   public:
    __aicore__ inline MergeRealImag() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<uint32_t> aGather)
    {
        this->InitBase(pipe, aGather);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                     GlobalTensor<T> aGlobalImag, int orgM, int orgN, int lda, int strideN,
                                     int strideBatchSrc = 0, int strideBatchDst = 0)
    {
        this->SetMatrixBase(aOrgGlobal, aGlobalReal, aGlobalImag, orgM, orgN, lda, strideN, strideBatchSrc,
                            strideBatchDst);
    }
    __aicore__ inline void MergeRealImagWork(int p, int id, int batch) { this->MergeWork(p, id, batch); }
    __aicore__ inline void Process(int idM, int idN, int batch_idx)
    {
        int realM = min(this->m, this->orgM - idM * this->m);
        int realN = min(this->n, this->orgN - idN * this->n);
        this->CopyIn(batch_idx * this->strideBatchSrc + idM * this->m * this->srcStride + idN * this->n, realM, realN,
                     this->srcStride);
        this->Compute(2 * realM * this->n);
        this->CopyOut(batch_idx * this->strideBatchDst + 2 * (idM * this->m * this->lda + idN * this->n), realM,
                      realN, this->n);
    }
};

// 转置合并：A^H 写回，行列互换
template <typename T>
class MergeRealImagWithTrans : public MergeRealImagBase<T, MergeRealImagWithTrans<T>>
{
   public:
    __aicore__ inline MergeRealImagWithTrans() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16>* pipe, GlobalTensor<uint32_t> aGather)
    {
        this->InitBase(pipe, aGather);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aOrgGlobal, GlobalTensor<T> aGlobalReal,
                                     GlobalTensor<T> aGlobalImag, int orgM, int orgN, int lda, int padStride,
                                     int strideBatchSrc = 0, int strideBatchDst = 0)
    {
        this->SetMatrixBase(aOrgGlobal, aGlobalReal, aGlobalImag, orgM, orgN, lda, padStride, strideBatchSrc,
                            strideBatchDst);
    }
    __aicore__ inline void MergeRealImagWork(int p, int id, int batch) { this->MergeWork(p, id, batch); }
    __aicore__ inline void Process(int idM, int idN, int batch_idx)
    {
        int realM = min(this->m, this->orgM - idM * this->m);
        int realN = min(this->n, this->orgN - idN * this->n);
        this->CopyIn(batch_idx * this->strideBatchSrc + idM * this->m * this->srcStride + idN * this->n, realM, realN,
                     this->srcStride);
        this->Compute(2 * this->m * this->n);
        this->CopyOut(batch_idx * this->strideBatchDst + 2 * (idN * this->n * this->lda + idM * this->m), realN,
                      realM, this->m);
    }
};

#endif  // CPOTRF_KERNEL_MERGE_HPP
