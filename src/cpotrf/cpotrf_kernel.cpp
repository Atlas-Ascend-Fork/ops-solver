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
 * \file cpotrf_kernel.cpp
 * \brief
 */

#ifndef CPOTRF_MIX_H
#define CPOTRF_MIX_H

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "./kernel/gemm.hpp"
#include "./kernel/merge.hpp"
#include "./kernel/pad.hpp"
#include "cpotrf_launchers.hpp"
#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class LLCustom
{
    GlobalTensor<T> aGlobalReal, aGlobalImag, aGlobalRealNeg, aGlobalImagNeg;
    int orgN, padN, blockNum;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    int N;
    int blockIdx;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 1> inQueueSrc1;
    TQue<QuePosition::VECIN, 1> inQueueSrc2;

    __aicore__ inline void CopyIn(int MatAOffset, int realM, int p, int offset, int N, LocalTensor<T>& srcLocalReal,
                                  LocalTensor<T>& srcLocalImag)
    {
        int lenth = (p + 1) * realM;
        int blockSize = realM * ((realM + p * realM + 7) / 8 * 8);

        srcLocalReal = inQueueSrc1.AllocTensor<T>();
        srcLocalImag = srcLocalReal[blockSize];

        DataCopyParams copyInParams1{static_cast<uint16_t>(realM), static_cast<uint16_t>(realM / 8),
                                     static_cast<uint16_t>((N - realM) / 8), static_cast<uint16_t>(p * realM / 8)};
        DataCopyParams copyInParams2{static_cast<uint16_t>(realM), static_cast<uint16_t>(p * realM / 8),
                                     static_cast<uint16_t>((N - p * realM) / 8), static_cast<uint16_t>(realM / 8)};

        DataCopy(srcLocalReal, aGlobalReal[MatAOffset], copyInParams1);
        DataCopy(srcLocalReal[realM], aGlobalReal[MatAOffset + offset], copyInParams2);
        DataCopy(srcLocalImag, aGlobalImag[MatAOffset], copyInParams1);
        DataCopy(srcLocalImag[realM], aGlobalImag[MatAOffset + offset], copyInParams2);
        inQueueSrc1.EnQue(srcLocalReal);
    }

    __aicore__ inline void Compute(int realM, int lenth, LocalTensor<T>& srcLocalReal, LocalTensor<T>& srcLocalImag)
    {
        srcLocalReal = inQueueSrc1.DeQue<T>();
        int blockSize = realM * ((realM + (lenth / realM - 1) * realM + 7) / 8 * 8);
        srcLocalImag = srcLocalReal[blockSize];

        LocalTensor<T> tmpLocalRealNeg = inQueueSrc2.AllocTensor<T>();
        LocalTensor<T> tmpLocalImag = tmpLocalRealNeg[8 * realM];
        LocalTensor<T> tmpLocalImagNeg = tmpLocalImag[8 * realM];

        for (int j = 0, offset3 = 0; j < realM; ++j, offset3 += lenth)
        {
            FactorizeColumn(j, offset3, realM, lenth, srcLocalReal, srcLocalImag, tmpLocalRealNeg, tmpLocalImag,
                            tmpLocalImagNeg);
        }
        inQueueSrc2.FreeTensor(tmpLocalRealNeg);
    }

    // 单列 Cholesky：先按主元归一化该列，再对其右侧各列做秩一更新
    __aicore__ inline void FactorizeColumn(int j, int offset3, int realM, int lenth, LocalTensor<T>& srcLocalReal,
                                           LocalTensor<T>& srcLocalImag, LocalTensor<T>& tmpLocalRealNeg,
                                           const LocalTensor<T>& tmpLocalImag, LocalTensor<T>& tmpLocalImagNeg)
    {
        T ljj = srcLocalReal.GetValue(offset3 + j);
        if (!(ljj > 0))
        {
            return;
        }
        ljj = sqrt(ljj);
        T colScalar = 1 / ljj;
        AscendC::Muls(srcLocalReal[offset3], srcLocalReal[offset3], colScalar, lenth);
        AscendC::Muls(srcLocalImag[offset3], srcLocalImag[offset3], colScalar, lenth);
        srcLocalReal.SetValue(offset3 + j, ljj);
        srcLocalImag.SetValue(offset3 + j, 0);

        AscendC::Brcb(tmpLocalRealNeg, srcLocalReal[offset3], realM / 8, {1, 8});
        AscendC::Brcb(tmpLocalImag, srcLocalImag[offset3], realM / 8, {1, 8});
        AscendC::Muls(tmpLocalRealNeg, tmpLocalRealNeg, T(-1), 8 * realM);
        AscendC::Muls(tmpLocalImagNeg, tmpLocalImag, T(-1), 8 * realM);
        UpdateTrailing(j, offset3, realM, lenth, srcLocalReal, srcLocalImag, tmpLocalRealNeg, tmpLocalImag,
                       tmpLocalImagNeg);
    }

    // 秩一更新：按 64 列一段滚动，末尾不足 64 列的余量单独处理
    __aicore__ inline void UpdateTrailing(int j, int offset3, int realM, int lenth, LocalTensor<T>& srcLocalReal,
                                          LocalTensor<T>& srcLocalImag, const LocalTensor<T>& tmpLocalRealNeg,
                                          const LocalTensor<T>& tmpLocalImag, const LocalTensor<T>& tmpLocalImagNeg)
    {
        int q = lenth / 64;
        int rem = lenth % 64;
        for (int i = 0; i < q; i++)
        {
            MulAddQuad(offset3 + lenth + i * 64, offset3 + i * 64, 64, j, realM, lenth, srcLocalReal, srcLocalImag,
                       tmpLocalRealNeg, tmpLocalImag, tmpLocalImagNeg);
        }
        if (rem)
        {
            MulAddQuad(offset3 + 2 * lenth - rem, offset3 + lenth - rem, rem, j, realM, lenth, srcLocalReal,
                       srcLocalImag, tmpLocalRealNeg, tmpLocalImag, tmpLocalImagNeg);
        }
    }

    // 一段列上的复数乘加：(a+bi) -= (v+wi)*ljj，实虚交叉共 4 次 MulAddDst
    __aicore__ inline void MulAddQuad(int dstOffset, int srcOffset, int count, int j, int realM, int lenth,
                                      LocalTensor<T>& srcLocalReal, LocalTensor<T>& srcLocalImag,
                                      const LocalTensor<T>& tmpLocalRealNeg, const LocalTensor<T>& tmpLocalImag,
                                      const LocalTensor<T>& tmpLocalImagNeg)
    {
        int repeat = realM - j - 1;
        uint8_t srcStride = static_cast<uint8_t>(lenth / 8);
        AscendC::MulAddDst(srcLocalReal[dstOffset], srcLocalReal[srcOffset], tmpLocalRealNeg[8 * (j + 1)], count,
                           repeat, {1, 1, 0, srcStride, 0, 1});
        AscendC::MulAddDst(srcLocalReal[dstOffset], srcLocalImag[srcOffset], tmpLocalImagNeg[8 * (j + 1)], count,
                           repeat, {1, 1, 0, srcStride, 0, 1});
        AscendC::MulAddDst(srcLocalImag[dstOffset], srcLocalReal[srcOffset], tmpLocalImag[8 * (j + 1)], count,
                           repeat, {1, 1, 0, srcStride, 0, 1});
        AscendC::MulAddDst(srcLocalImag[dstOffset], srcLocalImag[srcOffset], tmpLocalRealNeg[8 * (j + 1)], count,
                           repeat, {1, 1, 0, srcStride, 0, 1});
    }

    __aicore__ inline void CopyOut(int MatAOffset, int offset, int realM, int p, int Idx, LocalTensor<T>& srcLocalReal,
                                   LocalTensor<T>& srcLocalImag)
    {
        int lenth = (p + 1) * realM;
        int blockSize = realM * ((realM + p * realM + 7) / 8 * 8);

        DataCopyParams copyOutParams1{static_cast<uint16_t>(realM), static_cast<uint16_t>(realM / 8),
                                      static_cast<uint16_t>(p * realM / 8), static_cast<uint16_t>((N - realM) / 8)};
        DataCopyParams copyOutParams2{static_cast<uint16_t>(realM), static_cast<uint16_t>(p * realM / 8),
                                      static_cast<uint16_t>(realM / 8), static_cast<uint16_t>((N - p * realM) / 8)};

        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(aGlobalReal[MatAOffset + offset], srcLocalReal[realM], copyOutParams2);
        DataCopy(aGlobalImag[MatAOffset + offset], srcLocalImag[realM], copyOutParams2);
        if (Idx == 0)
        {
            DataCopy(aGlobalReal[MatAOffset], srcLocalReal, copyOutParams1);
            DataCopy(aGlobalImag[MatAOffset], srcLocalImag, copyOutParams1);
        }

        int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);

        AscendC::Muls(srcLocalReal, srcLocalReal, T(-1), blockSize);
        AscendC::Muls(srcLocalImag, srcLocalImag, T(-1), blockSize);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(aGlobalRealNeg[MatAOffset + offset], srcLocalReal[realM], copyOutParams2);
        DataCopy(aGlobalImagNeg[MatAOffset + offset], srcLocalImag[realM], copyOutParams2);
        if (Idx == 0)
        {
            DataCopy(aGlobalRealNeg[MatAOffset], srcLocalReal, copyOutParams1);
            DataCopy(aGlobalImagNeg[MatAOffset], srcLocalImag, copyOutParams1);
        }
        inQueueSrc1.FreeTensor(srcLocalReal);
    }

   public:
    __aicore__ inline LLCustom() {}
    __aicore__ inline void Init(TQue<QuePosition::VECIN, 1>& inQueueSrc1, TQue<QuePosition::VECIN, 1>& inQueueSrc2,
                                GlobalTensor<T> aGlobalReal, GlobalTensor<T> aGlobalImag,
                                GlobalTensor<T> aGlobalRealNeg, GlobalTensor<T> aGlobalImagNeg, int orgN, int padN,
                                int blockNum, int N)
    {
        this->orgN = orgN;
        this->padN = padN;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->aGlobalRealNeg = aGlobalRealNeg;
        this->aGlobalImagNeg = aGlobalImagNeg;
        this->elementsPerBlock = 32 / sizeof(T);
        this->inQueueSrc1 = inQueueSrc1;
        this->inQueueSrc2 = inQueueSrc2;
        this->blockNum = 2 * blockNum;
        this->blockIdx = GetBlockIdx();
        this->N = N;
    }

    __aicore__ inline void Process3(int MatAOffset, int offsetM, int offsetN, int realM, int realN, int batch)
    {
        // ---- 一次性计算所有索引 ----
        realN = (realN + realM - 1) / realM * realM;
        int count = realN / realM - 1;

        int batch_block_num = blockNum / batch;
        int r = blockNum % batch;
        int Idx;
        int batchIdx;
        if (blockIdx < (batch_block_num + 1) * r)
        {
            batch_block_num++;
            batchIdx = blockIdx / batch_block_num;
            Idx = blockIdx % batch_block_num;
        }
        else
        {
            batchIdx = r + (blockIdx - (batch_block_num + 1) * r) / batch_block_num;
            Idx = (blockIdx - (batch_block_num + 1) * r) % batch_block_num;
        }
        int sizeA = N * N;
        MatAOffset += batchIdx * sizeA;

        int p = count / batch_block_num;
        r = count % batch_block_num;
        int offset = 0;
        if (Idx < r)
        {
            p++;
            offset = Idx * p * realM;
        }
        else
        {
            offset = Idx * p * realM + r * realM;
        }
        offset += realM;
        if (count == 0)
        {
            if (Idx != 0) return;
            p = 0;
        }
        else if (p == 0)
        {
            return;
        }
        int lenth = (p + 1) * realM;

        LocalTensor<T> srcLocalReal, srcLocalImag;
        CopyIn(MatAOffset, realM, p, offset, N, srcLocalReal, srcLocalImag);
        Compute(realM, lenth, srcLocalReal, srcLocalImag);
        CopyOut(MatAOffset, offset, realM, p, Idx, srcLocalReal, srcLocalImag);
    }
};

template <typename T>
class CholeskySolver_CUBE
{
    int offsetA, offsetB, offsetC;
    GlobalTensor<int> aOffset;
    GlobalTensor<int> bOffset;
    GlobalTensor<int> cOffset;
    GlobalTensor<int> offsetIdx;
    int globalIdx;
    int workNum, workIdx;
    CMatmulCustom<float>* opmm;
    int blockN, orgN, strideN, batch, strideBatch;
    int blockIdx, blockNum;

   public:
    __aicore__ inline CholeskySolver_CUBE() {}
    __aicore__ inline void Init(CMatmulCustom<float>* opmm, int blockN)
    {
        this->opmm = opmm;
        this->blockN = blockN;
        this->globalIdx = this->workNum = this->workIdx = 0;
        this->blockIdx = GetBlockIdx();
        this->blockNum = GetBlockNum();
    }
    __aicore__ inline void SetOffset(GlobalTensor<int> aOffset, GlobalTensor<int> bOffset, GlobalTensor<int> cOffset,
                                     GlobalTensor<int> offsetIdx)
    {
        this->aOffset = aOffset;
        this->bOffset = bOffset;
        this->cOffset = cOffset;
        this->offsetIdx = offsetIdx;
    }
    __aicore__ inline void GemmWorkGenerate(int step)
    {
        workNum = offsetIdx.GetValue(step);
        if (step > 0) globalIdx += offsetIdx.GetValue(step - 1);
        workIdx = 0;
    }

    __aicore__ inline void WorkStep(int remainingSteps, int batch)
    {
        if (workIdx == workNum || remainingSteps <= 0) return;

        int remainingWork = workNum - workIdx;
        int computeWork = remainingWork / remainingSteps;
        int startIdx = globalIdx + workIdx;

        int p = computeWork * batch / blockNum;
        int r = (computeWork * batch) % blockNum;
        int Idx = 0;
        int idx = 0;
        if (blockIdx < r)
        {
            p++;
            Idx = blockIdx * p;
        }
        else
        {
            Idx = blockIdx * p + r;
        }

        for (int s = 0; s < batch; ++s)
        {
            for (int i = 0; i < computeWork; ++i)
            {
                if (Idx <= idx && idx < Idx + p)
                {
                    opmm->Process<true, false>(aOffset.GetValue(startIdx + i) + s * strideBatch,
                                               bOffset.GetValue(startIdx + i) + s * strideBatch,
                                               cOffset.GetValue(startIdx + i) + s * strideBatch, 128, 128, 128);
                }
                idx++;
            }
        }
        workIdx += computeWork;
    }

    __aicore__ inline void Update(int offset1, int offset2, int panel_size, int strideN, int batch)
    {
        int r = (strideN - offset2) % 128;
        int blocks = (strideN - offset2 + 128 - r) / 128;

        int tempC = offset2 * strideN + offset2 - 128 + r;
        int tempA = offset1 * strideN + offset2;
        int tempB = tempA - 128 + r;

        int p = blocks * batch / blockNum;
        r = (blocks * batch) % blockNum;
        int startIdx = 0;
        int idx = 0;
        if (blockIdx < r)
        {
            p++;
            startIdx = blockIdx * p;
        }
        else
        {
            startIdx = blockIdx * p + r;
        }
        for (int s = 0; s < batch; ++s)
        {
            for (int i = 0; i < blocks; ++i)
            {
                if (startIdx <= idx && idx < startIdx + p)
                {
                    opmm->Process<true, false>(tempA + s * strideBatch, tempB + i * 128 + s * strideBatch,
                                               tempC + i * 128 + s * strideBatch,
                                               panel_size,  // blockM
                                               128,         // blockN
                                               panel_size   // blockK
                    );
                }
                idx++;
            }
        }
    }
    __aicore__ inline void Process(int orgN, int strideN, int batch, int strideBatch)
    {
        this->orgN = orgN;
        this->strideN = strideN;
        this->batch = batch;
        this->strideBatch = strideBatch;

        int step = 0, stepIdx = 0;
        SetAtomicAdd<float>();
        for (int offset = blockN; offset < orgN; offset += blockN)
        {
            int curblock = (offset & -offset);
            CrossCoreWaitFlag(0x2);
            if (curblock < 128)
            {
                Update(offset - curblock, offset, curblock, strideN, batch);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);
                WorkStep(step--, batch);
            }
            else
            {
                Update(offset - 128, offset, 128, strideN, batch);
                CrossCoreSetFlag<0x2, PIPE_FIX>(0x1);

                step = 128 / blockN;
                GemmWorkGenerate(stepIdx);
                stepIdx++;
                WorkStep(step--, batch);
            }
        }
        SetAtomicNone();
    }
};

// 分块宽度：AIV/CUBE 两侧共用的 Cholesky 面板宽度
constexpr int CPOTRF_BLOCK_N = 32;

// cholesky_custom 用到的全局张量视图，避免各阶段重复绑定
struct CpotrfTensors
{
    GlobalTensor<float> aOrg, aReal, aImag, aRealNeg, aImagNeg;
    GlobalTensor<int> aOffset, bOffset, cOffset, offsetIdx;
    GlobalTensor<uint32_t> gather;
};

__aicore__ inline void BindCpotrfTensors(CpotrfTensors& t, int strideS, GM_ADDR A_org, GM_ADDR A_work,
                                         GM_ADDR offset1_gm, GM_ADDR offset2_gm, GM_ADDR offset3_gm,
                                         GM_ADDR offsetIdx_gm, GM_ADDR gather_gm)
{
    t.aOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_org));
    t.aReal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A_work));
    t.aImag = t.aReal[strideS];
    t.aRealNeg = t.aImag[strideS];
    t.aImagNeg = t.aRealNeg[strideS];
    t.aOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(offset1_gm));
    t.bOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(offset2_gm));
    t.cOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(offset3_gm));
    t.offsetIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int*>(offsetIdx_gm));
    t.gather.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(gather_gm));
}

// 复数矩阵拆成实部/虚部两块（转置与非转置两种搬运方式）
template <bool trans>
__aicore__ inline void CpotrfSplitStage(CpotrfTensors& t, TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgN,
                                        int lda, int strideN, int blockNum, int blockIdx)
{
    if constexpr (trans)
    {
        SplitRealImagWithTrans<float> opsp;
        opsp.Init(&tbufPool, t.gather[8192]);
        opsp.SetMatrix(t.aOrg, t.aReal, t.aImag, orgN, orgN, lda, strideN);
        opsp.SplitRealImagWork(blockNum << 1, blockIdx, 1);
    }
    else
    {
        SplitRealImag<float> opsp;
        opsp.Init(&tbufPool);
        opsp.SetMatrix(t.aOrg, t.aReal, t.aImag, orgN, lda, strideN);
        opsp.SplitRealImagWork(orgN, blockNum << 1, blockIdx);
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    tbufPool.Reset();
}

// AIV 侧的分块 Cholesky：每轮先做当前面板，再与 cube 侧握手更新尾块
__aicore__ inline void CpotrfCholeskyStage(CpotrfTensors& t, TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgN,
                                           int strideN, int blockNum)
{
    constexpr int maxN = 8192;
    int dataBlockNum = (maxN / CPOTRF_BLOCK_N + (blockNum << 1) - 1) / (blockNum << 1) + 1;
    int tempLocalSize = 3 * 8 * CPOTRF_BLOCK_N * sizeof(float);
    LLCustom<float> opLL;
    TQue<QuePosition::VECIN, 1> inQueueSrc1, inQueueSrc2;
    tbufPool.InitBuffer(inQueueSrc2, 1, tempLocalSize * sizeof(float));
    tbufPool.InitBuffer(inQueueSrc1, 1, CPOTRF_BLOCK_N * dataBlockNum * sizeof(float));
    CrossCoreWaitFlag(0x0);

    opLL.Init(inQueueSrc1, inQueueSrc2, t.aReal, t.aImag, t.aRealNeg, t.aImagNeg, orgN, strideN, blockNum, strideN);
    opLL.Process3(0, 0, 0, CPOTRF_BLOCK_N, orgN, 1);
    for (int offset = CPOTRF_BLOCK_N; offset < orgN; offset += CPOTRF_BLOCK_N)
    {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);

        CrossCoreSetFlag<0x2, PIPE_MTE3>(0x2);  // let gemm start
        CrossCoreWaitFlag(0x1);                 // wait gemm finish

        CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
        CrossCoreWaitFlag(0x0);

        opLL.Process3(offset * strideN + offset, offset, offset, CPOTRF_BLOCK_N, orgN - offset, 1);
    }
    tbufPool.Reset();
}

// 实部/虚部合并回复数矩阵（转置与非转置两种写回方式）
template <bool trans>
__aicore__ inline void CpotrfMergeStage(CpotrfTensors& t, TBufPool<TPosition::VECCALC, 16>& tbufPool, int orgN,
                                        int strideN, int blockNum, int blockIdx)
{
    if constexpr (trans)
    {  // Trans A^H -> W
        MergeRealImagWithTrans<float> opmg;
        opmg.Init(&tbufPool, t.gather);
        opmg.SetMatrix(t.aOrg, t.aReal, t.aImag, orgN, orgN, orgN, strideN);
        opmg.MergeRealImagWork(blockNum << 1, blockIdx, 1);
    }
    else
    {
        MergeRealImag<float> opmg;
        opmg.Init(&tbufPool, t.gather);
        opmg.SetMatrix(t.aOrg, t.aReal, t.aImag, orgN, orgN, orgN, strideN);
        opmg.MergeRealImagWork(blockNum << 1, blockIdx, 1);
    }
}

template <bool trans = true>
__aicore__ inline void cholesky_custom(int orgN, int lda, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR offset1_gm,
                                       GM_ADDR offset2_gm, GM_ADDR offset3_gm, GM_ADDR offsetIdx_gm, GM_ADDR gather_gm)
{
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    int strideN = (orgN + 127) / 128 * 128;
    int strideS = strideN * strideN;
    CpotrfTensors t;
    BindCpotrfTensors(t, strideS, A_org, A_work, offset1_gm, offset2_gm, offset3_gm, offsetIdx_gm, gather_gm);

    TPipe pipe;
#ifdef __DAV_C220_VEC__
    TBufPool<TPosition::VECCALC, 16> tbufPool;
    pipe.InitBufPool(tbufPool, 192 * 1024);

    CpotrfSplitStage<trans>(t, tbufPool, orgN, lda, strideN, blockNum, blockIdx);
    CpotrfCholeskyStage(t, tbufPool, orgN, strideN, blockNum);
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    CrossCoreWaitFlag(0x0);
    CpotrfMergeStage<trans>(t, tbufPool, orgN, strideN, blockNum, blockIdx);
#elif __DAV_C220_CUBE__
    if (CPOTRF_BLOCK_N >= orgN)
    {
        return;
    }
    CMatmulCustom<float> opmm;
    opmm.Init(&pipe);
    opmm.SetMatrix(t.aRealNeg, t.aImag, t.aReal, t.aImag, t.aImagNeg, t.aReal, t.aImag, strideN, strideN, strideN);
    CholeskySolver_CUBE<float> choleskySolver;
    choleskySolver.Init(&opmm, CPOTRF_BLOCK_N);
    choleskySolver.SetOffset(t.aOffset, t.bOffset, t.cOffset, t.offsetIdx);
    choleskySolver.Process(orgN, strideN, 1, 0);
#endif
}

__global__ __aicore__ void cpotrf_kernel(GM_ADDR sync, bool transpose, int orgN, int lda, GM_ADDR A_org, GM_ADDR A_work,
                                         GM_ADDR offset1_gm, GM_ADDR offset2_gm, GM_ADDR offset3_gm,
                                         GM_ADDR offsetIdx_gm, GM_ADDR gather_gm)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if (transpose)
        cholesky_custom<true>(orgN, lda, A_org, A_work, offset1_gm, offset2_gm, offset3_gm, offsetIdx_gm, gather_gm);
    else
        cholesky_custom<false>(orgN, lda, A_org, A_work, offset1_gm, offset2_gm, offset3_gm, offsetIdx_gm, gather_gm);
}

void cpotrf_kernel_do(GM_ADDR sync, bool transpose, int orgN, int lda, GM_ADDR A_org, GM_ADDR A_work,
                      GM_ADDR offset1_gm, GM_ADDR offset2_gm, GM_ADDR offset3_gm, GM_ADDR offsetIdx_gm,
                      GM_ADDR gather_gm, uint32_t numBlocks, void* stream)
{
    cpotrf_kernel<<<numBlocks, nullptr, stream>>>(sync, transpose, orgN, lda, A_org, A_work, offset1_gm, offset2_gm,
                                                  offset3_gm, offsetIdx_gm, gather_gm);
}

#endif  // CPOTRF_MIX_H
