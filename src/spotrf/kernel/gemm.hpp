/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class MatmulCustom_01
{
    GlobalTensor<T> aGlobal, bGlobal, cGlobal;
    TQue<QuePosition::A1, 2> inQueueA1;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1;
    int N, K;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int baseM, baseN, baseK;
    int stepM, stepN;
    int stepKa, stepKb;
    int sizeA, sizeB;
    int alignedN;
    int stepK;
    int workNum = 0;
    int workIdx = 0;
    int total_work = 0;
    int last_offsetC = -1;

   public:
    __aicore__ inline MatmulCustom_01() {}
    __aicore__ inline int getTotal() { return this->total_work; }
    __aicore__ inline void Init(TPipe *pipe)
    {
        this->stepK = 4;

        pipe->InitBuffer(inQueueA1, 2, 131072);
        pipe->InitBuffer(inQueueB1, 2, 131072);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1, 131072);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aGlobal, GlobalTensor<T> bGlobal, GlobalTensor<T> cGlobal, int N,
                                     int K)
    {
        this->N = N;
        this->K = K;
        this->aGlobal = aGlobal;
        this->bGlobal = bGlobal;
        this->cGlobal = cGlobal;
    }
    __aicore__ inline void Gemm(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int blockNum,
                                int blockIdx)
    {
        int L1M = min(128, blockM);
        int L1N = min(128, blockN);
        int L1MBLocks = (blockM / L1M);
        int L1NBLocks = (blockN / L1N);
        int L1BLocks = L1MBLocks * L1NBLocks;

        int averageNum = L1BLocks / blockNum;
        int remainderNum = L1BLocks % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = startIdx; idx < startIdx + calcNum; ++idx)
        {
            int mIdx = idx / L1NBLocks;
            int nIdx = idx % L1NBLocks;
            Process(offsetA + mIdx * L1M * K, offsetB + nIdx * L1N, offsetC + mIdx * L1M * N + nIdx * L1N, L1M, L1N,
                    blockK);
        }
    }
    __aicore__ inline void Gemm2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK,
                                 int blockNum, int blockIdx)
    {
        int L1N = min(128, blockN);
        int L1K = min(128, blockK);
        int L1NBLocks = (blockN / L1N);
        int L1KBLocks = (blockK / L1K);
        int L1BLocks = L1NBLocks * L1KBLocks;

        int averageNum = L1BLocks / blockNum;
        int remainderNum = L1BLocks % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = startIdx; idx < startIdx + calcNum;)
        {
            int nIdx = idx / L1KBLocks;
            int kIdx = idx % L1KBLocks;
            int batch = min(startIdx + calcNum - idx, L1KBLocks - kIdx);
            Process(offsetA + kIdx * L1K, offsetB + nIdx * L1N + kIdx * L1K * N, offsetC + nIdx * L1N, blockM, L1N,
                    L1K * batch);
            idx += batch;
        }
    }
    __aicore__ inline void GemmWorkClear()
    {
        workNum = 0;
        workIdx = 0;
    }

    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK)
    {
        int nFullBlocks, TailN, mFullBlocks, TailM, kBlocks, TailK;
        CalcTileShape(offsetA, offsetB, offsetC, blockM, blockN, blockK, nFullBlocks, TailN, mFullBlocks, TailM,
                      kBlocks, TailK);
        GemmFullBlocks(offsetA, offsetB, offsetC, nFullBlocks, mFullBlocks, kBlocks, TailK);
        if (!TailN) return;
        GemmTailN(offsetA, offsetB, offsetC, nFullBlocks, TailN, mFullBlocks, kBlocks, TailK);
        if (!TailM) return;
        GemmTailM(offsetA, offsetB, offsetC, mFullBlocks, kBlocks, TailK, TailM);
    }

   private:
    __aicore__ inline void CalcTileShape(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK,
                                         int &nFullBlocks, int &TailN, int &mFullBlocks, int &TailM, int &kBlocks,
                                         int &TailK)
    {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;  // 128
        this->singleCoreN = blockN;  // 128
        this->singleCoreK = blockK;  // 128

        baseN = min(singleCoreN, 256);  // 128
        int nn = 128;
        while (nn < baseN) nn <<= 1;
        baseM = min(singleCoreM, 32768 / nn);  // 128
        int mm = 16;
        while (mm < baseM) mm <<= 1;                          // 128
        baseK = min(singleCoreK, min(8192 / nn, 8192 / mm));  // 64

        // 除零防护：任一分块维度退化为 0 时跳过本核 GEMM（出参置零，Process 据此短路返回）
        if (baseN <= 0 || baseM <= 0 || baseK <= 0)
        {
            nFullBlocks = 0;
            TailN = 0;
            mFullBlocks = 0;
            TailM = 0;
            kBlocks = 0;
            TailK = 0;
            return;
        }

        nFullBlocks = singleCoreN / baseN;                        // 1
        TailN = singleCoreN % baseN;                              // 0
        alignedN = baseN <= 32 ? baseN : (baseN + 63) / 64 * 64;  // 128

        mFullBlocks = singleCoreM / baseM;  // 1
        TailM = singleCoreM % baseM;        // 0

        kBlocks = (singleCoreK + baseK - 1) / baseK;  // 2

        TailK = singleCoreK % baseK;  // 0
        if (!TailK) TailK = baseK;    // 64

        sizeA = baseM * baseK;     // 128 * 64
        sizeB = baseK * alignedN;  // 64 * 128
    }

    __aicore__ inline void GemmFullBlocks(int offsetA, int offsetB, int offsetC, int nFullBlocks, int mFullBlocks,
                                          int kBlocks, int TailK)
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        for (int i = 0; i < nFullBlocks; ++i)
        {
            for (int j = 0; j < mFullBlocks; ++j)
            {
                int offsetC_1 = offsetC + j * baseM * N + i * baseN;
                LocalTensor<T> c1Local = outBufCO1.Get<T>();
                for (int k = 0; k < kBlocks; k += stepK)
                {
                    int cnt = min(kBlocks - k, stepK);  // 2
                    Load3DSetFMatrixCal(1, baseK * cnt, padList);
                    Load3DSetFMatrixBCal(1, baseK * cnt, padList);

                    int offsetB_1 = offsetB + k * baseK * N + i * baseN;
                    CopyInB(offsetB_1, cnt);
                    LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
                    SplitB(b1Local, 0, k == kBlocks - 1 ? TailK : baseK);

                    CopyInA_01(offsetA + k * baseK * N + j * baseN, cnt);
                    LocalTensor<T> a1Transposed = inQueueA1.DeQue<T>();
                    SplitA(a1Transposed, 0, k == kBlocks - 1 ? TailK : baseK);

                    Compute(c1Local, !k, k == kBlocks - 1, k == kBlocks - 1 ? TailK : baseK);
                    for (int x = 1; x < cnt; ++x)
                    {
                        SplitB(b1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                        SplitA(a1Transposed, x, k + x == kBlocks - 1 ? TailK : baseK);
                        Compute(c1Local, !k && !x, k + x == kBlocks - 1, k + x == kBlocks - 1 ? TailK : baseK);
                    }

                    inQueueB1.FreeTensor(b1Local);
                    inQueueA1.FreeTensor(a1Transposed);
                }
                CopyOut(c1Local, offsetC_1);
            }
        }
    }

    __aicore__ inline void GemmKBlockLoop(int offsetA_1, int offsetB_1, const LocalTensor<T> &c1Local, int kBlocks,
                                          int TailK)
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        for (int k = 0; k < kBlocks; k += stepK)
        {
            int cnt = min(kBlocks - k, stepK);
            Load3DSetFMatrixBCal(1, baseK * cnt, padList);
            CopyInA(offsetA_1 + k * baseK, cnt);
            CopyInB(offsetB_1 + k * baseK * N, cnt);
            LocalTensor<T> a1Local = inQueueA1.DeQue<T>();
            LocalTensor<T> b1Local = inQueueB1.DeQue<T>();
            for (int x = 0; x < cnt; ++x)
            {
                SplitA(a1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                SplitB(b1Local, x, k + x == kBlocks - 1 ? TailK : baseK);
                Compute(c1Local, !k && !x, k + x == kBlocks - 1, k + x == kBlocks - 1 ? TailK : baseK);
            }
            inQueueA1.FreeTensor(a1Local);
            inQueueB1.FreeTensor(b1Local);
        }
    }

    __aicore__ inline void GemmTailN(int offsetA, int &offsetB, int &offsetC, int nFullBlocks, int TailN,
                                     int mFullBlocks, int kBlocks, int TailK)
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        offsetB += nFullBlocks * baseN;
        offsetC += nFullBlocks * baseN;
        baseN = TailN;
        alignedN = baseN;
        if (alignedN > 32) alignedN = (alignedN + 63) / 64 * 64;
        sizeB = baseK * alignedN;

        Load3DSetFMatrixCal(1, baseM, padList);
        for (int j = 0; j < mFullBlocks; ++j)
        {
            int offsetC_1 = offsetC + j * baseM * N;
            LocalTensor<T> c1Local = outBufCO1.Get<T>();
            GemmKBlockLoop(offsetA + j * baseM * K, offsetB, c1Local, kBlocks, TailK);
            CopyOut(c1Local, offsetC_1);
        }
    }

    __aicore__ inline void GemmTailM(int offsetA, int offsetB, int offsetC, int mFullBlocks, int kBlocks, int TailK,
                                     int TailM)
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        int _offsetA = offsetA + mFullBlocks * baseM * K;
        int _offsetC = offsetC + mFullBlocks * baseM * N;
        int _baseM = baseM;
        baseM = TailM;
        sizeA = baseM * baseK;
        Load3DSetFMatrixCal(1, baseM, padList);
        int offsetC_1 = _offsetC;
        LocalTensor<T> c1Local = outBufCO1.Get<T>();
        GemmKBlockLoop(_offsetA, offsetB, c1Local, kBlocks, TailK);
        CopyOut(c1Local, offsetC_1);

        baseM = _baseM;
        sizeA = baseM * baseK;
    }

   private:
    __aicore__ inline void CopyInA(int offsetA_1, int cnt)
    {
        LocalTensor<T> a1Local = inQueueA1.AllocTensor<T>();

        Nd2NzParams intriParamsMatA{1,
                                    static_cast<uint16_t>(baseM),
                                    static_cast<uint16_t>(baseK * cnt),
                                    0,
                                    static_cast<uint16_t>(K),
                                    static_cast<uint16_t>(baseM),
                                    1,
                                    1};
        DataCopy(a1Local, aGlobal[offsetA_1], intriParamsMatA);
        inQueueA1.EnQue(a1Local);
    }

    __aicore__ inline void CopyInA_01(int offsetA_1, int cnt)
    {
        LocalTensor<T> a1Local = inQueueA1.AllocTensor<T>();

        Nd2NzParams intriParamsMatA{1, static_cast<uint16_t>(baseK * cnt),
                                    // static_cast<uint16_t>(alignedN),
                                    static_cast<uint16_t>(baseM), 0, static_cast<uint16_t>(N),
                                    static_cast<uint16_t>(baseK * cnt), 1, 1};
        DataCopy(a1Local, aGlobal[offsetA_1], intriParamsMatA);
        inQueueA1.EnQue(a1Local);
    }

    __aicore__ inline void CopyInB(int offsetB_1, int cnt)
    {
        LocalTensor<T> b1Local = inQueueB1.AllocTensor<T>();
        Nd2NzParams intriParamsMatB{1,
                                    static_cast<uint16_t>(baseK * cnt),
                                    static_cast<uint16_t>(alignedN),
                                    0,
                                    static_cast<uint16_t>(N),
                                    static_cast<uint16_t>(baseK * cnt),
                                    1,
                                    1};
        DataCopy(b1Local, bGlobal[offsetB_1], intriParamsMatB);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA(const LocalTensor<T> &a1Local, int splitIdx, int TailK)
    {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        if (baseM == -1)
        {
            LoadData2DParams loadL0AParams;
            loadL0AParams.repeatTimes = baseK / 8;
            loadL0AParams.srcStride = 1;
            loadL0AParams.ifTranspose = true;
            LoadData(a2Local, a1Local[splitIdx * sizeA], loadL0AParams);
        }
        else
        {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = baseN;
            loadData3DV2.extConfig =
                ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) | ((uint64_t)TailK << 16) | (uint64_t)baseN;
            loadData3DV2.enTranspose = true;
            LoadData<T>(a2Local, a1Local[splitIdx * baseK * 8], loadData3DV2);
        }
        inQueueA2.EnQue(a2Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<T> &b1Local, int splitIdx, int TailK)
    {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        if (baseK == 16)
        {
            LoadData2dTransposeParams loadDataParams{0, static_cast<uint8_t>(baseN / 16), 1, 0,
                                                     static_cast<uint16_t>(baseN / 16 - 1)};
            LoadDataWithTranspose(b2Local, b1Local[splitIdx * sizeB], loadDataParams);
        }
        else
        {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sBL1NOffset_ = 0;
            loadData3DV2.channelSize = baseN;
            loadData3DV2.extConfig =
                ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) | ((uint64_t)TailK << 16) | (uint64_t)baseN;
            loadData3DV2.fMatrixCtrl = true;
            LoadData<T>(b2Local, b1Local[splitIdx * baseK * 8], loadData3DV2);
        }
        inQueueB2.EnQue(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<T> &c1Local, int first, const int unitFlag, const int TailK)
    {
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();

        LocalTensor<T> ngLocal = inQueueA2.AllocTensor<T>();
        MmadParams mmadParams2;
        mmadParams2.m = baseM;
        mmadParams2.n = TailK;
        mmadParams2.k = TailK;
        mmadParams2.unitFlag = unitFlag ? 3 : 2;
        // Mmad(a2Local, a2Local, ngLocal, mmadParams2);

        MmadParams mmadParams;
        mmadParams.m = baseM;
        mmadParams.n = baseN;
        mmadParams.k = TailK;
        mmadParams.unitFlag = unitFlag ? 3 : 2;
        if (first)
        {
            Mmad(c1Local, a2Local, b2Local, mmadParams);
        }
        else
        {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        }
        if (baseM * baseN >> 8 < 10) PipeBarrier<PIPE_M>();
        inQueueA2.FreeTensor(a2Local);
        inQueueA2.FreeTensor(ngLocal);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void CopyOut(const LocalTensor<T> &c1Local, int offsetC_1)
    {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = baseN;
        fixpipeParams.mSize = baseM;
        fixpipeParams.srcStride = baseM;
        fixpipeParams.dstStride = N;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        Fixpipe(cGlobal[offsetC_1], c1Local, fixpipeParams);
    }
};
