/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>

using namespace AscendC;
using namespace matmul;

#ifndef CANN_OPS_SOLVER_CGEMM
#define CANN_OPS_SOLVER_CGEMM

#define CACHE_LINE_SIZE_FP32 128 // 128 floats per cache line
#define OFFSET_LIST_SIZE 128
#define MMAD_BARRIER_SIZE 2560

enum class ComplexPart { Real, Imag, ImagNeg };

template<typename T>
class CMatmulCustom {
    GlobalTensor<T> aGlobalReal, aGlobalImag, bGlobalReal, bGlobalImag, bGlobalImagNeg, cGlobalReal, cGlobalImag;
    TQue<QuePosition::A1, 2> inQueueA1;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    TBuf<TPosition::CO1> outBufCO1Real;
    TBuf<TPosition::CO1> outBufCO1Imag;
    int N, K;
    int baseM, baseN, baseK;
    int stepK;
    int workNum;
    int workIdx;
    int offsetAList[OFFSET_LIST_SIZE];
    int offsetBList[OFFSET_LIST_SIZE];
    int offsetCList[OFFSET_LIST_SIZE];
public:
    __aicore__ inline CMatmulCustom() {}
    __aicore__ inline void Init(TPipe *pipe) {
        this->stepK = 2;

        pipe->InitBuffer(inQueueA1, 4, 65536);
        pipe->InitBuffer(inQueueB1, 4, 65536);
        pipe->InitBuffer(inQueueA2, 2, 32768);
        pipe->InitBuffer(inQueueB2, 2, 32768);
        pipe->InitBuffer(outBufCO1Real, 65536);
        pipe->InitBuffer(outBufCO1Imag, 65536);
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aGlobalReal, GlobalTensor<T> aGlobalImag, GlobalTensor<T> bGlobalReal, GlobalTensor<T> bGlobalImag, GlobalTensor<T> bGlobalImagNeg, GlobalTensor<T> cGlobalReal, GlobalTensor<T> cGlobalImag, int N, int K) {
        this->N = N;
        this->K = K;
        this->aGlobalReal = aGlobalReal;
        this->aGlobalImag = aGlobalImag;
        this->bGlobalReal = bGlobalReal;
        this->bGlobalImag = bGlobalImag;
        this->bGlobalImagNeg = bGlobalImagNeg;
        this->cGlobalReal = cGlobalReal;
        this->cGlobalImag = cGlobalImag;
    }
    __aicore__ inline void Gemm(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int blockNum, int blockIdx, int trans) {
        int L1M = min(CACHE_LINE_SIZE_FP32, blockM);
        int L1N = min(CACHE_LINE_SIZE_FP32, blockN);
        int L1MBLocks = (blockM / L1M);
        int L1NBLocks = (blockN / L1N);
        int L1BLocks = L1MBLocks * L1NBLocks;

        int averageNum = L1BLocks / blockNum;
        int remainderNum = L1BLocks % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = startIdx; idx < startIdx + calcNum; ++idx) {
            int mIdx = idx / L1NBLocks;
            int nIdx = idx % L1NBLocks;
            Process(offsetA + (trans ? mIdx * L1M : mIdx * L1M * K), offsetB + nIdx * L1N, offsetC + mIdx * L1M * N + nIdx * L1N, L1M, L1N, blockK, trans);
        }
    }
    __aicore__ inline void Gemm2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int blockNum, int blockIdx, int trans) {
        int L1N = min(CACHE_LINE_SIZE_FP32, blockN);
        int L1K = min(CACHE_LINE_SIZE_FP32, blockK);
        int L1NBLocks = (blockN / L1N);
        int L1KBLocks = (blockK / L1K);
        int L1BLocks = L1NBLocks * L1KBLocks;

        int averageNum = L1BLocks / blockNum;
        int remainderNum = L1BLocks % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = startIdx; idx < startIdx + calcNum; ) {
            int nIdx = idx / L1KBLocks;
            int kIdx = idx % L1KBLocks;
            int batch = min(startIdx + calcNum - idx, L1KBLocks - kIdx);
            Process(offsetA + (trans ? kIdx * L1K * K : kIdx * L1K), offsetB + nIdx * L1N + kIdx * L1K * N, offsetC + nIdx * L1N, blockM, L1N, L1K * batch, trans);
            idx += batch;
        }
    }
    __aicore__ inline void GemmWorkClear() {
        workNum = 0;
        workIdx = 0;
    }
    __aicore__ inline void GemmWorkGenerate(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int blockNum, int blockIdx, int trans) {
        int L1M = CACHE_LINE_SIZE_FP32;
        int L1N = CACHE_LINE_SIZE_FP32;
        int L1K = CACHE_LINE_SIZE_FP32;
        int L1MBLocks = (blockM / L1M);
        int L1NBLocks = (blockN / L1N);
        int L1KBLocks = (blockK / L1K);
        int L1BLocks = L1MBLocks * L1NBLocks * L1KBLocks;

        int averageNum = L1BLocks / blockNum;
        int remainderNum = L1BLocks % blockNum;
        int startIdx = blockIdx * averageNum + min(blockIdx, remainderNum);
        int calcNum = averageNum + (blockIdx < remainderNum);

        for (int idx = startIdx; idx < startIdx + calcNum; ++idx) {
            int mIdx = idx / L1KBLocks / L1NBLocks;
            int nIdx = idx / L1KBLocks % L1NBLocks;
            int kIdx = idx % L1KBLocks;
            offsetAList[workNum + idx - startIdx] = offsetA + (trans ? kIdx * L1K * K + mIdx * L1M : mIdx * L1M * K + kIdx * L1K);
            offsetBList[workNum + idx - startIdx] = offsetB + kIdx * L1K * N + nIdx * L1N;
            offsetCList[workNum + idx - startIdx] = offsetC + mIdx * L1M * N + nIdx * L1N;
        }
        workNum += calcNum;
    }
    __aicore__ inline void GemmWorkStep(int remainingSteps, int trans) {
        if (workIdx == workNum || remainingSteps <= 0) return;
        int L1M = CACHE_LINE_SIZE_FP32;
        int L1N = CACHE_LINE_SIZE_FP32;
        int L1K = CACHE_LINE_SIZE_FP32;

        int remainingWork = workNum - workIdx;
        int computeWork = remainingWork / remainingSteps;
        for (int i = 0; i < computeWork; ++i, ++workIdx) {
            Process(offsetAList[workIdx], offsetBList[workIdx], offsetCList[workIdx], L1M, L1N, L1K, trans);
        }
    }
    
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int trans) {
        baseN = min(blockN, 128);
        int nn = 128;
        while (nn < baseN) nn <<= 1;
        
        baseM = min(blockM, 16384 / nn);
        int mm = 16;
        while (mm < baseM) mm <<= 1;
        
        baseK = min(blockK, min(8192 / nn, 8192 / mm));

        // 计算带尾块的总循环次数
        int nBlocks = (blockN + baseN - 1) / baseN;
        int TailN = blockN % baseN; 
        if (!TailN) TailN = baseN;

        int mBlocks = (blockM + baseM - 1) / baseM;
        int TailM = blockM % baseM; 
        if (!TailM) TailM = baseM;

        int kBlocks = (blockK + baseK - 1) / baseK;
        int TailK = blockK % baseK; 
        if (!TailK) TailK = baseK;

        static constexpr uint8_t padList[]{0, 0, 0, 0};

        // 统一循环结构，动态处理尾块大小
        for (int i = 0; i < nBlocks; ++i) {
            int curN = (i == nBlocks - 1) ? TailN : baseN;
            
            for (int j = 0; j < mBlocks; ++j) {
                int curM = (j == mBlocks - 1) ? TailM : baseM;
                
                if (trans == 0) Load3DSetFMatrixCal(1, curM, padList);
                int offsetC_1 = offsetC + j * baseM * N + i * baseN;
                LocalTensor<T> c1LocalReal = outBufCO1Real.Get<T>();
                LocalTensor<T> c1LocalImag = outBufCO1Imag.Get<T>();

                ProcessKBlocks(offsetA, offsetB, offsetC_1, j, i, curM, curN, kBlocks, TailK, trans,
                    c1LocalReal, c1LocalImag);
                CopyOut<ComplexPart::Imag>(c1LocalImag, offsetC_1, curM, curN);
            }
        }
    }

private:
    __aicore__ inline void ProcessKBlocks(int offsetA, int offsetB, int offsetC, int j, int i, int curM, int curN,
        int kBlocks, int TailK, int trans, LocalTensor<T> c1LocalReal, LocalTensor<T> c1LocalImag) {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        for (int k = 0; k < kBlocks; k += stepK) {
            int cnt = min(kBlocks - k, stepK);
            if (trans) Load3DSetFMatrixCal(1, baseK * cnt, padList);
            Load3DSetFMatrixBCal(1, baseK * cnt, padList);

            int offsetA_1 = offsetA + (trans ? j * baseM + k * baseK * K : j * baseM * K + k * baseK);
            int offsetB_1 = offsetB + k * baseK * N + i * baseN;

            CopyInA<ComplexPart::Real>(offsetA_1, cnt, curM, trans);
            CopyInB<ComplexPart::Real>(offsetB_1, cnt, curN);

            LocalTensor<T> a1LocalReal = inQueueA1.DeQue<T>();
            LocalTensor<T> b1LocalReal = inQueueB1.DeQue<T>();
            ProcessRealProducts(a1LocalReal, b1LocalReal, c1LocalReal, k, cnt, kBlocks, TailK, curM, curN, trans);

            CopyInB<ComplexPart::Imag>(offsetB_1, cnt, curN);
            LocalTensor<T> b1LocalImag = inQueueB1.DeQue<T>();
            ProcessImaginaryProducts(a1LocalReal, b1LocalImag, c1LocalImag, k, cnt, kBlocks, TailK, curM, curN, trans);

            inQueueA1.FreeTensor(a1LocalReal);
            inQueueB1.FreeTensor(b1LocalImag);
            CopyInA<ComplexPart::Imag>(offsetA_1, cnt, curM, trans);
            CopyInB<ComplexPart::ImagNeg>(offsetB_1, cnt, curN);
            LocalTensor<T> a1LocalImag = inQueueA1.DeQue<T>();
            LocalTensor<T> b1LocalImagNeg = inQueueB1.DeQue<T>();
            ProcessImaginaryNegProducts(a1LocalImag, b1LocalImagNeg, c1LocalReal, k, cnt, kBlocks, TailK, curM, curN, trans);
            inQueueB1.FreeTensor(b1LocalImagNeg);

            if (k + stepK >= kBlocks) {
                CopyOut<ComplexPart::Real>(c1LocalReal, offsetC, curM, curN);
            }

            ProcessImaginaryTailProducts(a1LocalImag, b1LocalReal, c1LocalImag, k, cnt, kBlocks, TailK, curM, curN, trans);
            inQueueA1.FreeTensor(a1LocalImag);
            inQueueB1.FreeTensor(b1LocalReal);
        }
    }

    __aicore__ inline void ProcessRealProducts(LocalTensor<T> a1LocalReal, LocalTensor<T> b1LocalReal,
        LocalTensor<T> c1LocalReal, int k, int cnt, int kBlocks, int TailK, int curM, int curN, int trans) {
        for (int x = 0; x < cnt; ++x) {
            int curK = (k + x == kBlocks - 1) ? TailK : baseK;
            SplitA(a1LocalReal, x, curK, curM, trans);
            SplitB(b1LocalReal, x, curK, curN);
            Compute(c1LocalReal, !k && !x, 0, curK, curM, curN);
        }
    }

    __aicore__ inline void ProcessImaginaryProducts(LocalTensor<T> a1LocalReal, LocalTensor<T> b1LocalImag,
        LocalTensor<T> c1LocalImag, int k, int cnt, int kBlocks, int TailK, int curM, int curN, int trans) {
        for (int x = 0; x < cnt; ++x) {
            int curK = (k + x == kBlocks - 1) ? TailK : baseK;
            SplitA(a1LocalReal, x, curK, curM, trans);
            SplitB(b1LocalImag, x, curK, curN);
            Compute(c1LocalImag, !k && !x, 0, curK, curM, curN);
        }
    }

    __aicore__ inline void ProcessImaginaryNegProducts(LocalTensor<T> a1LocalImag, LocalTensor<T> b1LocalImagNeg,
        LocalTensor<T> c1LocalReal, int k, int cnt, int kBlocks, int TailK, int curM, int curN, int trans) {
        for (int x = 0; x < cnt; ++x) {
            int curK = (k + x == kBlocks - 1) ? TailK : baseK;
            SplitA(a1LocalImag, x, curK, curM, trans);
            SplitB(b1LocalImagNeg, x, curK, curN);
            Compute(c1LocalReal, 0, (k + x == kBlocks - 1), curK, curM, curN);
        }
    }

    __aicore__ inline void ProcessImaginaryTailProducts(LocalTensor<T> a1LocalImag, LocalTensor<T> b1LocalReal,
        LocalTensor<T> c1LocalImag, int k, int cnt, int kBlocks, int TailK, int curM, int curN, int trans) {
        for (int x = 0; x < cnt; ++x) {
            int curK = (k + x == kBlocks - 1) ? TailK : baseK;
            SplitA(a1LocalImag, x, curK, curM, trans);
            SplitB(b1LocalReal, x, curK, curN);
            Compute(c1LocalImag, 0, (k + x == kBlocks - 1), curK, curM, curN);
        }
    }

    template<ComplexPart cp>
    __aicore__ inline void CopyInA(int offsetA_1, int cnt, int curM, int trans) {
        LocalTensor<T> a1Local = inQueueA1.AllocTensor<T>();
        if (trans == 0) {
            Nd2NzParams intriParamsMatA {
                1,
                static_cast<uint16_t>(curM),
                static_cast<uint16_t>(baseK * cnt),
                0,
                static_cast<uint16_t>(K),
                static_cast<uint16_t>(curM),
                1,
                1
            };
            if constexpr (cp == ComplexPart::Real) {
                DataCopy(a1Local, aGlobalReal[offsetA_1], intriParamsMatA);
            } else if constexpr (cp == ComplexPart::Imag) {
                DataCopy(a1Local, aGlobalImag[offsetA_1], intriParamsMatA);
            }
        } else {
            Nd2NzParams intriParamsMatA {
                1,
                static_cast<uint16_t>(baseK * cnt),
                static_cast<uint16_t>(curM),
                0,
                static_cast<uint16_t>(K),
                static_cast<uint16_t>(baseK * cnt),
                1,
                1
            };
            if constexpr (cp == ComplexPart::Real) {
                DataCopy(a1Local, aGlobalReal[offsetA_1], intriParamsMatA);
            } else if constexpr (cp == ComplexPart::Imag) {
                DataCopy(a1Local, aGlobalImag[offsetA_1], intriParamsMatA);
            }
        }
        inQueueA1.EnQue(a1Local);
    }

    template<ComplexPart cp>
    __aicore__ inline void CopyInB(int offsetB_1, int cnt, int curN) {
        LocalTensor<T> b1Local = inQueueB1.AllocTensor<T>();
        Nd2NzParams intriParamsMatB {
            1,
            static_cast<uint16_t>(baseK * cnt),
            static_cast<uint16_t>(curN),
            0,
            static_cast<uint16_t>(N),
            static_cast<uint16_t>(baseK * cnt),
            1,
            1
        };
        if constexpr (cp == ComplexPart::Real) {
            DataCopy(b1Local, bGlobalReal[offsetB_1], intriParamsMatB);
        } else if constexpr (cp == ComplexPart::Imag) {
            DataCopy(b1Local, bGlobalImag[offsetB_1], intriParamsMatB);
        } else if constexpr (cp == ComplexPart::ImagNeg) {
            DataCopy(b1Local, bGlobalImagNeg[offsetB_1], intriParamsMatB);
        }
        inQueueB1.EnQue(b1Local);
    }

    __aicore__ inline void SplitA(const LocalTensor<T> &a1Local, int splitIdx, int curK, int curM, int trans) {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        if (trans == 0) {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = curK;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                    ((uint64_t)curM << 16) | (uint64_t)curK;
            LoadData<T>(a2Local, a1Local[splitIdx * curM * baseK], loadData3DV2);
        } else {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = curM;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                    ((uint64_t)curK << 16) | (uint64_t)curM;
            loadData3DV2.enTranspose = true;
            LoadData<T>(a2Local, a1Local[splitIdx * baseK * 8], loadData3DV2);
        }
        inQueueA2.EnQue(a2Local);
    }

    __aicore__ inline void SplitB(const LocalTensor<T> &b1Local, int splitIdx, int curK, int curN) {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        if (baseK == 16) {
            LoadData2dTransposeParams loadDataParams {
                0,
                static_cast<uint8_t>(curN / 16),
                1,
                0,
                static_cast<uint16_t>(curN / 16 - 1)
            };
            LoadDataWithTranspose(b2Local, b1Local[splitIdx * baseK * curN], loadDataParams);
        } else {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sBL1NOffset_ = 0;
            loadData3DV2.channelSize = curN;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) |
                                    ((uint64_t)curK << 16) | (uint64_t)curN;
            loadData3DV2.fMatrixCtrl = true;
            LoadData<T>(b2Local, b1Local[splitIdx * baseK * 8], loadData3DV2);
        }
        inQueueB2.EnQue(b2Local);
    }

    __aicore__ inline void Compute(const LocalTensor<T> &c1Local, int first, const int unitFlag, int curK, int curM, int curN) {
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
        MmadParams mmadParams;
        mmadParams.m = curM;
        mmadParams.n = curN;
        mmadParams.k = curK;
        mmadParams.unitFlag = unitFlag ? 3 : 2;
        if (first) {
            Mmad(c1Local, a2Local, b2Local, mmadParams);
        } else {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        }
        if (curM * curN < MMAD_BARRIER_SIZE) 
            PipeBarrier<PIPE_M>();
        inQueueA2.FreeTensor(a2Local);
        inQueueB2.FreeTensor(b2Local);
    }

    template<ComplexPart cp>
    __aicore__ inline void CopyOut(const LocalTensor<T> &c1Local, int offsetC_1, int curM, int curN) {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = curN;
        fixpipeParams.mSize = curM;
        fixpipeParams.srcStride = curM;
        fixpipeParams.dstStride = N;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        
        if constexpr (cp == ComplexPart::Real) {
            Fixpipe(cGlobalReal[offsetC_1], c1Local, fixpipeParams);
        } else if constexpr (cp == ComplexPart::Imag) {
            Fixpipe(cGlobalImag[offsetC_1], c1Local, fixpipeParams);
        }
    }
};

#endif
