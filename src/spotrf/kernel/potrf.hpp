/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifdef DUMMY
#include <tikicpulib.h>
#endif

#include <lib/matrix/matmul/matmul.h>

#include <cstdint>

#include "kernel_operator.h"

using namespace AscendC;
using namespace matmul;

template <typename T>
class LLCustom
{
    GlobalTensor<T> aGlobal;
    GlobalTensor<T> workGlobal2;
    int orgN, padN, blockNum;
    int blockIdx;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 2> inQueueSrc;
    // TBuf<TPosition::VECCALC> workBuf;

   public:
    __aicore__ inline LLCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, TQue<QuePosition::VECIN, 2> inQueueSrc,
                                GlobalTensor<T> &aGlobal, GlobalTensor<T> &workGlobal2, int orgN, int padN,
                                int blockNum)
    {
        this->orgN = orgN;
        this->padN = padN;
        this->aGlobal = aGlobal;
        this->workGlobal2 = workGlobal2;
        this->elementsPerBlock = 32 / sizeof(T);  // 8
        this->inQueueSrc = inQueueSrc;
        this->blockNum = 2 * blockNum;
        this->blockIdx = GetBlockIdx();
        // pipe->InitBuffer(workBuf, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process(int MatAOffset, int offsetM, int offsetN, int realM, int realN)
    {
        int p = 0;
        int offset = 0;
        int length = 0;
        bool flag = true;
        CalcSchedule(realN, realM, p, offset, length, flag);
        if (flag)
        {  // CopyIn
            CopyInPanel(MatAOffset, offset, realM, p);
        }
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        if (flag)
        {  // Cholesky分解计算
            LocalTensor<T> srcLocal = inQueueSrc.DeQue<T>();
            LocalTensor<T> tmpLocal = inQueueSrc.AllocTensor<T>();
            FactorizeColumns(srcLocal, tmpLocal, length, realM);
            CopyOutPanels(srcLocal, MatAOffset, offset, realM, p, length, eventIDVToMTE3);
            inQueueSrc.FreeTensor(srcLocal);
            inQueueSrc.FreeTensor(tmpLocal);
        }
    }

   private:
    __aicore__ inline void CalcSchedule(int realN, int realM, int &p, int &offset, int &length, bool &flag)
    {
        // 非法入参（含除零防护）直接跳过本核处理，flag=false 使 Process 不做拷入/分解/拷出
        if (realM <= 0 || realN <= 0 || blockNum <= 0)
        {
            p = 0;
            offset = 0;
            length = 0;
            flag = false;
            return;
        }
        // 计算正确的拷贝参数：realN 对齐 realM，面板列按 blockNum 均分
        realN = (realN + realM - 1) / realM * realM;  // 200 --> 224

        int count = realN / realM - 1;  // 224 / 32 - 1 = 6
        p = count / blockNum;           // 6 / 2 = 3
        int r = count % blockNum;       // 6 - 3 * 2 = 0
        flag = true;
        offset = 0;
        if (blockIdx < r)
        {
            p++;
            offset = blockIdx * p * realM;
        }
        else
        {
            offset = blockIdx * p * realM + r * realM;
        }
        offset += realM;
        length = (p + 1) * realM;
        if (count && !p)
        {
            flag = false;
        }
        if (!count && blockIdx)
        {
            flag = false;
        }
    }

    __aicore__ inline void CopyInPanel(int MatAOffset, int offset, int realM, int p)
    {
        DataCopyParams copyInParams1{static_cast<uint16_t>(realM), static_cast<uint16_t>(realM / 8),
                                     static_cast<uint16_t>((padN - realM) / 8), static_cast<uint16_t>(p * realM / 8)};
        DataCopyParams copyInParams2{static_cast<uint16_t>(realM), static_cast<uint16_t>(p * realM / 8),
                                     static_cast<uint16_t>((padN - p * realM) / 8), static_cast<uint16_t>(realM / 8)};
        LocalTensor<T> srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopy(srcLocal[realM], aGlobal[MatAOffset + offset], copyInParams2);
        DataCopy(srcLocal, aGlobal[MatAOffset], copyInParams1);
        inQueueSrc.EnQue(srcLocal);
    }

    __aicore__ inline void FactorizeColumns(LocalTensor<T> &srcLocal, LocalTensor<T> &tmpLocal, int length, int realM)
    {
        int offset3 = 0;
        int q = length / 64;
        int r = length % 64;
        for (int j = 0; j < realM; ++j)
        {
            T ljj = srcLocal.GetValue(offset3 + j);
            if (ljj > 0)
            {
                T colScalar = 1 / sqrt(ljj);

                AscendC::Muls(srcLocal[offset3], srcLocal[offset3], colScalar, length);
                AscendC::Brcb(tmpLocal, srcLocal[offset3], realM / 8, {1, 8});
                Muls(tmpLocal, tmpLocal, T(-1), 8 * realM);

                for (int i = 0; i < q; i++)
                {
                    AscendC::MulAddDst(srcLocal[offset3 + length + i * 64], srcLocal[offset3 + i * 64],
                                       tmpLocal[8 * (j + 1)], 64, realM - j - 1,
                                       {1, 1, 0, static_cast<uint8_t>(length / 8), 0, 1});
                }
                if (r)
                {
                    AscendC::MulAddDst(srcLocal[(j + 2) * length - r], srcLocal[offset3 + length - r],
                                       tmpLocal[8 * (j + 1)], r, realM - j - 1,
                                       {1, 1, 0, static_cast<uint8_t>(length / 8), 0, 1});
                }
            }
            else
            {
                j = realM;
            }
            offset3 += length;
        }
    }

    __aicore__ inline void CopyOutPanels(LocalTensor<T> &srcLocal, int MatAOffset, int offset, int realM, int p,
                                         int length, int32_t eventIDVToMTE3)
    {
        DataCopyParams copyOutParams1{static_cast<uint16_t>(realM), static_cast<uint16_t>(realM / 8),
                                      static_cast<uint16_t>(p * realM / 8), static_cast<uint16_t>((padN - realM) / 8)};
        DataCopyParams copyOutParams2{static_cast<uint16_t>(realM), static_cast<uint16_t>(p * realM / 8),
                                      static_cast<uint16_t>(realM / 8), static_cast<uint16_t>((padN - p * realM) / 8)};

        // CopyOut
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        if (blockIdx == 0)
        {
            DataCopy(aGlobal[MatAOffset], srcLocal, copyOutParams1);
        }
        DataCopy(aGlobal[MatAOffset + offset], srcLocal[realM], copyOutParams2);

        int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);

        int len = length * realM;
        Muls(srcLocal, srcLocal, T(-1), len);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(workGlobal2[MatAOffset + offset], srcLocal[realM], copyOutParams2);
    }
};
