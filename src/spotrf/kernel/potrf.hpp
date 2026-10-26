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

#include <cstdint>

#include "kernel_operator.h"
#include <lib/matrix/matmul/matmul.h>

using namespace AscendC;
using namespace matmul;

#define debug(x) printf("%s: %d\n", #x, x)
#define debugf(x) printf("%s: %f\n", #x, x)
#define puts(x) printf("%s\n", x)


template<typename T>
class LLCustom {                          
    GlobalTensor<T> aGlobal;
    GlobalTensor<T> workGlobal2;
    int orgN, padN,blockNum;
    int blockIdx;
    int blockM, blockN;
    int realM;
    int elementsPerBlock;
    static constexpr int TILE_LENGTH = 8192;
    TQue<QuePosition::VECIN, 2> inQueueSrc;
    //TBuf<TPosition::VECCALC> workBuf;
    
public:
    __aicore__ inline LLCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe, 
                               TQue<QuePosition::VECIN, 2> inQueueSrc, 
                               GlobalTensor<T> &aGlobal, GlobalTensor<T> &workGlobal2,
                               int orgN, int padN,int blockNum) {
        this->orgN = orgN;
        this->padN = padN;
        this->aGlobal = aGlobal;
        this->workGlobal2 = workGlobal2;
        this->elementsPerBlock = 32 / sizeof(T);// 8
        this->inQueueSrc = inQueueSrc;
        this->blockNum = 2 * blockNum;
        this->blockIdx = GetBlockIdx();
        //pipe->InitBuffer(workBuf, TILE_LENGTH * sizeof(T));
    }
    
__aicore__ inline void Process(int MatAOffset, int offsetM, int offsetN, 
                                      int realM, int realN) {
    // 计算正确的拷贝参数
    realN = (realN + realM - 1) / realM * realM;//200 --> 224
    LocalTensor<T> srcLocal;
    LocalTensor<T> tmpLocal;

    int count = realN / realM - 1;// 224 / 32 - 1 = 6
    int p = count / blockNum;// 6 / 2 = 3
    int r = count % blockNum;// 6 - 3 * 2 = 0
    bool flag = true;
    int offset = 0;
    if(blockIdx < r){
        p++;
        offset = blockIdx * p * realM;
    }else{
        offset = blockIdx * p * realM + r * realM;
    }
    offset += realM;
    int lenth = (p + 1) * realM;
    if(count && !p){
        flag = false;
    }
    if(!count && blockIdx){
        flag = false;
    }
    DataCopyParams copyInParams1 {
        static_cast<uint16_t>(realM),
        static_cast<uint16_t>(realM/8),
        static_cast<uint16_t>((padN - realM)/8),
        static_cast<uint16_t>(p * realM/8)
    };
    DataCopyParams copyInParams2 {
        static_cast<uint16_t>(realM),
        static_cast<uint16_t>(p * realM/8),
        static_cast<uint16_t>((padN - p * realM)/8),
        static_cast<uint16_t>(realM/8)
    };
    DataCopyParams copyOutParams1 {
        static_cast<uint16_t>(realM),
        static_cast<uint16_t>(realM/8),
        static_cast<uint16_t>(p * realM/8),
        static_cast<uint16_t>((padN - realM)/8)
    };
    DataCopyParams copyOutParams2 {
        static_cast<uint16_t>(realM),
        static_cast<uint16_t>(p * realM/8),
        static_cast<uint16_t>(realM/8),
        static_cast<uint16_t>((padN - p * realM)/8)
    };

    if(flag){   // CopyIn
        srcLocal = inQueueSrc.AllocTensor<T>();
        DataCopy(srcLocal[realM], aGlobal[MatAOffset + offset], copyInParams2);
        DataCopy(srcLocal, aGlobal[MatAOffset], copyInParams1);
        inQueueSrc.EnQue(srcLocal);
    }
    int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    if(flag){   // Cholesky分解计算
        srcLocal = inQueueSrc.DeQue<T>();
        tmpLocal = inQueueSrc.AllocTensor<T>();
        int offset3 = 0;
        
        int q = lenth / 64;
        int r = lenth % 64;
        for (int j = 0; j < realM; ++j) {
            T ljj = srcLocal.GetValue(offset3 + j);
            if(ljj > 0){
                T colScalar = 1 / sqrt(ljj);

                AscendC::Muls(srcLocal[offset3],srcLocal[offset3], colScalar,lenth);
                AscendC::Brcb(tmpLocal,srcLocal[offset3],realM/8,{1,8});
                Muls(tmpLocal,tmpLocal,T(-1),8 * realM);
                
                for(int i = 0;i < q; i++){
                    AscendC::MulAddDst(srcLocal[offset3 + lenth + i * 64],srcLocal[offset3 + i * 64],tmpLocal[8 * (j + 1)], 
                                                64, realM - j - 1, {1,1,0,static_cast<uint8_t>(lenth / 8),0,1});
                }
                if(r){
                    AscendC::MulAddDst(srcLocal[(j+2) * lenth - r],srcLocal[offset3 + lenth - r],tmpLocal[8 * (j + 1)], 
                                            r, realM - j - 1, {1,1,0,static_cast<uint8_t>(lenth / 8),0,1});
                }
            }else{
                j = realM;
            }
            offset3 += lenth;
        }

        // CopyOut   
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        
        if(blockIdx == 0){
            DataCopy(aGlobal[MatAOffset], srcLocal, copyOutParams1);
        }
        DataCopy(aGlobal[MatAOffset + offset], srcLocal[realM], copyOutParams2);

        int32_t eventIDMTE3ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3ToV);

        int len = lenth * realM;
        Muls(srcLocal,srcLocal,T(-1),len);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopy(workGlobal2[MatAOffset + offset], srcLocal[realM], copyOutParams2);

        inQueueSrc.FreeTensor(srcLocal);
        inQueueSrc.FreeTensor(tmpLocal);
    }
    }
};
