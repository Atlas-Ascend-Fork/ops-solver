/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SPOTRF_MIX_H
#define SPOTRF_MIX_H

#include <cstdint>

#include "kernel_operator.h"
#include "kernel/potrf.hpp"
#include "kernel/gemm.hpp"
#include "spotrf_launchers.hpp"

using namespace AscendC;
using namespace matmul;

#define ceil(x, y) (((x) + (y) - 1) / (y) * (y))
#define vec_wait_cube 0x1
#define vec_wait_vec 0x2
#define cube_wait_vec 0x3
#define cube_wait_cube 0x4

template<typename T>
class PadCustom {
    static constexpr int TILE_LENGTH = 8192;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> queBind;
    TQue<QuePosition::VECIN, 1> Que;
    TQue<QuePosition::VECOUT, 1> Que2;
    TQue<QuePosition::VECIN, 1> localQue;
    GlobalTensor<T> srcGlobal, dstGlobal;
    GlobalTensor<uint32_t> gather;
    int N;
    int alignedN;
    int srcN;
    int dstN;
    int elementsPerBlock;
    int padN;
public:
    __aicore__ inline PadCustom() {}
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe) {
        pipe->InitBuffer(queBind, 2, 2 * 8192 * sizeof(T));
        this->gather = gather;
    }
    __aicore__ inline void Init2(TBufPool<TPosition::VECCALC, 16> *pipe,GlobalTensor<uint32_t> gather) {
        pipe->InitBuffer(Que, 1, 8192 * sizeof(T));
        pipe->InitBuffer(Que2, 1, 8192 * sizeof(T));
        pipe->InitBuffer(localQue, 1, 128 * 64 * sizeof(uint32_t));
        this->gather = gather;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> srcGlobal, GlobalTensor<T> dstGlobal, int N, int srcN, int dstN) {
        this->srcGlobal = srcGlobal;
        this->dstGlobal = dstGlobal;
        this->N = N;
        this->srcN = srcN;
        this->dstN = dstN;
        elementsPerBlock = 32 / sizeof(T);
        this->alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        padN = (elementsPerBlock - (N % elementsPerBlock)) % elementsPerBlock;
    }
    __aicore__ inline void PadCustomWork(int M, int p, int id) {
        int count = M / 8;
        int r = M % 8;
        for(int i = 0;i < count;++i){
            if(i % p == id){
                Process((i * 8) * (srcN + 1), (i * 8) * (dstN + 1), 8);
            }
            N -= 8;
        }
        if(count % p == id){
            Process((count * 8) * (srcN + 1), (count * 8) * (dstN + 1), r);
        }
    }
    __aicore__ inline void PadCustomWork2(int M, int p, int id) {
        int m_cubes = M / 128;
        int m1 = M % 128;
        int n_cubes = N / 64;
        int n1 = N % 64;
        int idx = -1;
        for(int i = 0;i < m_cubes + 1;i++){
            int t = min(2 * i + 1, n_cubes);
            for(int j = 0;j <= t;j++){
                idx++;
                if(p > 0 && idx % p == id){
                    Process2(i,j,(i == m_cubes)? m1 : 128,(j == n_cubes)? n1 : 64);
                }
            }
        }
    }
    __aicore__ inline void PadCustomWork3(int M, int p, int id) {
        int m_cubes = M / 128;
        int m1 = M % 128;
        int n_cubes = N / 64;
        int n1 = N % 64;
        int idx = -1;
        for(int i = 0;i < m_cubes + 1;i++){
            int t = min(2 * i, n_cubes);
            for(int j = t;j < n_cubes + 1;j++){
                idx++;
                if(p > 0 && idx % p == id){
                    Process2(i,j,(i == m_cubes)? m1 : 128,(j == n_cubes)? n1 : 64);
                }
            }
        }
    }
    __aicore__ inline void Process(int srcOffset, int dstOffset, int lines) {
        int alignedN = (N + elementsPerBlock - 1) / elementsPerBlock * elementsPerBlock;
        int linesPerIter = 2 * TILE_LENGTH / alignedN;
        int loopCount = lines / linesPerIter;
        for (int i = 0; i < loopCount; ++i) {
            CopyLines(srcOffset, dstOffset, linesPerIter);
            srcOffset += linesPerIter * srcN;
            dstOffset += linesPerIter * dstN;
        }
        lines -= linesPerIter * loopCount;
        if (!lines) return;
        CopyLines(srcOffset, dstOffset, lines);
    }
    __aicore__ inline void Process2(int idxm, int idxn, int M, int N) {
        if(M <= 0 || N <= 0){
            return;
        }

        LocalTensor<T> bindLocal = Que.AllocTensor<T>();
        LocalTensor<T> bindLocal2 = Que2.AllocTensor<T>();
        LocalTensor<uint32_t> local = localQue.AllocTensor<uint32_t>();
        DataCopy(local,gather,128 * 64);
        DataCopyExtParams copyInParams {
            static_cast<uint16_t>(M),
            static_cast<uint16_t>(N * sizeof(T)),
            static_cast<uint16_t>((srcN - N) * sizeof(T)),
            static_cast<uint16_t>((64 - N) / 8),
            0
        };

        int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        int32_t eventIDVToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        DataCopyPadExtParams<T> padParams {false, 0, static_cast<uint8_t>(64), 0};
        DataCopyPad(bindLocal, srcGlobal[idxm * 128 * srcN + idxn * 64], copyInParams, padParams);
        
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventIDMTE2ToV);

        AscendC::Gather(bindLocal2,bindLocal,local, 0, 64 * 128);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(N),
            static_cast<uint16_t>(M * sizeof(T)),
            static_cast<uint16_t>((128 - M) / 8),
            static_cast<uint16_t>((dstN - M) * sizeof(T)),
            0
        };
        DataCopyPad(dstGlobal[idxn * 64 * dstN + idxm * 128], bindLocal2, copyOutParams);
        Que.FreeTensor(bindLocal);
        Que2.FreeTensor(bindLocal2);
        localQue.FreeTensor(local);
    }
private:
    __aicore__ inline void CopyLines(int srcOffset, int dstOffset, int lines) {
        LocalTensor<T> bindLocal = queBind.AllocTensor<T>();
        DataCopyExtParams copyInParams {
            static_cast<uint16_t>(lines),
            static_cast<uint16_t>(N * sizeof(T)),
            static_cast<uint16_t>((srcN - N) * sizeof(T)),
            0,
            0
        };
        DataCopyPadExtParams<T> padParams {false, 0, static_cast<uint8_t>(padN), 0};
        DataCopyPad(bindLocal, srcGlobal[srcOffset], copyInParams, padParams);
        queBind.EnQue(bindLocal);
        bindLocal = queBind.DeQue<T>();
        DataCopyExtParams copyOutParams{
            static_cast<uint16_t>(lines),
            static_cast<uint16_t>(N * sizeof(T)),
            0,
            static_cast<uint16_t>((dstN - N) * sizeof(T)),
            0
        };
        DataCopyPad(dstGlobal[dstOffset], bindLocal, copyOutParams);
        queBind.FreeTensor(bindLocal);
    }
};

template<typename T>
class CholeskySolver {
    LLCustom<T> opLL;
    GlobalTensor<T> aGlobal;
    GlobalTensor<T> workGlobal2;
    TQue<QuePosition::VECIN, 2> inQueueSrc;
    TBufPool<TPosition::VECCALC, 16> *pipe;
    int orgN, padN,blockNum;
    
public:
    __aicore__ inline CholeskySolver() {}
    
    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe,
                               GlobalTensor<T> aGlobal,GlobalTensor<T> workGlobal2,
                               int orgN, int padN,int blockNum) {
        this->pipe = pipe;
        this->aGlobal = aGlobal;
        this->workGlobal2 = workGlobal2;
        this->orgN = orgN;
        this->padN = padN;
        this->blockNum = blockNum;

        pipe->InitBuffer(inQueueSrc, 2, 4 * 8192 * sizeof(T)); //32 * 32 * 4

        opLL.Init(pipe, inQueueSrc, aGlobal,workGlobal2, orgN, padN,blockNum);
    }
    
    __aicore__ inline void Process(int offsetM, int offsetN,
                              int blockM, int blockN,int flag) {
        

        int srcOffset = offsetM * padN + offsetN;  
    
        int actualM = blockM;
        int actualN = min(blockN,orgN - offsetN);
    
        if(flag == 1){
            opLL.Process(srcOffset, offsetM, offsetN, actualM, actualN);
        }
    }
};

template<typename T>
class GemmSolver{
    MatmulCustom_01<float> opmm;
    int offsetA,offsetB,offsetC;
    int tempA,tempB,tempC;
    int blockNum,blockIdx;
    int idx = 0;
    GlobalTensor<int> aOffset;
    GlobalTensor<int> bOffset;
    GlobalTensor<int> cOffset;
    GlobalTensor<int> offsetIdx;
    int globalIdx = 0;
    int workNum,workIdx;
public:
    __aicore__ inline GemmSolver(){}
    __aicore__ inline void Init(MatmulCustom_01<float> opmm){
        this->opmm = opmm;
    }
    __aicore__ inline void SetOffset(GlobalTensor<int> aOffset, GlobalTensor<int> bOffset, GlobalTensor<int> cOffset, GlobalTensor<int> offsetIdx){
        this->aOffset = aOffset;
        this->bOffset = bOffset;
        this->cOffset = cOffset;
        this->offsetIdx = offsetIdx;
    }
    __aicore__ inline void SetMatrix(GlobalTensor<T> aGlobal, GlobalTensor<T> bGlobal, GlobalTensor<T> cGlobal, int N, int K,int blockNum,int blockIdx){
        this->blockNum = blockNum;
        this->blockIdx = blockIdx;
        opmm.SetMatrix(aGlobal, bGlobal, cGlobal, N, K);
    }
    __aicore__ inline void GemmWorkGenerate(int step){
        workNum = offsetIdx.GetValue(step);
        if(step)
            globalIdx += offsetIdx.GetValue(step-1);
        workIdx = 0;
    }
    __aicore__ inline void WorkStep(int remainingSteps){
        if (workIdx == workNum || remainingSteps <= 0) return;
        int blockNum = GetBlockNum();
        int blockIdx = GetBlockIdx();
        
        int remainingWork = workNum - workIdx;
        int computeWork = remainingWork / remainingSteps;
        int p = computeWork / blockNum;
        int r = computeWork % blockNum;
        
        int startIdx = globalIdx + workIdx;

        for(int i = 0;i < computeWork;++i){
            if((startIdx + i) % blockNum == blockIdx){
                opmm.Process(aOffset.GetValue(startIdx + i), bOffset.GetValue(startIdx + i), cOffset.GetValue(startIdx + i), 128, 128, 128);
            }
        }
        workIdx += computeWork;
    }

    __aicore__ inline void update(int offset1,int offset2,int panel_size,int strideN){
        int r = (strideN - offset2) % 128;
        int blocks = (strideN - offset2 + 128 - r) / 128;

        tempC = offset2 * strideN + offset2 - 128 + r;
        tempA = offset1 * strideN + offset2;
        tempB = tempA - 128 + r;
        int p = blocks / blockNum;
        r = blocks % blockNum;
        offsetA = tempA;
        if(blockIdx < r){
            p++;
            offsetC = tempC + blockIdx * p * 128;
            offsetB = tempB + blockIdx * p * 128;
        }else{
            offsetC = tempC + blockIdx * p * 128 + r * 128;
            offsetB = tempB + blockIdx * p * 128 + r * 128;
        }
        if(p){
            opmm.Process(
                offsetA,
                offsetB,
                offsetC,
                panel_size,       // blockM
                p * 128,          // blockN
                panel_size        // blockK
            );
        }
    }
};

// custom_llt 各子步骤共用的 GM 张量视图
struct CustomLltTensors {
    GlobalTensor<float> aGlobalOrg;
    GlobalTensor<float> aGlobal;
    GlobalTensor<int> aOffset;
    GlobalTensor<int> bOffset;
    GlobalTensor<int> cOffset;
    GlobalTensor<int> offsetIdx;
    GlobalTensor<uint32_t> gather;
};

__aicore__ inline void CustomLltInitData(GM_ADDR A_org, GM_ADDR A_work, GM_ADDR Gather, __gm__ uint32_t *offset1,
                                         __gm__ uint32_t *offset2, __gm__ uint32_t *offset3, __gm__ uint32_t *OffsetIdx,
                                         CustomLltTensors &tensors) {
    tensors.aGlobalOrg.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_org));
    tensors.aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(A_work));
    tensors.aOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(offset1));
    tensors.bOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(offset2));
    tensors.cOffset.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(offset3));
    tensors.offsetIdx.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(OffsetIdx));
    tensors.gather.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(Gather));
}

// vector 核：输入矩阵 pad/转置、分块 Cholesky 分解、结果还原
__aicore__ inline void CustomLltVector(TPipe &pipe, TBufPool<TPosition::VECCALC, 16> &tbufPool, int orgM, int orgN,
                                       int blockN, int lda, bool isTranse, int strideN, int blockNum, int blockIdx,
                                       CustomLltTensors &tensors) {
    int N = orgN;
    int cnt = 0;
    if(blockN == 32){
        cnt = strideN / 32;
    }else{
        cnt = strideN / 64;
    }
    
    pipe.InitBufPool(tbufPool, 192 * 1024 * 4);

    PadCustom<float> oppad;
    oppad.SetMatrix(tensors.aGlobalOrg, tensors.aGlobal, N, lda, strideN);
    if(isTranse){
        oppad.Init2(&tbufPool,tensors.gather);
        oppad.PadCustomWork2(orgM, blockNum << 1, blockIdx);
    }else{
        oppad.Init(&tbufPool);
        oppad.PadCustomWork(orgM, blockNum << 1, blockIdx);
    }
    tbufPool.Reset();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(vec_wait_vec);
    CrossCoreWaitFlag(vec_wait_vec);
    CholeskySolver<float> choleskySolver;
    choleskySolver.Init(&tbufPool, tensors.aGlobal, tensors.aGlobalOrg, N, strideN, blockNum);
    choleskySolver.Process(0, 0, blockN, N, 1);
    int offset = blockN;
    for(int i = 1;i < cnt;++i,offset += blockN) {
        CrossCoreSetFlag<0x0, PIPE_MTE3>(vec_wait_vec);
        CrossCoreWaitFlag(vec_wait_vec);
        CrossCoreSetFlag<0x2, PIPE_MTE3>(cube_wait_vec);//let gemm start
        CrossCoreWaitFlag(vec_wait_cube);//wait gemm finish
        CrossCoreSetFlag<0x0, PIPE_MTE3>(vec_wait_vec);
        CrossCoreWaitFlag(vec_wait_vec);
        choleskySolver.Process(offset, offset, blockN, N - offset, 1);
    }
    CrossCoreSetFlag<0x0, PIPE_MTE3>(vec_wait_vec);
    CrossCoreWaitFlag(vec_wait_vec);
    tbufPool.Reset();
    oppad.SetMatrix(tensors.aGlobal, tensors.aGlobalOrg, N, strideN, lda);
    if(isTranse){
        oppad.Init2(&tbufPool,tensors.gather);
        oppad.PadCustomWork3(orgM, blockNum << 1, blockIdx);
    }else{
        oppad.Init(&tbufPool);
        oppad.PadCustomWork(orgM, blockNum << 1, blockIdx);
    }
}

// cube 核：按步长派发 GEMM 更新工作量
__aicore__ inline void CustomLltCube(TPipe &pipe, int orgM, int orgN, int blockN, int strideN, int blockNum,
                                     int blockIdx, CustomLltTensors &tensors) {
    int cnt = 0;
    if(blockN == 32){
        cnt = strideN / 32;
    }else{
        cnt = strideN / 64;
    }
    MatmulCustom_01<float> opmm;
    opmm.Init(&pipe);
    SetAtomicAdd<float>();
    int stepIdx = 0;

    GemmSolver<float> gemmSolver;
    gemmSolver.Init(opmm);
    SetAtomicAdd<float>();
    gemmSolver.SetMatrix(tensors.aGlobalOrg, tensors.aGlobal, tensors.aGlobal, strideN, strideN, blockNum, blockIdx);
    opmm.SetMatrix(tensors.aGlobalOrg, tensors.aGlobal, tensors.aGlobal, strideN, strideN);
    gemmSolver.SetOffset(tensors.aOffset, tensors.bOffset, tensors.cOffset, tensors.offsetIdx);

    int step = 0;
    int baseM = 128;
    int offset = blockN;
    for(int i = 1;i < cnt;++i,offset += blockN){
        int curblockM = (i & -i) * blockN;
        CrossCoreWaitFlag(cube_wait_vec);
        if(curblockM < baseM){
            gemmSolver.update(offset - curblockM,offset,curblockM,strideN);
            CrossCoreSetFlag<0x2, PIPE_FIX>(vec_wait_cube);
            gemmSolver.WorkStep(step--);
        }else{
            gemmSolver.update(offset - baseM,offset,baseM,strideN);
            CrossCoreSetFlag<0x2, PIPE_FIX>(vec_wait_cube);

            if(blockN == 32){
                step = 4;
            }else{
                step = 2;
            }
            gemmSolver.GemmWorkGenerate(stepIdx);
            stepIdx++;
            gemmSolver.WorkStep(step--);
        }
    }
    SetAtomicNone();
}

__aicore__ inline void custom_llt(int orgM, int orgN, int blockM, int blockN, int lda, bool isTranse, GM_ADDR A_org, GM_ADDR A_work, GM_ADDR Gather, __gm__ uint32_t * offset1, __gm__ uint32_t * offset2, __gm__ uint32_t * offset3, __gm__ uint32_t * OffsetIdx) {
    int alignsize = 128;
    int strideN = (orgN + alignsize-1) / alignsize * alignsize;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();

    CustomLltTensors tensors;
    CustomLltInitData(A_org, A_work, Gather, offset1, offset2, offset3, OffsetIdx, tensors);

    TPipe pipe;
    TBufPool<TPosition::VECCALC, 16> tbufPool;

    #ifdef __DAV_C220_VEC__
        CustomLltVector(pipe, tbufPool, orgM, orgN, blockN, lda, isTranse, strideN, blockNum, blockIdx, tensors);
    #elif  __DAV_C220_CUBE__
        CustomLltCube(pipe, orgM, orgN, blockN, strideN, blockNum, blockIdx, tensors);
    #endif
}

__global__ __aicore__ void spotrf_kernel(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int lda, bool isTranse,
                                            GM_ADDR A_org, GM_ADDR A_work, GM_ADDR Gather,
                                            __gm__ uint32_t* offset1, __gm__ uint32_t* offset2, __gm__ uint32_t* offset3, __gm__ uint32_t* OffsetIdx) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  custom_llt(orgM, orgN, blockM, blockN, lda, isTranse,A_org, A_work, Gather, offset1, offset2, offset3, OffsetIdx);
}
//getrf
void spotrf_kernel_do(GM_ADDR sync, int orgM, int orgN, int blockM, int blockN, int lda,bool isTranse,
                      GM_ADDR A_org, GM_ADDR A_work, GM_ADDR Gather, 
                      __gm__ uint32_t* offset1, __gm__ uint32_t* offset2, __gm__ uint32_t* offset3, __gm__ uint32_t* OffsetIdx,
                      uint32_t numBlocks, void *stream) {
  spotrf_kernel<<<numBlocks, nullptr, stream>>>(
      sync, orgM, orgN, blockM, blockN, lda, isTranse, A_org, A_work, Gather, offset1, offset2, offset3, OffsetIdx);
}
#endif  // SPOTRF_MIX_H