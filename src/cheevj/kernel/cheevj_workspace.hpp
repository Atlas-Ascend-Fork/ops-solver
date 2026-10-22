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
 * \file cheevj_workspace.hpp
 * \brief Workspace helpers for the Cheevj large-size AscendC path.
 */

#ifndef CHEEVJ_C64_WORKSPACE_HPP
#define CHEEVJ_C64_WORKSPACE_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t *
#endif

#include "../../utils/kernel/c64/pad.hpp"

using namespace AscendC;

namespace Cheevj
{

constexpr int CHEEVJ_WORKSPACE_STRIDE_N_ALIGN = 128;
constexpr int CHEEVJ_WORKSPACE_M_ALIGN = 16;
constexpr int CHEEVJ_WORKSPACE_TILE_LENGTH = 8192;

constexpr int CHEEVJ_WS_PLANE_A_REAL = 0;
constexpr int CHEEVJ_WS_PLANE_A_IMAG = 1;
constexpr int CHEEVJ_WS_PLANE_A_IMAG_NEG = 2;
constexpr int CHEEVJ_WS_PLANE_V_REAL = 3;
constexpr int CHEEVJ_WS_PLANE_V_IMAG = 4;
constexpr int CHEEVJ_WS_PLANES_NO_VECTOR = 3;
constexpr int CHEEVJ_WS_PLANES_VECTOR = 5;

__aicore__ inline int AlignUp(int value, int align) { return align > 0 ? (value + align - 1) / align * align : value; }

__aicore__ inline int AlignUp128(int value) { return AlignUp(value, CHEEVJ_WORKSPACE_STRIDE_N_ALIGN); }

__aicore__ inline int AlignUp16(int value) { return AlignUp(value, CHEEVJ_WORKSPACE_M_ALIGN); }

__aicore__ inline int MinInt(int lhs, int rhs) { return lhs < rhs ? lhs : rhs; }

__aicore__ inline int CheevjWorkspaceWorkerCount()
{
#ifdef CHEEVJ_ENABLE_MIXED_AIV_WORKER_COUNT
    // A mixed 1:2 launch uses one block count per AIC and two AIV sub-blocks.
    return GetBlockNum() * GetSubBlockNum();
#else
    return GetBlockNum();
#endif
}

__aicore__ inline int CheevjWorkspaceWorkerIndex()
{
    // On the AIV side of a mixed launch GetBlockIdx() is already flattened.
    return GetBlockIdx();
}

__aicore__ inline void CheevjWorkspaceSync()
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    PipeBarrier<PIPE_ALL>();
    CrossCoreSetFlag<0x0, PIPE_MTE3>(0x2);
    CrossCoreWaitFlag(0x2);
#endif
}

struct CheevjPlanarWorkspaceLayout
{
    int n;
    int strideN;
    int workM;
    int planeElems;
    int basePlanes;
    int scratchPlanes;
    int totalPlanes;

    __aicore__ inline int PlaneOffset(int plane) const { return plane * planeElems; }

    __aicore__ inline int ARealOffset() const { return PlaneOffset(CHEEVJ_WS_PLANE_A_REAL); }

    __aicore__ inline int AImagOffset() const { return PlaneOffset(CHEEVJ_WS_PLANE_A_IMAG); }

    __aicore__ inline int AImagNegOffset() const { return PlaneOffset(CHEEVJ_WS_PLANE_A_IMAG_NEG); }

    __aicore__ inline int VRealOffset() const { return PlaneOffset(CHEEVJ_WS_PLANE_V_REAL); }

    __aicore__ inline int VImagOffset() const { return PlaneOffset(CHEEVJ_WS_PLANE_V_IMAG); }

    __aicore__ inline int ScratchOffset(int scratchIdx) const { return PlaneOffset(basePlanes + scratchIdx); }

    __aicore__ inline int TotalElems() const { return totalPlanes * planeElems; }

    __aicore__ inline uint64_t TotalBytes() const { return static_cast<uint64_t>(TotalElems()) * sizeof(float); }
};

__aicore__ inline CheevjPlanarWorkspaceLayout MakeCheevjPlanarWorkspaceLayout(int n, bool needVectors,
                                                                              int scratchPlanes = 0)
{
    CheevjPlanarWorkspaceLayout layout;
    layout.n = n;
    layout.strideN = AlignUp128(n);
    layout.workM = AlignUp16(n);
    layout.planeElems = layout.strideN * layout.workM;
    layout.basePlanes = needVectors ? CHEEVJ_WS_PLANES_VECTOR : CHEEVJ_WS_PLANES_NO_VECTOR;
    layout.scratchPlanes = scratchPlanes;
    layout.totalPlanes = layout.basePlanes + scratchPlanes;
    return layout;
}

struct CheevjPlanarWorkspace
{
    CheevjPlanarWorkspaceLayout layout;
    GlobalTensor<float> base;
    GlobalTensor<float> aReal;
    GlobalTensor<float> aImag;
    GlobalTensor<float> aImagNeg;
    GlobalTensor<float> vReal;
    GlobalTensor<float> vImag;
    GlobalTensor<float> scratch0;
    GlobalTensor<float> scratch1;
    GlobalTensor<float> scratch2;
    GlobalTensor<float> scratch3;
    GlobalTensor<float> scratch4;

    __aicore__ inline void Bind(GM_ADDR workspace, int n, bool needVectors, int scratchPlanes = 0)
    {
        base.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        Bind(base, n, needVectors, scratchPlanes);
    }

    __aicore__ inline void Bind(GlobalTensor<float> workspace, int n, bool needVectors, int scratchPlanes = 0)
    {
        layout = MakeCheevjPlanarWorkspaceLayout(n, needVectors, scratchPlanes);
        base = workspace;
        aReal = base[layout.ARealOffset()];
        aImag = base[layout.AImagOffset()];
        aImagNeg = base[layout.AImagNegOffset()];
        vReal = base[needVectors ? layout.VRealOffset() : 0];
        vImag = base[needVectors ? layout.VImagOffset() : 0];
        scratch0 = base[scratchPlanes > 0 ? layout.ScratchOffset(0) : 0];
        scratch1 = base[scratchPlanes > 1 ? layout.ScratchOffset(1) : 0];
        scratch2 = base[scratchPlanes > 2 ? layout.ScratchOffset(2) : 0];
        scratch3 = base[scratchPlanes > 3 ? layout.ScratchOffset(3) : 0];
        scratch4 = base[scratchPlanes > 4 ? layout.ScratchOffset(4) : 0];
    }

    __aicore__ inline GlobalTensor<float> ScratchPlane(int scratchIdx) const
    {
        return base[layout.ScratchOffset(scratchIdx)];
    }
};

class CheevjCompactToPlanarSplit
{
   public:
    __aicore__ inline CheevjCompactToPlanarSplit() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe) { splitter.Init(pipe); }

    __aicore__ inline void SetMatrix(GM_ADDR compactInterleaved, GM_ADDR workspace, int n, int inputLda,
                                     bool needVectors, int scratchPlanes = 0)
    {
        GlobalTensor<float> compactGlobal;
        compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
        CheevjPlanarWorkspace planarWorkspace;
        planarWorkspace.Bind(workspace, n, needVectors, scratchPlanes);
        SetMatrix(compactGlobal, planarWorkspace, n, inputLda);
    }

    __aicore__ inline void SetMatrix(GlobalTensor<float> compactInterleaved,
                                     const CheevjPlanarWorkspace &planarWorkspace, int n, int inputLda)
    {
        splitter.SetMatrix(compactInterleaved, planarWorkspace.aReal, planarWorkspace.aImag, inputLda, inputLda,
                           planarWorkspace.layout.strideN);
        this->n = n;
    }

    __aicore__ inline void Process()
    {
        splitter.SplitRealImagWork(n, CheevjWorkspaceWorkerCount(), CheevjWorkspaceWorkerIndex());
    }

   private:
    int n;
    SplitRealImag<float> splitter;
};

class CheevjImagNegMaterialize
{
   public:
    __aicore__ inline CheevjImagNegMaterialize() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(inQueue, 2, CHEEVJ_WORKSPACE_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outQueue, 2, CHEEVJ_WORKSPACE_TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void SetPlanes(GlobalTensor<float> imagGlobal, GlobalTensor<float> imagNegGlobal, int strideN,
                                     int workM)
    {
        aImag = imagGlobal;
        aImagNeg = imagNegGlobal;
        totalElems = strideN * workM;
    }

    __aicore__ inline void SetWorkspace(const CheevjPlanarWorkspace &planarWorkspace)
    {
        SetPlanes(planarWorkspace.aImag, planarWorkspace.aImagNeg, planarWorkspace.layout.strideN,
                  planarWorkspace.layout.workM);
    }

    __aicore__ inline void Process()
    {
        const int tileCount = AlignUp(totalElems, CHEEVJ_WORKSPACE_TILE_LENGTH) / CHEEVJ_WORKSPACE_TILE_LENGTH;
        const int workerCount = CheevjWorkspaceWorkerCount();
        for (int tileIdx = CheevjWorkspaceWorkerIndex(); tileIdx < tileCount; tileIdx += workerCount)
        {
            const int offset = tileIdx * CHEEVJ_WORKSPACE_TILE_LENGTH;
            const int curElems = MinInt(CHEEVJ_WORKSPACE_TILE_LENGTH, totalElems - offset);
            ProcessTile(offset, curElems);
        }
    }

   private:
    TQue<QuePosition::VECIN, 2> inQueue;
    TQue<QuePosition::VECOUT, 2> outQueue;
    GlobalTensor<float> aImag;
    GlobalTensor<float> aImagNeg;
    int totalElems;

    __aicore__ inline void ProcessTile(int offset, int elems)
    {
        LocalTensor<float> srcLocal = inQueue.AllocTensor<float>();
        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(srcLocal, aImag[offset], copyParams, padParams);
        inQueue.EnQue(srcLocal);

        srcLocal = inQueue.DeQue<float>();
        LocalTensor<float> dstLocal = outQueue.AllocTensor<float>();
        Muls(dstLocal, srcLocal, -1.0f, elems);
        PipeBarrier<PIPE_V>();
        outQueue.EnQue(dstLocal);
        inQueue.FreeTensor(srcLocal);

        dstLocal = outQueue.DeQue<float>();
        DataCopyPad(aImagNeg[offset], dstLocal, copyParams);
        outQueue.FreeTensor(dstLocal);
    }
};

class CheevjIdentityInit
{
   public:
    __aicore__ inline CheevjIdentityInit() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(outQueue, 2, CHEEVJ_WORKSPACE_TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void SetPlanes(GlobalTensor<float> vRealGlobal, GlobalTensor<float> vImagGlobal, int n,
                                     int strideN, int workM)
    {
        vReal = vRealGlobal;
        vImag = vImagGlobal;
        this->n = n;
        this->strideN = strideN;
        totalElems = strideN * workM;
    }

    __aicore__ inline void SetWorkspace(const CheevjPlanarWorkspace &planarWorkspace)
    {
        SetPlanes(planarWorkspace.vReal, planarWorkspace.vImag, planarWorkspace.layout.n,
                  planarWorkspace.layout.strideN, planarWorkspace.layout.workM);
    }

    __aicore__ inline void Process()
    {
        ZeroPlane(vReal);
        ZeroPlane(vImag);
        CheevjWorkspaceSync();
        WriteIdentityDiagonalScalarTail();
        CheevjWorkspaceSync();
    }

   private:
    TQue<QuePosition::VECOUT, 2> outQueue;
    GlobalTensor<float> vReal;
    GlobalTensor<float> vImag;
    int n;
    int strideN;
    int totalElems;

    __aicore__ inline void ZeroPlane(GlobalTensor<float> plane)
    {
        const int tileCount = AlignUp(totalElems, CHEEVJ_WORKSPACE_TILE_LENGTH) / CHEEVJ_WORKSPACE_TILE_LENGTH;
        const int workerCount = CheevjWorkspaceWorkerCount();
        for (int tileIdx = CheevjWorkspaceWorkerIndex(); tileIdx < tileCount; tileIdx += workerCount)
        {
            const int offset = tileIdx * CHEEVJ_WORKSPACE_TILE_LENGTH;
            const int curElems = MinInt(CHEEVJ_WORKSPACE_TILE_LENGTH, totalElems - offset);
            ZeroTile(plane, offset, curElems);
        }
    }

    __aicore__ inline void ZeroTile(GlobalTensor<float> plane, int offset, int elems)
    {
        LocalTensor<float> tileLocal = outQueue.AllocTensor<float>();
        Duplicate(tileLocal, 0.0f, elems);
        PipeBarrier<PIPE_V>();
        outQueue.EnQue(tileLocal);

        tileLocal = outQueue.DeQue<float>();
        const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(plane[offset], tileLocal, copyParams);
        outQueue.FreeTensor(tileLocal);
    }

    __aicore__ inline void WriteIdentityDiagonalScalarTail()
    {
        if (CheevjWorkspaceWorkerIndex() != 0)
        {
            return;
        }

        // Identity initialization runs once per solve. Keeping the diagonal
        // scalar avoids reserving another index workspace in the hot matrix path.
        LocalTensor<float> oneLocal = outQueue.AllocTensor<float>();
        Duplicate(oneLocal, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        outQueue.EnQue(oneLocal);

        oneLocal = outQueue.DeQue<float>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        for (int idx = 0; idx < n; ++idx)
        {
            DataCopyPad(vReal[idx * strideN + idx], oneLocal, copyParams);
        }
        outQueue.FreeTensor(oneLocal);
    }
};

__aicore__ inline void SplitCompactInterleavedToPlanar(TBufPool<TPosition::VECCALC, 16> &tbufPool,
                                                       GM_ADDR compactInterleaved, GM_ADDR workspace, int n,
                                                       int inputLda, bool needVectors, int scratchPlanes = 0)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjCompactToPlanarSplit op;
    op.Init(&tbufPool);
    op.SetMatrix(compactInterleaved, workspace, n, inputLda, needVectors, scratchPlanes);
    op.Process();
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)compactInterleaved;
    (void)workspace;
    (void)n;
    (void)inputLda;
    (void)needVectors;
    (void)scratchPlanes;
#endif
}

__aicore__ inline void MaterializeImagNeg(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR workspace, int n,
                                          bool needVectors, int scratchPlanes = 0)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjPlanarWorkspace planar;
    planar.Bind(workspace, n, needVectors, scratchPlanes);
    CheevjImagNegMaterialize op;
    op.Init(&tbufPool);
    op.SetWorkspace(planar);
    op.Process();
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)workspace;
    (void)n;
    (void)needVectors;
    (void)scratchPlanes;
#endif
}

__aicore__ inline void InitPlanarIdentity(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR workspace, int n,
                                          int scratchPlanes = 0)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjPlanarWorkspace planar;
    planar.Bind(workspace, n, true, scratchPlanes);
    CheevjIdentityInit op;
    op.Init(&tbufPool);
    op.SetWorkspace(planar);
    op.Process();
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)workspace;
    (void)n;
    (void)scratchPlanes;
#endif
}

__aicore__ inline void StageDenseTileForStridedRow(LocalTensor<float> staged, LocalTensor<float> dense, int elems)
{
    const int stageElems = elems * 8;
    Duplicate(staged, 0.0f, stageElems);
    PipeBarrier<PIPE_V>();
    for (int idx = 0; idx < elems; ++idx)
    {
        staged.SetValue(idx * 8, dense.GetValue(idx));
    }
    PipeBarrier<PIPE_ALL>();
}

}  // namespace Cheevj

#endif  // CHEEVJ_C64_WORKSPACE_HPP
