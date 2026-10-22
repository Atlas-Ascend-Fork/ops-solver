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
 * \file cheevj_output.hpp
 * \brief Output helpers for the Cheevj large-size AscendC path.
 */

#ifndef CHEEVJ_C64_OUTPUT_HPP
#define CHEEVJ_C64_OUTPUT_HPP

#include <cstdint>

#include "cheevj_workspace.hpp"
#include "kernel_operator.h"

using namespace AscendC;

namespace Cheevj
{

constexpr int CHEEVJ_OUTPUT_TILE_LENGTH = 8192;
constexpr int CHEEVJ_OUTPUT_ROW_TILE_LENGTH = CHEEVJ_OUTPUT_TILE_LENGTH / 2;
constexpr int CHEEVJ_OUTPUT_LOCAL_SORT_MAX_N = 2048;
constexpr int CHEEVJ_OUTPUT_DIAGONAL_TILE = CHEEVJ_OUTPUT_TILE_LENGTH / 8;
constexpr int CHEEVJ_OUTPUT_DIAGONAL_STAGED_FLOATS = CHEEVJ_OUTPUT_DIAGONAL_TILE * 8;
constexpr int CHEEVJ_OUTPUT_SORT_UNIT = 32;

class CheevjPlanarOutputWriter
{
   public:
    __aicore__ inline CheevjPlanarOutputWriter() {}

    __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
    {
        pipe->InitBuffer(inQueue, 1, CHEEVJ_OUTPUT_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(outQueue, 1, CHEEVJ_OUTPUT_TILE_LENGTH * sizeof(float));
        pipe->InitBuffer(interleaveGatherBuf, CHEEVJ_OUTPUT_TILE_LENGTH * sizeof(uint32_t));
        pipe->InitBuffer(wBuf, CHEEVJ_OUTPUT_LOCAL_SORT_MAX_N * sizeof(float));
        pipe->InitBuffer(columnMapBuf, CHEEVJ_OUTPUT_LOCAL_SORT_MAX_N * sizeof(float));
        pipe->InitBuffer(diagonalStagedBuf, CHEEVJ_OUTPUT_DIAGONAL_STAGED_FLOATS * sizeof(float));
    }

    __aicore__ inline void SetOutput(GM_ADDR compactInterleaved, GM_ADDR w, GM_ADDR workspace, int n, bool needVectors,
                                     int scratchPlanes = 0)
    {
        GlobalTensor<float> compactGlobal;
        GlobalTensor<float> wGlobal;
        compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
        wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));

        CheevjPlanarWorkspace planarWorkspace;
        planarWorkspace.Bind(workspace, n, needVectors, scratchPlanes);
        SetOutput(compactGlobal, wGlobal, planarWorkspace);
    }

    __aicore__ inline void SetOutput(GlobalTensor<float> compactInterleaved, GlobalTensor<float> w,
                                     const CheevjPlanarWorkspace &planarWorkspace)
    {
        compact = compactInterleaved;
        wOut = w;
        planar = planarWorkspace;
    }

    __aicore__ inline void Process(bool writeVectors)
    {
        if (!IsSupportedN())
        {
            return;
        }

        const bool storeColumnMap = writeVectors && HasColumnMapScratch();
        WriteSortedEigenvalues(storeColumnMap);
        CheevjWorkspaceSync();

        if (writeVectors)
        {
            if (storeColumnMap)
            {
                WriteVectorOutputFromStoredColumnMap();
            }
            else
            {
                WriteVectorOutputFromLocalColumnMap();
            }
        }
        CheevjWorkspaceSync();
    }

    __aicore__ inline void WriteSortedEigenvalues(bool storeColumnMap)
    {
        if (GetBlockIdx() != 0 || !IsSupportedN())
        {
            return;
        }
        if (planar.layout.n > CHEEVJ_OUTPUT_LOCAL_SORT_MAX_N || planar.layout.n % 8 != 0)
        {
            WriteSortedEigenvaluesScalar(storeColumnMap);
            return;
        }

        LocalTensor<float> values = wBuf.Get<float>();
        LocalTensor<float> columnMap = columnMapBuf.Get<float>();
        CopyDiagonalToLocal(values);
        SortValuesAndColumnMap(values, columnMap);
        CopySortedValuesToW(values);
        if (storeColumnMap && HasColumnMapScratch())
        {
            CopyColumnMapToScratch(columnMap);
        }
    }

    __aicore__ inline void WriteVectorOutputIdentityOrder()
    {
        if (!IsSupportedN())
        {
            return;
        }
        WritePlanarToCompact(planar.vReal, planar.vImag);
    }

    __aicore__ inline void WriteMatrixOutputIdentityOrder()
    {
        if (!IsSupportedN())
        {
            return;
        }
        WritePlanarToCompact(planar.aReal, planar.aImag);
    }

    __aicore__ inline void WriteVectorOutputFromStoredColumnMap()
    {
        if (!IsSupportedN() || !HasColumnMapScratch())
        {
            return;
        }
        if (planar.layout.n % 8 != 0)
        {
            WritePlanarToCompactByGlobalColumnMapScalar(planar.vReal, planar.vImag, planar.scratch0);
            return;
        }
        WritePlanarToCompactByGlobalColumnMap(planar.vReal, planar.vImag, planar.scratch0);
    }

    __aicore__ inline void WritePlanarToCompact(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane)
    {
        PrepareInterleaveGather();
        const int rowTileCount = RowTileCount();
        const int activeRowTileCount = rowTileCount > 0 ? rowTileCount : 1;
        const int taskCount = planar.layout.n * rowTileCount;
        const int workerCount = CheevjWorkspaceWorkerCount();
        for (int taskIdx = GetBlockIdx(); taskIdx < taskCount; taskIdx += workerCount)
        {
            const int dstCol = taskIdx / activeRowTileCount;
            const int rowTile = taskIdx - dstCol * rowTileCount;
            const int rowOffset = rowTile * CHEEVJ_OUTPUT_ROW_TILE_LENGTH;
            const int elems = MinInt(CHEEVJ_OUTPUT_ROW_TILE_LENGTH, planar.layout.n - rowOffset);
            WriteCompactColumnTile(realPlane, imagPlane, dstCol, dstCol, rowOffset, elems);
        }
    }

    __aicore__ inline void WritePlanarToCompactByGlobalColumnMap(GlobalTensor<float> realPlane,
                                                                 GlobalTensor<float> imagPlane,
                                                                 GlobalTensor<float> columnMapGlobal)
    {
        PrepareInterleaveGather();
        const int rowTileCount = RowTileCount();
        const int activeRowTileCount = rowTileCount > 0 ? rowTileCount : 1;
        const int taskCount = planar.layout.n * rowTileCount;
        const int workerCount = CheevjWorkspaceWorkerCount();
        for (int taskIdx = GetBlockIdx(); taskIdx < taskCount; taskIdx += workerCount)
        {
            const int dstCol = taskIdx / activeRowTileCount;
            const int srcCol = static_cast<int>(columnMapGlobal.GetValue(dstCol));
            const int rowTile = taskIdx - dstCol * rowTileCount;
            const int rowOffset = rowTile * CHEEVJ_OUTPUT_ROW_TILE_LENGTH;
            const int elems = MinInt(CHEEVJ_OUTPUT_ROW_TILE_LENGTH, planar.layout.n - rowOffset);
            WriteCompactColumnTile(realPlane, imagPlane, srcCol, dstCol, rowOffset, elems);
        }
    }

    __aicore__ inline void WritePlanarToCompactByLocalColumnMap(GlobalTensor<float> realPlane,
                                                                GlobalTensor<float> imagPlane,
                                                                LocalTensor<float> columnMapLocal)
    {
        if (GetBlockIdx() != 0)
        {
            return;
        }

        PrepareInterleaveGather();
        const int rowTileCount = RowTileCount();
        for (int dstCol = 0; dstCol < planar.layout.n; ++dstCol)
        {
            const int srcCol = static_cast<int>(columnMapLocal.GetValue(dstCol));
            for (int rowTile = 0; rowTile < rowTileCount; ++rowTile)
            {
                const int rowOffset = rowTile * CHEEVJ_OUTPUT_ROW_TILE_LENGTH;
                const int elems = MinInt(CHEEVJ_OUTPUT_ROW_TILE_LENGTH, planar.layout.n - rowOffset);
                WriteCompactColumnTile(realPlane, imagPlane, srcCol, dstCol, rowOffset, elems);
            }
        }
    }

   private:
    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> interleaveGatherBuf;
    TBuf<TPosition::VECCALC> wBuf;
    TBuf<TPosition::VECCALC> columnMapBuf;
    TBuf<TPosition::VECCALC> diagonalStagedBuf;
    GlobalTensor<float> compact;
    GlobalTensor<float> wOut;
    CheevjPlanarWorkspace planar;

    __aicore__ inline bool IsSupportedN() const { return planar.layout.n > 0; }

    __aicore__ inline bool HasColumnMapScratch() const { return planar.layout.scratchPlanes > 0; }

    __aicore__ inline int RowTileCount() const
    {
        return AlignUp(planar.layout.n, CHEEVJ_OUTPUT_ROW_TILE_LENGTH) / CHEEVJ_OUTPUT_ROW_TILE_LENGTH;
    }

    __aicore__ inline int PlanarOffset(int row, int col) const { return col * planar.layout.strideN + row; }

    __aicore__ inline int CompactComplexOffset(int row, int col) const { return (row + col * planar.layout.n) * 2; }

    __aicore__ inline void CopyDiagonalToLocal(LocalTensor<float> values)
    {
        LocalTensor<float> staged = diagonalStagedBuf.Get<float>();
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};

        for (int diagOffset = 0; diagOffset < planar.layout.n; diagOffset += CHEEVJ_OUTPUT_DIAGONAL_TILE)
        {
            const int elems = MinInt(CHEEVJ_OUTPUT_DIAGONAL_TILE, planar.layout.n - diagOffset);
            DataCopyExtParams copyParams{static_cast<uint16_t>(elems), static_cast<uint32_t>(sizeof(float)),
                                         static_cast<uint32_t>(planar.layout.strideN * sizeof(float)), 0, 0};
            DataCopyPad(staged, planar.aReal[PlanarOffset(diagOffset, diagOffset)], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();

            PrepareStridedFloatGather(elems);
            LocalTensor<uint32_t> gatherLocal = interleaveGatherBuf.Get<uint32_t>();
            Gather(values[diagOffset], staged, gatherLocal, 0, elems);
            PipeBarrier<PIPE_ALL>();
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareStridedFloatGather(int elems)
    {
        LocalTensor<uint32_t> gatherLocal = interleaveGatherBuf.Get<uint32_t>();
        LocalTensor<int32_t> signedIndex = gatherLocal.template ReinterpretCast<int32_t>();
        CreateVecIndex(signedIndex, static_cast<int32_t>(0), static_cast<uint32_t>(elems));
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherLocal, gatherLocal, static_cast<uint32_t>(5), elems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SortValuesAndColumnMap(LocalTensor<float> values, LocalTensor<float> columnMap)
    {
        const int alignedN = AlignUp(planar.layout.n, CHEEVJ_OUTPUT_SORT_UNIT);
        const int repeatTimes = alignedN / CHEEVJ_OUTPUT_SORT_UNIT;
        const uint32_t sortedFloats = GetSortLen<float>(static_cast<uint32_t>(alignedN));

        LocalTensor<uint32_t> columnIndex = interleaveGatherBuf.Get<uint32_t>();
        LocalTensor<int32_t> signedColumnIndex = columnIndex.template ReinterpretCast<int32_t>();
        LocalTensor<float> sortStorage = diagonalStagedBuf.Get<float>();
        LocalTensor<float> sorted = sortStorage;
        LocalTensor<float> sortTmp = sortStorage[sortedFloats];

        // C220 Sort32/MrgSort orders proposals descending. Negating the keys
        // before and after the sort produces the ascending LAPACK order.
        Muls(values, values, -1.0f, planar.layout.n);
        if (alignedN > planar.layout.n)
        {
            Duplicate(values[planar.layout.n], -3.4028234663852886e+38F, alignedN - planar.layout.n);
        }
        CreateVecIndex(signedColumnIndex, static_cast<int32_t>(0), static_cast<uint32_t>(alignedN));
        PipeBarrier<PIPE_V>();

        LocalTensor<float> concat;
        Concat(concat, values, sortTmp, repeatTimes);
        Sort<float, true>(sorted, concat, columnIndex, sortTmp, repeatTimes);
        PipeBarrier<PIPE_V>();
        Extract(values, columnIndex, sorted, repeatTimes);
        PipeBarrier<PIPE_V>();

        Muls(values, values, -1.0f, planar.layout.n);
        Cast(columnMap, signedColumnIndex, RoundMode::CAST_NONE, planar.layout.n);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopySortedValuesToW(LocalTensor<float> values)
    {
        const uint32_t bytes = static_cast<uint32_t>(planar.layout.n * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(wOut[0], values, copyParams);
    }

    __aicore__ inline void CopyColumnMapToScratch(LocalTensor<float> columnMap)
    {
        const uint32_t bytes = static_cast<uint32_t>(planar.layout.n * sizeof(float));
        DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
        DataCopyPad(planar.scratch0[0], columnMap, copyParams);
    }

    __aicore__ inline void WriteVectorOutputFromLocalColumnMap()
    {
        LocalTensor<float> columnMap = columnMapBuf.Get<float>();
        WritePlanarToCompactByLocalColumnMap(planar.vReal, planar.vImag, columnMap);
    }

    __aicore__ inline void WriteSortedEigenvaluesScalar(bool storeColumnMap)
    {
        for (int col = 0; col < planar.layout.n; ++col)
        {
            const float value = planar.aReal.GetValue(PlanarOffset(col, col));
            int insertAt = col;
            while (insertAt > 0 && wOut.GetValue(insertAt - 1) > value)
            {
                wOut.SetValue(insertAt, wOut.GetValue(insertAt - 1));
                if (storeColumnMap)
                {
                    planar.scratch0.SetValue(insertAt, planar.scratch0.GetValue(insertAt - 1));
                }
                --insertAt;
            }
            wOut.SetValue(insertAt, value);
            if (storeColumnMap)
            {
                planar.scratch0.SetValue(insertAt, static_cast<float>(col));
            }
        }
    }

    __aicore__ inline void WritePlanarToCompactByGlobalColumnMapScalar(GlobalTensor<float> realPlane,
                                                                       GlobalTensor<float> imagPlane,
                                                                       GlobalTensor<float> columnMapGlobal)
    {
        if (GetBlockIdx() != 0)
        {
            return;
        }
        for (int dstCol = 0; dstCol < planar.layout.n; ++dstCol)
        {
            const int srcCol = static_cast<int>(columnMapGlobal.GetValue(dstCol));
            for (int row = 0; row < planar.layout.n; ++row)
            {
                const int srcOffset = PlanarOffset(row, srcCol);
                const int dstOffset = CompactComplexOffset(row, dstCol);
                compact.SetValue(dstOffset, realPlane.GetValue(srcOffset));
                compact.SetValue(dstOffset + 1, imagPlane.GetValue(srcOffset));
            }
        }
    }

    __aicore__ inline void PrepareInterleaveGather()
    {
        LocalTensor<uint32_t> gatherLocal = interleaveGatherBuf.Get<uint32_t>();
        LocalTensor<int32_t> gatherIndex = gatherLocal.template ReinterpretCast<int32_t>();
        LocalTensor<uint32_t> parityBits = diagonalStagedBuf.Get<uint32_t>();
        LocalTensor<int32_t> parity = parityBits.template ReinterpretCast<int32_t>();
        constexpr int indexElems = CHEEVJ_OUTPUT_ROW_TILE_LENGTH * 2;

        // For output lane j, build
        //   ((j >> 1) + (j & 1) * rowTileLength) * sizeof(float).
        // This selects real[j / 2] on even lanes and imag[j / 2] on odd lanes.
        CreateVecIndex(gatherIndex, static_cast<int32_t>(0), static_cast<uint32_t>(indexElems));
        Duplicate(parity, static_cast<int32_t>(1), indexElems);
        PipeBarrier<PIPE_V>();
        And(parity, gatherIndex, parity, indexElems);
        PipeBarrier<PIPE_V>();
        ShiftRight(gatherLocal, gatherLocal, static_cast<uint32_t>(1), indexElems);
        ShiftLeft(parityBits, parityBits, static_cast<uint32_t>(12), indexElems);
        PipeBarrier<PIPE_V>();
        Add(gatherIndex, gatherIndex, parity, indexElems);
        PipeBarrier<PIPE_V>();
        ShiftLeft(gatherLocal, gatherLocal, static_cast<uint32_t>(2), indexElems);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteCompactColumnTile(GlobalTensor<float> realPlane, GlobalTensor<float> imagPlane,
                                                  int srcCol, int dstCol, int rowOffset, int elems)
    {
        if (elems <= 0)
        {
            return;
        }

        LocalTensor<float> srcLocal = inQueue.AllocTensor<float>();
        const uint32_t planeBytes = static_cast<uint32_t>(elems * sizeof(float));
        DataCopyExtParams planeCopyParams{1, planeBytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
        DataCopyPad(srcLocal, realPlane[PlanarOffset(rowOffset, srcCol)], planeCopyParams, padParams);
        DataCopyPad(srcLocal[CHEEVJ_OUTPUT_ROW_TILE_LENGTH], imagPlane[PlanarOffset(rowOffset, srcCol)],
                    planeCopyParams, padParams);
        inQueue.EnQue(srcLocal);

        srcLocal = inQueue.DeQue<float>();
        LocalTensor<float> compactLocal = outQueue.AllocTensor<float>();
        LocalTensor<uint32_t> gatherLocal = interleaveGatherBuf.Get<uint32_t>();
        Gather(compactLocal, srcLocal, gatherLocal, 0, elems * 2);
        PipeBarrier<PIPE_ALL>();
        inQueue.FreeTensor(srcLocal);
        outQueue.EnQue(compactLocal);

        compactLocal = outQueue.DeQue<float>();
        const uint32_t compactBytes = static_cast<uint32_t>(elems * 2 * sizeof(float));
        DataCopyExtParams compactCopyParams{1, compactBytes, 0, 0, 0};
        DataCopyPad(compact[CompactComplexOffset(rowOffset, dstCol)], compactLocal, compactCopyParams);
        outQueue.FreeTensor(compactLocal);
    }
};

__aicore__ inline void WriteCheevjPlanarOutput(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR compactInterleaved,
                                               GM_ADDR w, GM_ADDR workspace, int n, bool needVectors,
                                               int scratchPlanes = 0)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjPlanarOutputWriter op;
    op.Init(&tbufPool);
    op.SetOutput(compactInterleaved, w, workspace, n, needVectors, scratchPlanes);
    op.Process(needVectors);
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)compactInterleaved;
    (void)w;
    (void)workspace;
    (void)n;
    (void)needVectors;
    (void)scratchPlanes;
#endif
}

__aicore__ inline void WriteCheevjPlanarEigenvalues(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR w,
                                                    GM_ADDR workspace, int n, bool needVectors, int scratchPlanes = 0,
                                                    bool storeColumnMap = false)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjPlanarWorkspace planar;
    planar.Bind(workspace, n, needVectors, scratchPlanes);

    GlobalTensor<float> compactUnused;
    GlobalTensor<float> wGlobal;
    compactUnused.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    wGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));

    CheevjPlanarOutputWriter op;
    op.Init(&tbufPool);
    op.SetOutput(compactUnused, wGlobal, planar);
    op.WriteSortedEigenvalues(storeColumnMap);
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)w;
    (void)workspace;
    (void)n;
    (void)needVectors;
    (void)scratchPlanes;
    (void)storeColumnMap;
#endif
}

__aicore__ inline void WriteCheevjPlanarVectorsToCompact(TBufPool<TPosition::VECCALC, 16> &tbufPool,
                                                         GM_ADDR compactInterleaved, GM_ADDR workspace, int n,
                                                         int scratchPlanes = 0, bool useStoredColumnMap = false)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjPlanarWorkspace planar;
    planar.Bind(workspace, n, true, scratchPlanes);

    GlobalTensor<float> compactGlobal;
    GlobalTensor<float> wUnused;
    compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
    wUnused.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

    CheevjPlanarOutputWriter op;
    op.Init(&tbufPool);
    op.SetOutput(compactGlobal, wUnused, planar);
    if (useStoredColumnMap)
    {
        op.WriteVectorOutputFromStoredColumnMap();
    }
    else
    {
        op.WriteVectorOutputIdentityOrder();
    }
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)compactInterleaved;
    (void)workspace;
    (void)n;
    (void)scratchPlanes;
    (void)useStoredColumnMap;
#endif
}

__aicore__ inline void WriteCheevjPlanarMatrixToCompact(TBufPool<TPosition::VECCALC, 16> &tbufPool,
                                                        GM_ADDR compactInterleaved, GM_ADDR workspace, int n,
                                                        bool needVectors, int scratchPlanes = 0)
{
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    CheevjPlanarWorkspace planar;
    planar.Bind(workspace, n, needVectors, scratchPlanes);

    GlobalTensor<float> compactGlobal;
    GlobalTensor<float> wUnused;
    compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
    wUnused.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));

    CheevjPlanarOutputWriter op;
    op.Init(&tbufPool);
    op.SetOutput(compactGlobal, wUnused, planar);
    op.WriteMatrixOutputIdentityOrder();
    tbufPool.Reset();
    CheevjWorkspaceSync();
#else
    (void)tbufPool;
    (void)compactInterleaved;
    (void)workspace;
    (void)n;
    (void)needVectors;
    (void)scratchPlanes;
#endif
}

}  // namespace Cheevj

#endif  // CHEEVJ_C64_OUTPUT_HPP
