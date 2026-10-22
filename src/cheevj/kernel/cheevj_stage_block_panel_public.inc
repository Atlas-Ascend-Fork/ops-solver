/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifdef CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
    __aicore__ inline bool ProcessSinglePairRightPanelAiv(int stage, int pairIndex, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanUseSinglePairRightPanel(pair, localDim) || slotIndex < 0 || slotIndex >= PanelServiceSlotCount() ||
            lane < 0 || lane > 1)
        {
            return false;
        }

        ProcessSinglePairRightPanelAivSplit(pair, pairIndex, localDim, slotIndex, lane, planar.aReal, planar.aImag,
                                            planar.aReal, planar.aImag, planar.aImagNeg, true, true);
        return true;
#else
        (void)stage;
        (void)pairIndex;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairRightVectorPanelAiv(int stage, int pairIndex, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!computeVectors || !HasVectorPlanes() || !CanUseSinglePairRightPanel(pair, localDim) || slotIndex < 0 ||
            slotIndex >= PanelServiceSlotCount() || lane < 0 || lane > 1)
        {
            return false;
        }

        ProcessSinglePairRightPanelAivSplit(pair, pairIndex, localDim, slotIndex, lane, planar.vReal, planar.vImag,
                                            planar.vReal, planar.vImag, planar.aImagNeg, false, false);
        return true;
#else
        (void)stage;
        (void)pairIndex;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool ProcessRightPanelStageAiv(int stage, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        return ProcessRightPanelStageAivSplit(stage, slotIndex, lane, planar.aReal, planar.aImag, planar.aReal,
                                              planar.aImag, planar.aImagNeg, true);
#else
        (void)stage;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool ProcessRightVectorPanelStageAiv(int stage, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        if (!computeVectors || !HasVectorPlanes())
        {
            return false;
        }
        return ProcessRightPanelStageAivSplit(stage, slotIndex, lane, planar.vReal, planar.vImag, planar.vReal,
                                              planar.vImag, planar.aImagNeg, false);
#else
        (void)stage;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool ProcessLeftPanelStageAiv(int stage, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        return ProcessLeftPanelStageAivSplit(stage, slotIndex, lane);
#else
        (void)stage;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairLeftPanelAiv(int stage, int pairIndex, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanUseSinglePairRightPanel(pair, localDim) || slotIndex < 0 || slotIndex >= PanelServiceSlotCount() ||
            lane < 0 || lane > 1)
        {
            return false;
        }

        ProcessSinglePairLeftPanelAivSplit(pair, pairIndex, localDim, slotIndex, lane);
        return true;
#else
        (void)stage;
        (void)pairIndex;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool CanProcessSinglePairRightPanel(int stage, int pairIndex) const
    {
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        return CanUseSinglePairRightPanel(pair, localDim);
    }

    __aicore__ inline int PanelServiceSlotCount() const
    {
        return ComputePanelServiceSlotCount(2 * schedule.BlockSize());
    }

    __aicore__ inline int PanelTaskCount() const { return schedule.PairCountPerStage() * PanelTaskSliceCount(); }

    __aicore__ inline bool CanUseDirectSegmentedRightPanel() const
    {
        const int physicalAicGroups = GetBlockNum() > 0 ? GetBlockNum() : 1;
        return computeVectors && planar.layout.n == 512 && planar.layout.strideN == 512 && schedule.BlockSize() == 16 &&
               schedule.PairCountPerStage() == 16 && physicalAicGroups >= 16;
    }

    __aicore__ inline bool ProcessDirectRightPanelStageAiv(int stage, int slotIndex, int lane)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, slotIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanUseDirectSegmentedRightPanel() || slotIndex < 0 || slotIndex >= schedule.PairCountPerStage() ||
            lane < 0 || lane > 1 || !CanUseSinglePairRightPanel(pair, localDim))
        {
            return false;
        }

        // Both AIV siblings publish readiness only after the stage-wide U store
        // barrier. One CUBE wait observes the pair, and its single completion
        // signal releases both siblings after direct A and V have finished.
        PipeBarrier<PIPE_ALL>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(CHEEVJ_STAGE_DIRECT_RIGHT_READY_FLAG);
        CrossCoreWaitFlag(CHEEVJ_STAGE_DIRECT_RIGHT_DONE_FLAG);
        PipeBarrier<PIPE_ALL>();
        return true;
#else
        (void)stage;
        (void)slotIndex;
        (void)lane;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairDirectRightPanelCube(int stage, int pairIndex, CMatmulCustom<float> *opmm)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanUseDirectSegmentedRightPanel() || pairIndex < 0 || pairIndex >= schedule.PairCountPerStage() ||
            !CanUseSinglePairRightPanel(pair, localDim) || localDim != 32 || opmm == nullptr)
        {
            return false;
        }

        CrossCoreWaitFlag(CHEEVJ_STAGE_DIRECT_RIGHT_READY_FLAG);

        opmm->SetMatrixWithAImagNegSegmented(StoredUnitaryReal(pairIndex), StoredUnitaryImag(pairIndex),
                                             StoredUnitaryImagNeg(pairIndex), planar.aReal, planar.aImag, planar.aReal,
                                             planar.aImag, planar.layout.n, localDim, planar.layout.strideN,
                                             pair.left.begin, pair.left.size, pair.right.begin, pair.right.size);
        for (int rowBegin = 0; rowBegin < planar.layout.n; rowBegin += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int rowCount = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowBegin);
            opmm->Process(0, rowBegin, rowBegin, localDim, rowCount, localDim);
        }

        opmm->SetMatrixWithAImagNegSegmented(StoredUnitaryReal(pairIndex), StoredUnitaryImag(pairIndex),
                                             StoredUnitaryImagNeg(pairIndex), planar.vReal, planar.vImag, planar.vReal,
                                             planar.vImag, planar.layout.n, localDim, planar.layout.strideN,
                                             pair.left.begin, pair.left.size, pair.right.begin, pair.right.size);
        for (int rowBegin = 0; rowBegin < planar.layout.n; rowBegin += CHEEVJ_STAGE_BLOCK_JACOBI16_TILE)
        {
            const int rowCount = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - rowBegin);
            opmm->Process(0, rowBegin, rowBegin, localDim, rowCount, localDim);
        }

        CrossCoreSetFlag<0x2, PIPE_FIX>(CHEEVJ_STAGE_DIRECT_RIGHT_DONE_FLAG);
        return true;
#else
        (void)stage;
        (void)pairIndex;
        (void)opmm;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairRightPanelCube(int stage, int pairIndex)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanUseSinglePairRightPanel(pair, localDim) || GetBlockIdx() != 0)
        {
            return false;
        }

        CheevjPanelUpdate panelUpdate;
        ConfigureSinglePairRightPanelUpdate(&panelUpdate, pair, pairIndex, 0, localDim,
                                            CHEEVJ_STAGE_PANEL_PACK_READY_FLAG, CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG);
        panelUpdate.ProcessMixedCube();
        return true;
#else
        (void)stage;
        (void)pairIndex;
        return false;
#endif
    }

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    __aicore__ inline bool CanProcessSinglePairPanelCube(const CheevjStageBlockPair16 &pair, int localDim,
                                                         int slotIndex, CMatmulCustom<float> *opmm) const
    {
        return CanUseSinglePairRightPanel(pair, localDim) && slotIndex >= 0 &&
               slotIndex < PanelServiceSlotCount() && opmm != nullptr;
    }

    __aicore__ inline bool CompleteSinglePairPanelCube(CMatmulCustom<float> *opmm, int localDim)
    {
        opmm->Process(0, 0, 0, localDim, planar.layout.n, localDim);
        CrossCoreSetFlag<0x2, PIPE_FIX>(CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG);
        return true;
    }
#endif

    __aicore__ inline bool ProcessSinglePairRightPanelCube(int stage, int pairIndex, int slotIndex,
                                                           CMatmulCustom<float> *opmm)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanProcessSinglePairPanelCube(pair, localDim, slotIndex, opmm))
        {
            return false;
        }

        // Compute (A[:, I] * U)^T = U^T * A[:, I]^T. Both operands are
        // naturally row-major in their existing physical layouts: stored U is
        // compact column-major (therefore U^T when reinterpreted), while each
        // selected column of A is one contiguous row of A^T.
        CrossCoreWaitFlag(CHEEVJ_STAGE_PANEL_PACK_READY_FLAG);
        opmm->SetMatrix(StoredUnitaryReal(pairIndex), StoredUnitaryImag(pairIndex),
                        PanelSlotInputReal(slotIndex, localDim), PanelSlotInputImag(slotIndex, localDim),
                        LeftPanelImagNeg(slotIndex, localDim), PanelSlotOutputReal(slotIndex, localDim),
                        PanelSlotOutputImag(slotIndex, localDim), planar.layout.n, localDim);
        return CompleteSinglePairPanelCube(opmm, localDim);
#else
        (void)stage;
        (void)pairIndex;
        (void)slotIndex;
        (void)opmm;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairRightPanelCubeBuffered(int stage, int pairIndex, int slotIndex,
                                                                   int bufferIndex, CMatmulCustom<float> *opmm)
    {
        return ProcessSinglePairPanelCubeBuffered<false>(stage, pairIndex, slotIndex, bufferIndex, opmm);
    }

    __aicore__ inline bool ProcessRightPanelTaskCube(int stage, int taskIndex, int taskOrdinal,
                                                     CMatmulCustom<float> *opmm)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        const int sliceCount = PanelTaskSliceCount();
        const int pairIndex = taskIndex / sliceCount;
        const int sliceIndex = taskIndex - pairIndex * sliceCount;
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        const int slotCount = PanelServiceSlotCount();
        if (!CanUseSinglePairRightPanel(pair, localDim) || slotCount <= 0 || opmm == nullptr)
        {
            return false;
        }
        const int slotIndex = pairIndex % slotCount;
        const int bufferIndex = pairIndex / slotCount;
        if (bufferIndex < 0 || bufferIndex > 1)
        {
            return false;
        }

        const int ping = taskOrdinal & 1;
        const int sliceBegin = sliceIndex * CHEEVJ_STAGE_BLOCK_JACOBI16_TILE;
        const int sliceElems = MinInt(CHEEVJ_STAGE_BLOCK_JACOBI16_TILE, planar.layout.n - sliceBegin);
        GlobalTensor<float> panelReal = RightPanelBufferReal(slotIndex, localDim, bufferIndex);
        GlobalTensor<float> panelImag = RightPanelBufferImag(slotIndex, localDim, bufferIndex);
        CrossCoreWaitFlag(RightPanelPackReadyFlag(ping));
        opmm->SetMatrixWithAImagNeg(StoredUnitaryReal(pairIndex), StoredUnitaryImag(pairIndex),
                                    StoredUnitaryImagNeg(pairIndex), panelReal, panelImag, panelReal, panelImag,
                                    planar.layout.n, localDim);
        opmm->Process(0, sliceBegin, sliceBegin, localDim, sliceElems, localDim);
        SignalPanelGemmDone(ping);
        return true;
#else
        (void)stage;
        (void)taskIndex;
        (void)taskOrdinal;
        (void)opmm;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairLeftPanelCube(int stage, int pairIndex, int slotIndex,
                                                          CMatmulCustom<float> *opmm)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanProcessSinglePairPanelCube(pair, localDim, slotIndex, opmm))
        {
            return false;
        }

        GlobalTensor<float> panelInputReal = PanelSlotInputReal(slotIndex, localDim);
        GlobalTensor<float> panelInputImag = PanelSlotInputImag(slotIndex, localDim);
        GlobalTensor<float> panelOutputReal = PanelSlotOutputReal(slotIndex, localDim);
        GlobalTensor<float> panelOutputImag = PanelSlotOutputImag(slotIndex, localDim);
        CrossCoreWaitFlag(CHEEVJ_STAGE_PANEL_PACK_READY_FLAG);
        opmm->SetMatrix(planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 0)],
                        planar.scratch2[PackedUnitaryStorageOffset(pairIndex, 1)], panelInputReal, panelInputImag,
                        LeftPanelImagNeg(slotIndex, localDim), panelOutputReal, panelOutputImag, planar.layout.n,
                        localDim);
        return CompleteSinglePairPanelCube(opmm, localDim);
#else
        (void)stage;
        (void)pairIndex;
        (void)slotIndex;
        (void)opmm;
        return false;
#endif
    }

    __aicore__ inline bool ProcessSinglePairLeftPanelCubeBuffered(int stage, int pairIndex, int slotIndex,
                                                                  int bufferIndex, CMatmulCustom<float> *opmm)
    {
        return ProcessSinglePairPanelCubeBuffered<true>(stage, pairIndex, slotIndex, bufferIndex, opmm);
    }

    template <bool LeftUpdate>
    __aicore__ inline bool ProcessSinglePairPanelCubeBuffered(int stage, int pairIndex, int slotIndex,
                                                               int bufferIndex, CMatmulCustom<float> *opmm)
    {
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
        const CheevjStageBlockPair16 pair = schedule.PairForStage(stage, pairIndex);
        const int localDim = pair.valid ? pair.left.size + pair.right.size : 0;
        if (!CanUseSinglePairRightPanel(pair, localDim) || slotIndex < 0 || slotIndex >= PanelServiceSlotCount() ||
            bufferIndex < 0 || bufferIndex > 1 || opmm == nullptr)
        {
            return false;
        }

        GlobalTensor<float> panelReal = RightPanelBufferReal(slotIndex, localDim, bufferIndex);
        GlobalTensor<float> panelImag = RightPanelBufferImag(slotIndex, localDim, bufferIndex);
        CrossCoreWaitFlag(RightPanelPackReadyFlag(bufferIndex));
        if constexpr (LeftUpdate)
        {
            opmm->SetMatrixWithAImagNeg(StoredUnitaryReal(pairIndex), StoredUnitaryImagNeg(pairIndex),
                                        StoredUnitaryImag(pairIndex), panelReal, panelImag, panelReal, panelImag,
                                        planar.layout.n, localDim);
        }
        else
        {
            opmm->SetMatrixWithAImagNeg(StoredUnitaryReal(pairIndex), StoredUnitaryImag(pairIndex),
                                        StoredUnitaryImagNeg(pairIndex), panelReal, panelImag, panelReal, panelImag,
                                        planar.layout.n, localDim);
        }
        opmm->Process(0, 0, 0, localDim, planar.layout.n, localDim);
        SignalPanelGemmDone(bufferIndex);
        return true;
#else
        (void)stage;
        (void)pairIndex;
        (void)slotIndex;
        (void)bufferIndex;
        (void)opmm;
        return false;
#endif
    }

    __aicore__ inline void SignalPanelGemmDone(int bufferIndex)
    {
        if (bufferIndex == 0)
        {
            CrossCoreSetFlag<0x2, PIPE_FIX>(CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG);
        }
        else
        {
            CrossCoreSetFlag<0x2, PIPE_FIX>(CHEEVJ_STAGE_PANEL_GEMM_DONE_FLAG_PONG);
        }
    }
#endif
