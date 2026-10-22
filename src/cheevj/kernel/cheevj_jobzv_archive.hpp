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
 * \file cheevj_jobzv_archive.hpp
 * \brief Shared fixed-shape reflector archive implementation.
 */

#ifndef CHEEVJ_C64_JOBZV_ARCHIVE_HPP
#define CHEEVJ_C64_JOBZV_ARCHIVE_HPP

#include <cstdint>

#include "kernel_operator.h"

namespace Cheevj
{

template <int MatrixN, int PanelWidth, int PanelLeadingDimension, int GatherTile>
class CheevjArchivePanelReflectors
{
   public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR panelVReal, GM_ADDR panelVImag, GM_ADDR reflectorReal,
                                GM_ADDR reflectorImag, int panelStart, int blockOffset = 0)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        panelVRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelVReal));
        panelVImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(panelVImag));
        reflectorRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reflectorReal));
        reflectorImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(reflectorImag));
        start = panelStart;
        archiveBlockOffset = blockOffset;
        pipe->InitBuffer(stridedBuf, GatherTile * 8 * sizeof(float));
        pipe->InitBuffer(denseBuf, GatherTile * sizeof(float));
        pipe->InitBuffer(gatherIndexBuf, GatherTile * sizeof(uint32_t));
#else
        (void)pipe;
        (void)panelVReal;
        (void)panelVImag;
        (void)reflectorReal;
        (void)reflectorImag;
        (void)panelStart;
        (void)blockOffset;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int localStep = static_cast<int>(AscendC::GetBlockIdx()) - archiveBlockOffset;
        if (AscendC::GetBlockNum() < archiveBlockOffset + PanelWidth || localStep < 0 || localStep >= PanelWidth ||
            start < 0 || start >= MatrixN || (start % PanelWidth) != 0)
        {
            return;
        }
        const int globalStep = start + localStep;
        if (globalStep >= MatrixN)
        {
            return;
        }
        const int localBegin = localStep + 1;
        const int globalBegin = globalStep + 1;
        const int active = MatrixN - globalBegin;
        if (active <= 0)
        {
            return;
        }
        PrepareGatherIndex();
        ArchiveOnePlane(panelVRealGlobal, reflectorRealGlobal, localStep, localBegin, globalStep, globalBegin, active);
        ArchiveOnePlane(panelVImagGlobal, reflectorImagGlobal, localStep, localBegin, globalStep, globalBegin, active);
#endif
    }

   private:
    AscendC::GlobalTensor<float> panelVRealGlobal;
    AscendC::GlobalTensor<float> panelVImagGlobal;
    AscendC::GlobalTensor<float> reflectorRealGlobal;
    AscendC::GlobalTensor<float> reflectorImagGlobal;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridedBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> denseBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndexBuf;
    int start = 0;
    int archiveBlockOffset = 0;

    __aicore__ inline void PrepareGatherIndex()
    {
        auto signedIndex = gatherIndexBuf.Get<int32_t>();
        auto index = signedIndex.template ReinterpretCast<uint32_t>();
        AscendC::CreateVecIndex(signedIndex, static_cast<int32_t>(0), static_cast<uint32_t>(GatherTile));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(5), GatherTile);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ArchiveOnePlane(const AscendC::GlobalTensor<float> &panel,
                                           AscendC::GlobalTensor<float> archive, int localStep, int localBegin,
                                           int globalStep, int globalBegin, int active)
    {
        auto strided = stridedBuf.Get<float>();
        auto dense = denseBuf.Get<float>();
        auto index = gatherIndexBuf.Get<uint32_t>();
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        const uint32_t sourceStride = static_cast<uint32_t>((PanelLeadingDimension - 1) * sizeof(float));
        for (int tileBegin = 0; tileBegin < active; tileBegin += GatherTile)
        {
            const int remaining = active - tileBegin;
            const int valid = remaining < GatherTile ? remaining : GatherTile;
            AscendC::DataCopyExtParams gatherCopy{static_cast<uint16_t>(valid), static_cast<uint32_t>(sizeof(float)),
                                                  sourceStride, 0, 0};
            const int panelOffset = (localBegin + tileBegin) * PanelLeadingDimension + localStep;
            AscendC::DataCopyPad(strided, panel[panelOffset], gatherCopy, noPad);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::Gather(dense, strided, index, 0, valid);
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams storeCopy{1, static_cast<uint32_t>(valid * sizeof(float)), 0, 0, 0};
            const int archiveOffset = globalStep * MatrixN + globalBegin + tileBegin;
            AscendC::DataCopyPad(archive[archiveOffset], dense, storeCopy);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_JOBZV_ARCHIVE_HPP
