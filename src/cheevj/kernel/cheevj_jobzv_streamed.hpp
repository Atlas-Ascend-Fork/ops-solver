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
 * \file cheevj_jobzv_streamed.hpp
 * \brief Shared streamed fixed-shape eigenvector backtransform.
 */

#ifndef CHEEVJ_C64_JOBZV_STREAMED_HPP
#define CHEEVJ_C64_JOBZV_STREAMED_HPP

#include <cstdint>

#include "kernel_operator.h"

namespace Cheevj
{

template <int MatrixN, int Workers, int ColumnsPerWave, int Waves, int LocalAllocation, int GatherTile,
          int ReductionLanes>
class CheevjStreamedJobzVBacktransform
{
   public:
#include "cheevj_jobzv_init.inc"

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != Workers || block < 0 || block >= Workers || LoadInfo() != 0)
        {
            return;
        }
        auto localReal = localRealBuf.Get<float>();
        auto localImag = localImagBuf.Get<float>();
        auto tauReal = tauRealBuf.Get<float>();
        auto tauImag = tauImagBuf.Get<float>();
        PrepareGatherIndex();
        CopyIn(tauReal, tauRealGlobal, MatrixN);
        CopyIn(tauImag, tauImagGlobal, MatrixN);
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int wave = 0; wave < Waves; ++wave)
        {
            const int columnBegin = (wave * Workers + block) * ColumnsPerWave;
            LoadRowMajorColumns(columnBegin, localReal);
            AscendC::Duplicate(localImag, 0.0f, MatrixN * ColumnsPerWave);
            AscendC::PipeBarrier<PIPE_ALL>();
            ApplyReflectors(localReal, localImag, tauReal, tauImag);
            const int outputOffset = columnBegin * MatrixN;
            CopyOut(eigenvectorRealGlobal, outputOffset, localReal, MatrixN * ColumnsPerWave);
            CopyOut(eigenvectorImagGlobal, outputOffset, localImag, MatrixN * ColumnsPerWave);
        }
#endif
    }

   private:
#include "cheevj_jobzv_common.inc"
    AscendC::TBuf<AscendC::TPosition::VECCALC> localRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> localImagBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stridedBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gatherIndexBuf;
#include "cheevj_jobzv_buffers.inc"

    __aicore__ inline void InitBuffers(AscendC::TPipe *pipe)
    {
        pipe->InitBuffer(localRealBuf, LocalAllocation * sizeof(float));
        pipe->InitBuffer(localImagBuf, LocalAllocation * sizeof(float));
        pipe->InitBuffer(stridedBuf, GatherTile * 8 * sizeof(float));
        pipe->InitBuffer(gatherIndexBuf, GatherTile * sizeof(uint32_t));
        InitCommonBuffers(pipe, MatrixN, ReductionLanes);
    }

    __aicore__ inline void PrepareGatherIndex()
    {
        auto signedIndex = gatherIndexBuf.Get<int32_t>();
        auto index = signedIndex.template ReinterpretCast<uint32_t>();
        AscendC::CreateVecIndex(signedIndex, static_cast<int32_t>(0), static_cast<uint32_t>(GatherTile));
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(5), GatherTile);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadRowMajorColumns(int columnBegin, const AscendC::LocalTensor<float> &localReal)
    {
        for (int localColumn = 0; localColumn < ColumnsPerWave; ++localColumn)
        {
            const int globalColumn = columnBegin + localColumn;
            CopyIn(localReal[localColumn * MatrixN], tridiagonalEigenvectorsGlobal[globalColumn * MatrixN], MatrixN);
        }
    }

    __aicore__ inline void LoadReflector(int step)
    {
        const int offset = step * MatrixN;
        CopyIn(reflectorRealBuf.Get<float>(), reflectorRealGlobal[offset], MatrixN);
        CopyIn(reflectorImagBuf.Get<float>(), reflectorImagGlobal[offset], MatrixN);
    }

    __aicore__ inline float Reduce(const AscendC::LocalTensor<float> &source)
    {
        constexpr int segments = MatrixN / ReductionLanes;
        auto reduction = reductionBuf.Get<float>();
        AscendC::WholeReduceSum<float, true>(reduction, source, ReductionLanes, static_cast<int32_t>(segments), 1, 1,
                                             ReductionLanes / 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::WholeReduceSum<float, true>(reduction[segments], reduction, segments, static_cast<int32_t>(1), 1, 1,
                                             segments);
        AscendC::PipeBarrier<PIPE_ALL>();
        return reduction.GetValue(segments);
    }

    __aicore__ inline void ApplyReflectors(const AscendC::LocalTensor<float> &localReal,
                                           const AscendC::LocalTensor<float> &localImag,
                                           const AscendC::LocalTensor<float> &tauReal,
                                           const AscendC::LocalTensor<float> &tauImag)
    {
        for (int step = MatrixN - 2; step >= 0; --step)
        {
            const float tauR = tauReal.GetValue(step);
            const float tauI = tauImag.GetValue(step);
            if (tauR == 0.0f && tauI == 0.0f)
            {
                continue;
            }
            LoadReflector(step);
            for (int localColumn = 0; localColumn < ColumnsPerWave; ++localColumn)
            {
                ApplyReflector(tauR, tauI, localReal[localColumn * MatrixN], localImag[localColumn * MatrixN]);
            }
        }
    }

    __aicore__ inline void ApplyReflector(float tauR, float tauI, const AscendC::LocalTensor<float> &vectorReal,
                                          const AscendC::LocalTensor<float> &vectorImag)
    {
        auto reflectorReal = reflectorRealBuf.Get<float>();
        auto reflectorImag = reflectorImagBuf.Get<float>();
        auto scratch0 = scratch0Buf.Get<float>();
        auto scratch1 = scratch1Buf.Get<float>();
        auto scratch2 = scratch2Buf.Get<float>();
        AscendC::Mul(scratch0, reflectorReal, vectorReal, MatrixN);
        AscendC::Mul(scratch1, reflectorImag, vectorImag, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch0, scratch0, scratch1, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        const float dotReal = Reduce(scratch0);
        AscendC::Mul(scratch1, reflectorReal, vectorImag, MatrixN);
        AscendC::Mul(scratch2, reflectorImag, vectorReal, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(scratch1, scratch1, scratch2, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        const float dotImag = Reduce(scratch1);
        const float factorReal = tauR * dotReal - tauI * dotImag;
        const float factorImag = tauR * dotImag + tauI * dotReal;
        AscendC::Muls(scratch0, reflectorReal, factorReal, MatrixN);
        AscendC::Muls(scratch1, reflectorImag, factorImag, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(scratch0, scratch0, scratch1, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(vectorReal, vectorReal, scratch0, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scratch1, reflectorReal, factorImag, MatrixN);
        AscendC::Muls(scratch2, reflectorImag, factorReal, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch1, scratch1, scratch2, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(vectorImag, vectorImag, scratch1, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_JOBZV_STREAMED_HPP
