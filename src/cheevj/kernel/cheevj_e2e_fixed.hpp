/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */

/*!
 * \file cheevj_e2e_fixed.hpp
 * \brief Shared fixed-shape jobz=N assembly helpers.
 *
 * The assembly packs A(s:n,s:n) into the upper-left corner of a zero-padded
 * fixed-shape matrix, runs an nb=4 panel and the device HER2K update, and
 * unpacks only the valid trailing square. All numeric work remains on device.
 */

#ifndef CHEEVJ_C64_E2E_FIXED_HPP
#define CHEEVJ_C64_E2E_FIXED_HPP

#include <cstdint>

#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

namespace Cheevj
{

constexpr int CHEEVJ_FIXED_PANEL_WIDTH = 32;

__aicore__ inline void CopyFloatRow(const AscendC::LocalTensor<float>& row, const AscendC::GlobalTensor<float>& source,
                                    int sourceOffset, AscendC::GlobalTensor<float> destination, int destinationOffset,
                                    int count)
{
    AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(row, source[sourceOffset], copy, pad);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyPad(destination[destinationOffset], row, copy);
    AscendC::PipeBarrier<PIPE_ALL>();
}

template <int MatrixN>
class CheevjE2EPreparePacked
{
   public:
    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR packedReal,
                                GM_ADDR packedImag, GM_ADDR panelVReal, GM_ADDR panelVImag, GM_ADDR panelWReal,
                                GM_ADDR panelWImag, GM_ADDR wHReal, GM_ADDR wHImag, GM_ADDR wHImagNeg,
                                GM_ADDR updateReal, GM_ADDR updateImag, GM_ADDR panelWorkspace,
                                GM_ADDR barrierWorkspace, int panelStart)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        matrixRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixReal));
        matrixImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixImag));
        packedRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedReal));
        packedImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedImag));
        panelVRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panelVReal));
        panelVImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panelVImag));
        panelWRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panelWReal));
        panelWImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panelWImag));
        wHRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wHReal));
        wHImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wHImag));
        wHImagNegGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wHImagNeg));
        updateRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(updateReal));
        updateImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(updateImag));
        panelWorkspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(panelWorkspace));
        barrierWorkspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(barrierWorkspace));
        start = panelStart;

        pipe->InitBuffer(zeroBuf, MatrixN * sizeof(float));
        pipe->InitBuffer(rowRealBuf, MatrixN * sizeof(float));
        pipe->InitBuffer(rowImagBuf, MatrixN * sizeof(float));
#else
        (void)pipe;
        (void)matrixReal;
        (void)matrixImag;
        (void)packedReal;
        (void)packedImag;
        (void)panelVReal;
        (void)panelVImag;
        (void)panelWReal;
        (void)panelWImag;
        (void)wHReal;
        (void)wHImag;
        (void)wHImagNeg;
        (void)updateReal;
        (void)updateImag;
        (void)panelWorkspace;
        (void)barrierWorkspace;
        (void)panelStart;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        // panelSize 按形状取值，须满足 host 侧分配上界 CHEEVJ_FIXED_PANEL_SIZE=16
        // （cheevj_fixed_host.inc 经 static_assert 绑定，issue #160）；
        // 布局消费公式 4*panelSize*MatrixN + 2*MatrixN 与 host 分配公式逐项对应
        constexpr int panelSize = MatrixN == 512 ? 4 : (MatrixN == 2048 ? 16 : 8);
        constexpr int barrierInts = MatrixN == 512 ? 33 * 8 : 33 * 16;
        if (start < 0 || start >= MatrixN || (start % panelSize) != 0)
        {
            return;
        }
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        const int blocks = static_cast<int>(AscendC::GetBlockNum());
        if (blocks <= 0 || block < 0 || block >= blocks)
        {
            return;
        }

        auto zero = zeroBuf.Get<float>();
        AscendC::Duplicate(zero, 0.0f, MatrixN);
        AscendC::PipeBarrier<PIPE_ALL>();

        // Kernel completion is the inter-stage barrier.  Clearing here (as
        // opposed to host memset) keeps every panel transition stream ordered.
        ClearPlane(packedRealGlobal, (MatrixN * MatrixN), zero, block, blocks);
        ClearPlane(packedImagGlobal, (MatrixN * MatrixN), zero, block, blocks);
        ClearPlane(panelVRealGlobal, (MatrixN * CHEEVJ_FIXED_PANEL_WIDTH), zero, block, blocks);
        ClearPlane(panelVImagGlobal, (MatrixN * CHEEVJ_FIXED_PANEL_WIDTH), zero, block, blocks);
        ClearPlane(wHRealGlobal, (MatrixN * CHEEVJ_FIXED_PANEL_WIDTH), zero, block, blocks);
        ClearPlane(wHImagGlobal, (MatrixN * CHEEVJ_FIXED_PANEL_WIDTH), zero, block, blocks);
        ClearPlane(wHImagNegGlobal, (MatrixN * CHEEVJ_FIXED_PANEL_WIDTH), zero, block, blocks);
        ClearPlane(panelWorkspaceGlobal, (4 * panelSize * MatrixN + 2 * MatrixN), zero, block, blocks);
        ClearPlane(barrierWorkspaceGlobal, barrierInts, zero.template ReinterpretCast<int32_t>(), block, blocks);

        // Each packed row and its clearing chunk have the same
        // owner (row % blocks), so no cross-core barrier is required here.
        const int trailing = MatrixN - start;
        auto rowReal = rowRealBuf.Get<float>();
        auto rowImag = rowImagBuf.Get<float>();
        for (int row = block; row < trailing; row += blocks)
        {
            const int sourceOffset = (start + row) * MatrixN + start;
            const int packedOffset = row * MatrixN;
            CopyFloatRow(rowReal, matrixRealGlobal, sourceOffset, packedRealGlobal, packedOffset, trailing);
            CopyFloatRow(rowImag, matrixImagGlobal, sourceOffset, packedImagGlobal, packedOffset, trailing);
        }
#endif
    }

   private:
    AscendC::GlobalTensor<float> matrixRealGlobal;
    AscendC::GlobalTensor<float> matrixImagGlobal;
    AscendC::GlobalTensor<float> packedRealGlobal;
    AscendC::GlobalTensor<float> packedImagGlobal;
    AscendC::GlobalTensor<float> panelVRealGlobal;
    AscendC::GlobalTensor<float> panelVImagGlobal;
    AscendC::GlobalTensor<float> panelWRealGlobal;
    AscendC::GlobalTensor<float> panelWImagGlobal;
    AscendC::GlobalTensor<float> wHRealGlobal;
    AscendC::GlobalTensor<float> wHImagGlobal;
    AscendC::GlobalTensor<float> wHImagNegGlobal;
    AscendC::GlobalTensor<float> updateRealGlobal;
    AscendC::GlobalTensor<float> updateImagGlobal;
    AscendC::GlobalTensor<float> panelWorkspaceGlobal;
    AscendC::GlobalTensor<int32_t> barrierWorkspaceGlobal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> zeroBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowImagBuf;
    int start = 0;

    __aicore__ inline int Min(int lhs, int rhs) const { return lhs < rhs ? lhs : rhs; }

    template <typename T>
    __aicore__ inline void ClearPlane(AscendC::GlobalTensor<T> destination, int count,
                                      const AscendC::LocalTensor<T>& zero, int block, int blocks)
    {
        AscendC::PipeBarrier<PIPE_ALL>();
        for (int offset = block * MatrixN; offset < count; offset += blocks * MatrixN)
        {
            const int chunk = Min(MatrixN, count - offset);
            AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(chunk * sizeof(T)), 0, 0, 0};
            AscendC::DataCopyPad(destination[offset], zero, copy);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

template <int MatrixN>
class CheevjE2EUnpack
{
   public:
    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR packedReal, GM_ADDR packedImag, GM_ADDR matrixReal,
                                GM_ADDR matrixImag, int panelStart)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        packedRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedReal));
        packedImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(packedImag));
        matrixRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixReal));
        matrixImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixImag));
        start = panelStart;
        pipe->InitBuffer(rowRealBuf, MatrixN * sizeof(float));
        pipe->InitBuffer(rowImagBuf, MatrixN * sizeof(float));
#else
        (void)pipe;
        (void)packedReal;
        (void)packedImag;
        (void)matrixReal;
        (void)matrixImag;
        (void)panelStart;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        const int blocks = static_cast<int>(AscendC::GetBlockNum());
        if (start < 0 || start >= MatrixN || blocks <= 0)
        {
            return;
        }
        const int trailing = MatrixN - start;
        auto rowReal = rowRealBuf.Get<float>();
        auto rowImag = rowImagBuf.Get<float>();
        for (int row = block; row < trailing; row += blocks)
        {
            const int packedOffset = row * MatrixN;
            const int destinationOffset = (start + row) * MatrixN + start;
            CopyFloatRow(rowReal, packedRealGlobal, packedOffset, matrixRealGlobal, destinationOffset, trailing);
            CopyFloatRow(rowImag, packedImagGlobal, packedOffset, matrixImagGlobal, destinationOffset, trailing);
        }
#endif
    }

   private:
    AscendC::GlobalTensor<float> packedRealGlobal;
    AscendC::GlobalTensor<float> packedImagGlobal;
    AscendC::GlobalTensor<float> matrixRealGlobal;
    AscendC::GlobalTensor<float> matrixImagGlobal;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowImagBuf;
    int start = 0;
};

template <int MatrixN>
class CheevjE2EFinalize
{
   public:
    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal,
                                GM_ADDR tauImag, GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR bounds, GM_ADDR info)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        diagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(diagonal));
        offDiagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(offDiagonal));
        tauRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tauReal));
        tauImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tauImag));
        matrixRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixReal));
        matrixImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixImag));
        boundsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(bounds));
        infoGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(info));

        constexpr int vectorBytes = MatrixN * sizeof(float);
        pipe->InitBuffer(rowRealBuf, vectorBytes);
        pipe->InitBuffer(rowImagBuf, vectorBytes);
        pipe->InitBuffer(diagonalBuf, vectorBytes);
        pipe->InitBuffer(leftEdgeBuf, vectorBytes);
        pipe->InitBuffer(rightEdgeBuf, vectorBytes);
        pipe->InitBuffer(lowerBuf, vectorBytes);
        pipe->InitBuffer(upperBuf, vectorBytes);
        pipe->InitBuffer(scratchBuf, vectorBytes);
        pipe->InitBuffer(stagingBuf, 64 * sizeof(float));
#else
        (void)pipe;
        (void)diagonal;
        (void)offDiagonal;
        (void)tauReal;
        (void)tauImag;
        (void)matrixReal;
        (void)matrixImag;
        (void)bounds;
        (void)info;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        const int blocks = static_cast<int>(AscendC::GetBlockNum());
        if (blocks <= 0 || block < 0 || block >= blocks)
        {
            return;
        }
        if (block == 0)
        {
            EnforceTailConventionAndStoreBounds();
        }
        for (int row = block; row < MatrixN; row += blocks)
        {
            RebuildRow(row);
        }
#endif
    }

   private:
    AscendC::GlobalTensor<float> diagonalGlobal;
    AscendC::GlobalTensor<float> offDiagonalGlobal;
    AscendC::GlobalTensor<float> tauRealGlobal;
    AscendC::GlobalTensor<float> tauImagGlobal;
    AscendC::GlobalTensor<float> matrixRealGlobal;
    AscendC::GlobalTensor<float> matrixImagGlobal;
    AscendC::GlobalTensor<float> boundsGlobal;
    AscendC::GlobalTensor<int32_t> infoGlobal;

    AscendC::TBuf<AscendC::TPosition::VECCALC> rowRealBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowImagBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> diagonalBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> leftEdgeBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> rightEdgeBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> lowerBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> upperBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> stagingBuf;

    __aicore__ inline float Abs(float value) const { return value < 0.0f ? -value : value; }

    __aicore__ inline float Max(float lhs, float rhs) const { return lhs > rhs ? lhs : rhs; }

    __aicore__ inline float LoadControl(const AscendC::GlobalTensor<float>& source, int offset)
    {
        auto staging = stagingBuf.Get<float>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{true, 0, 7, 0.0f};
        AscendC::DataCopyPad(staging, source[offset], copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
        return staging.GetValue(0);
    }

    __aicore__ inline void StoreZero(AscendC::GlobalTensor<float> destination, int offset)
    {
        auto staging = stagingBuf.Get<float>();
        AscendC::Duplicate(staging, 0.0f, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams copy{1, sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(destination[offset], staging, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteLane(const AscendC::LocalTensor<float>& destination, int lane, float value)
    {
        const int base = lane & ~63;
        const int bit = lane - base;
        uint64_t mask[2] = {1ULL << static_cast<uint32_t>(bit), 0ULL};
        AscendC::Duplicate(destination[base], value, mask, static_cast<uint8_t>(1), 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void EnforceTailConventionAndStoreBounds()
    {
        // last panel processes the final three real reflectors.  Its padded local
        // step 3 extracts the final diagonal and leaves the non-existent e/tau tail zero.
        StoreZero(offDiagonalGlobal, MatrixN - 1);
        StoreZero(tauRealGlobal, MatrixN - 1);
        StoreZero(tauImagGlobal, MatrixN - 1);

        auto diagonal = diagonalBuf.Get<float>();
        auto left = leftEdgeBuf.Get<float>();
        auto right = rightEdgeBuf.Get<float>();
        auto lower = lowerBuf.Get<float>();
        auto upper = upperBuf.Get<float>();
        auto scratch = scratchBuf.Get<float>();
        AscendC::DataCopyExtParams diagonalCopy{1, MatrixN * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(diagonal, diagonalGlobal, diagonalCopy, noPad);

        AscendC::DataCopyExtParams edgeCopy{1, (MatrixN - 1) * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> leftPad{true, 1, 0, 0.0f};
        AscendC::DataCopyPadExtParams<float> rightPad{true, 0, 1, 0.0f};
        AscendC::DataCopyPad(left, offDiagonalGlobal, edgeCopy, leftPad);
        AscendC::DataCopyPad(right, offDiagonalGlobal, edgeCopy, rightPad);
        AscendC::PipeBarrier<PIPE_ALL>();

        AscendC::Abs(left, left, MatrixN);
        AscendC::Abs(right, right, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(scratch, left, right, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(lower, diagonal, scratch, MatrixN);
        AscendC::Add(upper, diagonal, scratch, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();

        // min(lower) = -max(-lower); scalar reads are reduction controls only.
        AscendC::Muls(lower, lower, -1.0f, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceMax(lower, lower, scratch, MatrixN, true);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::ReduceMax(upper, upper, scratch, MatrixN, true);
        AscendC::PipeBarrier<PIPE_ALL>();
        float globalLow = -lower.GetValue(0);
        float globalHigh = upper.GetValue(0);
        const float scale = Max(1.0f, Max(Abs(globalLow), Abs(globalHigh)));
        const float margin = 64.0f * 1.1920928955078125e-7f * scale;
        globalLow -= margin;
        globalHigh += margin;

        StoreBoundsAndInfo(globalLow, globalHigh);
    }

    __aicore__ inline void StoreBoundsAndInfo(float globalLow, float globalHigh)
    {
        auto staging = stagingBuf.Get<float>();
        AscendC::Duplicate(staging, globalLow, 8);
        uint64_t laneOne[2] = {2ULL, 0ULL};
        AscendC::Duplicate(staging, globalHigh, laneOne, static_cast<uint8_t>(1), 1, 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams boundsCopy{1, 2 * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(boundsGlobal, staging, boundsCopy);
        AscendC::PipeBarrier<PIPE_ALL>();

        auto info = staging.template ReinterpretCast<int32_t>();
        AscendC::Duplicate(info, static_cast<int32_t>(0), 8);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams infoCopy{1, sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPad(infoGlobal, info, infoCopy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RebuildRow(int row)
    {
        auto rowReal = rowRealBuf.Get<float>();
        auto rowImag = rowImagBuf.Get<float>();
        AscendC::Duplicate(rowReal, 0.0f, MatrixN);
        AscendC::Duplicate(rowImag, 0.0f, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();

        WriteLane(rowReal, row, LoadControl(diagonalGlobal, row));
        if (row > 0)
        {
            WriteLane(rowReal, row - 1, LoadControl(offDiagonalGlobal, row - 1));
        }
        if (row + 1 < MatrixN)
        {
            WriteLane(rowReal, row + 1, LoadControl(offDiagonalGlobal, row));
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        const int offset = row * MatrixN;
        AscendC::DataCopyExtParams copy{1, MatrixN * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(matrixRealGlobal[offset], rowReal, copy);
        AscendC::DataCopyPad(matrixImagGlobal[offset], rowImag, copy);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_E2E_FIXED_HPP
