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
 * \file cheevj_her2k.hpp
 * \brief Shared fixed-shape C220 complex HER2K for the Cheevj trailing update.
 */

#ifndef CHEEVJ_C64_HER2K_HPP
#define CHEEVJ_C64_HER2K_HPP

#include <cstdint>

#include "../../utils/kernel/c64/gemm.hpp"
#include "kernel_operator.h"

// 本地定义 GM_ADDR：kernel 编译单元不能使用 utils/gm_addr.h（原因见 cheevj_kernel.cpp 同宏定义处注释）。
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

namespace Cheevj
{

constexpr int CHEEVJ_HER2K_PANEL_WIDTH = 32;
constexpr int CHEEVJ_HER2K_CUBE_M = 128;
constexpr int CHEEVJ_HER2K_CUBE_N = 128;
constexpr int CHEEVJ_HER2K_VEC_TILE = 64;

// wHImag is -transpose(W.imag); wHImagNeg is +transpose(W.imag).
// Keeping this staging explicit lets the implementation measure the four-real-GEMM body independently.
template <int MatrixN>
__aicore__ inline void CheevjHer2kBuildD(GM_ADDR vReal, GM_ADDR vImag, GM_ADDR wHReal, GM_ADDR wHImag,
                                         GM_ADDR wHImagNeg, GM_ADDR dReal, GM_ADDR dImag, int activeN)
{
#if defined(__DAV_C310_CUBE__) || defined(__DAV_C220_CUBE__) || defined(__DAV_310R6_CUBE__)
    AscendC::GlobalTensor<float> vRealGlobal;
    AscendC::GlobalTensor<float> vImagGlobal;
    AscendC::GlobalTensor<float> wHRealGlobal;
    AscendC::GlobalTensor<float> wHImagGlobal;
    AscendC::GlobalTensor<float> wHImagNegGlobal;
    AscendC::GlobalTensor<float> dRealGlobal;
    AscendC::GlobalTensor<float> dImagGlobal;
    vRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vReal));
    vImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(vImag));
    wHRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wHReal));
    wHImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wHImag));
    wHImagNegGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(wHImagNeg));
    dRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dReal));
    dImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dImag));

    AscendC::TPipe pipe;
    CMatmulCustom<float> mm;
    mm.Init(&pipe);
    mm.SetMatrix(vRealGlobal, vImagGlobal, wHRealGlobal, wHImagGlobal, wHImagNegGlobal, dRealGlobal, dImagGlobal,
                 MatrixN, CHEEVJ_HER2K_PANEL_WIDTH);

    const int activeRowTiles = (activeN + CHEEVJ_HER2K_CUBE_M - 1) / CHEEVJ_HER2K_CUBE_M;
    const int activeColTiles = (activeN + CHEEVJ_HER2K_CUBE_N - 1) / CHEEVJ_HER2K_CUBE_N;
    const int taskCount = activeRowTiles * activeColTiles;
    const int block = AscendC::GetBlockIdx();
    const int blocks = AscendC::GetBlockNum();
    for (int task = block; task < taskCount; task += blocks)
    {
        // Column-major tile order shares one W panel across each fixed-shape core wave.
        const int tileCol = task / activeRowTiles;
        const int tileRow = task - tileCol * activeRowTiles;
        const int row = tileRow * CHEEVJ_HER2K_CUBE_M;
        const int col = tileCol * CHEEVJ_HER2K_CUBE_N;
        mm.Process(row * CHEEVJ_HER2K_PANEL_WIDTH, col, row * MatrixN + col, CHEEVJ_HER2K_CUBE_M, CHEEVJ_HER2K_CUBE_N,
                   CHEEVJ_HER2K_PANEL_WIDTH);
    }
    pipe.Destroy();
#else
    (void)vReal;
    (void)vImag;
    (void)wHReal;
    (void)wHImag;
    (void)wHImagNeg;
    (void)dReal;
    (void)dImag;
    (void)activeN;
#endif
}

template <int MatrixN>
class CheevjHer2kEpilogue
{
   public:
    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR cReal, GM_ADDR cImag, GM_ADDR dReal, GM_ADDR dImag,
                                GM_ADDR matrixReal, GM_ADDR matrixImag, int panelStart, int activeN)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        cRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cReal));
        cImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cImag));
        dRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dReal));
        dImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dImag));
        matrixRealGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixReal));
        matrixImagGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(matrixImag));
        matrixOffset = panelStart * MatrixN + panelStart;
        activeOrder = activeN;
        constexpr int tileBytes = CHEEVJ_HER2K_VEC_TILE * CHEEVJ_HER2K_VEC_TILE * sizeof(float);
        pipe->InitBuffer(tileBuf, tileBytes);
        pipe->InitBuffer(transposeBuf, tileBytes);
        pipe->InitBuffer(outputBuf, tileBytes);
#else
        (void)pipe;
        (void)cReal;
        (void)cImag;
        (void)dReal;
        (void)dImag;
        (void)matrixReal;
        (void)matrixImag;
        (void)panelStart;
        (void)activeN;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int tileCount = (activeOrder + CHEEVJ_HER2K_VEC_TILE - 1) / CHEEVJ_HER2K_VEC_TILE;
        int ordinal = 0;
        for (int distance = 0; distance < tileCount; ++distance)
        {
            for (int tileCol = 0; tileCol + distance < tileCount; ++tileCol, ++ordinal)
            {
                if (ordinal % AscendC::GetBlockNum() != AscendC::GetBlockIdx())
                {
                    continue;
                }
                ProcessTile(tileCol + distance, tileCol);
            }
        }
#endif
    }

   private:
    AscendC::GlobalTensor<float> cRealGlobal;
    AscendC::GlobalTensor<float> cImagGlobal;
    AscendC::GlobalTensor<float> dRealGlobal;
    AscendC::GlobalTensor<float> dImagGlobal;
    AscendC::GlobalTensor<float> matrixRealGlobal;
    AscendC::GlobalTensor<float> matrixImagGlobal;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tileBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> transposeBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> outputBuf;
    int matrixOffset = 0;
    int activeOrder = MatrixN;

    __aicore__ inline void LoadTile(const AscendC::LocalTensor<float>& dst, const AscendC::GlobalTensor<float>& src,
                                    int offset)
    {
        constexpr int tile = CHEEVJ_HER2K_VEC_TILE;
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::DataCopyExtParams params{tile, static_cast<uint32_t>(tile * sizeof(float)),
                                          static_cast<uint32_t>((MatrixN - tile) * sizeof(float)), 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(dst, src[offset], params, pad);
    }

    __aicore__ inline void StoreTile(AscendC::GlobalTensor<float> dst, const AscendC::LocalTensor<float>& src,
                                     int offset, int tileRow, int tileCol)
    {
        constexpr int tile = CHEEVJ_HER2K_VEC_TILE;
        const int validRows = Min(tile, activeOrder - tileRow * tile);
        const int validCols = Min(tile, activeOrder - tileCol * tile);
        AscendC::PipeBarrier<PIPE_ALL>();
        if (validCols == tile)
        {
            AscendC::DataCopyExtParams params{static_cast<uint16_t>(validRows),
                                              static_cast<uint32_t>(tile * sizeof(float)), 0,
                                              static_cast<uint32_t>((MatrixN - tile) * sizeof(float)), 0};
            AscendC::DataCopyPad(dst[offset], src, params);
        }
        else
        {
            AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(validCols * sizeof(float)), 0, 0, 0};
            for (int row = 0; row < validRows; ++row)
            {
                AscendC::DataCopyPad(dst[offset + row * MatrixN], src[row * tile], params);
            }
        }
    }

    __aicore__ inline void Transpose64(const AscendC::LocalTensor<float>& dst, const AscendC::LocalTensor<float>& src)
    {
        constexpr int tile = CHEEVJ_HER2K_VEC_TILE;
        constexpr int fp32PerBlock = 8;
        for (int rowGroup = 0; rowGroup < tile / 16; ++rowGroup)
        {
            uint64_t srcList[16];
            uint64_t dstList[16];
            for (int i = 0; i < 16; ++i)
            {
                srcList[i] = reinterpret_cast<uint64_t>(src[(rowGroup * 16 + i) * tile].GetPhyAddr());
                dstList[i] = reinterpret_cast<uint64_t>(
                    dst[rowGroup * 16 + (i / 2) * tile + (i % 2) * fp32PerBlock].GetPhyAddr());
            }
            AscendC::TransDataTo5HDParams params(false, false, tile / fp32PerBlock, tile, 1);
            AscendC::TransDataTo5HD<float>(dstList, srcList, params);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline int Min(int lhs, int rhs) const { return lhs < rhs ? lhs : rhs; }

    __aicore__ inline void ProcessPlane(AscendC::GlobalTensor<float> c, AscendC::GlobalTensor<float> d,
                                        AscendC::GlobalTensor<float> matrix, int lowerOffset, int upperOffset,
                                        int lowerTileRow, int lowerTileCol, bool imag, bool diagonal)
    {
        constexpr int elems = CHEEVJ_HER2K_VEC_TILE * CHEEVJ_HER2K_VEC_TILE;
        auto tile = tileBuf.Get<float>();
        auto transposed = transposeBuf.Get<float>();
        auto output = outputBuf.Get<float>();

        LoadTile(tile, d, upperOffset);
        AscendC::PipeBarrier<PIPE_ALL>();
        Transpose64(transposed, tile);
        LoadTile(tile, d, lowerOffset);
        LoadTile(output, c, lowerOffset);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Sub(output, output, tile, elems);
        AscendC::PipeBarrier<PIPE_V>();
        if (imag)
        {
            AscendC::Add(output, output, transposed, elems);
        }
        else
        {
            AscendC::Sub(output, output, transposed, elems);
        }
        AscendC::PipeBarrier<PIPE_V>();

        // On a diagonal tile, each imaginary diagonal entry is computed as
        // 0 - Di(i,i) + Di(i,i), so it is already exactly zero.  Do not issue
        // one-element Duplicate operations at the mostly non-32-byte-aligned
        // diagonal addresses: that scalar-shaped vector access faults on C220.
        StoreTile(matrix, output, matrixOffset + lowerOffset, lowerTileRow, lowerTileCol);
        AscendC::PipeBarrier<PIPE_ALL>();

        if (!diagonal)
        {
            Transpose64(transposed, output);
            if (imag)
            {
                AscendC::Muls(transposed, transposed, -1.0f, elems);
                AscendC::PipeBarrier<PIPE_V>();
            }
            StoreTile(matrix, transposed, matrixOffset + upperOffset, lowerTileCol, lowerTileRow);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void ProcessTile(int lowerTileRow, int lowerTileCol)
    {
        constexpr int tile = CHEEVJ_HER2K_VEC_TILE;
        const int lowerOffset = lowerTileRow * tile * MatrixN + lowerTileCol * tile;
        const int upperOffset = lowerTileCol * tile * MatrixN + lowerTileRow * tile;
        const bool diagonal = lowerTileRow == lowerTileCol;
        ProcessPlane(cRealGlobal, dRealGlobal, matrixRealGlobal, lowerOffset, upperOffset, lowerTileRow, lowerTileCol,
                     false, diagonal);
        ProcessPlane(cImagGlobal, dImagGlobal, matrixImagGlobal, lowerOffset, upperOffset, lowerTileRow, lowerTileCol,
                     true, diagonal);
    }
};

}  // namespace Cheevj

#endif  // CHEEVJ_C64_HER2K_HPP
