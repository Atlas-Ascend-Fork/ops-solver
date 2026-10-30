/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/kernel/basic_matmul_tla.hpp"
#include "catlass/gemm/tile/tile_copy_tla.hpp"
#include "catlass/layout/layout.hpp"
#include "cholesky_internal.h"
#include "tla/tensor.hpp"

namespace Cholesky
{
using namespace Catlass;
using Policy = Gemm::MmadPingpong<Arch::Ascend950, true, false, 1, false, 2, 2, 2, 2>;
using L1Shape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<64>>;
using L0Shape = tla::Shape<tla::Int<128>, tla::Int<128>, tla::Int<32>>;
using Copy = Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, float, layout::RowMajor, float, layout::ColumnMajor, float,
                                           layout::RowMajor, void>;
// All products use ElementSrc=void: the pinned BlockMmadTla resets L0C on
// the first K tile and overwrites destination GM (beta=0); it never reads old
// product contents. Preserve this contract when changing the policy/epilogue.
using Mmad = Gemm::Block::BlockMmadTla<Policy, L1Shape, L0Shape, float, float, float, void, Copy>;

__global__ __aicore__ void GramKernel(GM_ADDR wAddr, GM_ADDR pAddr, int pitch, int n, int start, int width, int count,
                                      GM_ADDR infoAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (g_coreType == AscendC::AIV) return;
    Arch::Resource<Arch::Ascend950> resource;
    Mmad mm(resource);
    int m = n - start - width, tiles = (m + 127) / 128;
    AscendC::GlobalTensor<float> w, p;
    AscendC::GlobalTensor<int> info;
    w.SetGlobalBuffer((__gm__ float *)wAddr);
    p.SetGlobalBuffer((__gm__ float *)pAddr);
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    for (int task = AscendC::GetBlockIdx(); task < count * tiles * tiles; task += AscendC::GetBlockNum())
    {
        int q = task / (tiles * tiles), tile = task % (tiles * tiles);
        int i = tile / tiles * 128, j = tile % tiles * 128;
        if (j > i || info.GetValue(q)) continue;
        int mi = m - i < 128 ? m - i : 128, nj = m - j < 128 ? m - j : 128;
        int64_t base = int64_t(q) * pitch * pitch;
        auto a = tla::MakeTensor(w[base + int64_t(start + width + i) * pitch + start],
                                 tla::MakeLayout(tla::MakeShape(mi, width), tla::MakeStride(pitch, tla::Int<1>{})),
                                 Arch::PositionGM{});
        auto b = tla::MakeTensor(w[base + int64_t(start + width + j) * pitch + start],
                                 tla::MakeLayout(tla::MakeShape(width, nj), tla::MakeStride(tla::Int<1>{}, pitch)),
                                 Arch::PositionGM{});
        auto c = tla::MakeTensor(p[base + int64_t(i) * pitch + j],
                                 tla::MakeLayout(tla::MakeShape(mi, nj), tla::MakeStride(pitch, tla::Int<1>{})),
                                 Arch::PositionGM{});
        mm(a, b, c, GemmCoord(mi, nj, width));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

__aicore__ inline __gm__ float *MatrixPointer(GM_ADDR single, GM_ADDR pointers, int index)
{
    if (!pointers) return (__gm__ float *)single;
    AscendC::GlobalTensor<uint64_t> array;
    array.SetGlobalBuffer((__gm__ uint64_t *)pointers);
    return (__gm__ float *)array.GetValue(index);
}

// Left-looking panel products. Only the next 64 columns are materialized;
// completed factors remain in the caller's selected triangle.
template <bool Upper>
__global__ __aicore__ void LeftGramKernel(GM_ADDR aAddr, GM_ADDR aArrayAddr, int firstA, int n, int lda, int count,
                                          GM_ADDR productAddr, int productRows, int start, GM_ADDR infoAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (g_coreType == AscendC::AIV) return;
    using ATag = std::conditional_t<Upper, layout::RowMajor, layout::ColumnMajor>;
    using BTag = std::conditional_t<Upper, layout::ColumnMajor, layout::RowMajor>;
    using LCopy =
        Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, float, ATag, float, BTag, float, layout::RowMajor, void>;
    using LMmad = Gemm::Block::BlockMmadTla<Policy, L1Shape, L0Shape, float, float, float, void, LCopy>;
    Arch::Resource<Arch::Ascend950> resource;
    LMmad mm(resource);
    AscendC::GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    int rows = n - start, width = rows < PANEL ? rows : PANEL;
    int tiles = (rows + 127) / 128;
    for (int64_t task = AscendC::GetBlockIdx(); task < int64_t(count) * tiles; task += AscendC::GetBlockNum())
    {
        int q = task / tiles, firstRow = task % tiles * 128;
        if (info.GetValue(q)) continue;
        auto pa = MatrixPointer(aAddr, aArrayAddr, firstA + q);
        if (!pa) continue;
        AscendC::GlobalTensor<float> a, product;
        a.SetGlobalBuffer(pa);
        product.SetGlobalBuffer((__gm__ float *)productAddr + int64_t(q) * productRows * PANEL);
        int m = rows - firstRow < 128 ? rows - firstRow : 128;
        auto c = tla::MakeTensor(product[int64_t(firstRow) * PANEL],
                                 tla::MakeLayout(tla::MakeShape(m, width), tla::MakeStride(PANEL, tla::Int<1>{})),
                                 Arch::PositionGM{});
        if constexpr (Upper)
        {
            auto x = tla::MakeTensor(a[int64_t(start + firstRow) * lda],
                                     tla::MakeLayout(tla::MakeShape(m, start), tla::MakeStride(lda, tla::Int<1>{})),
                                     Arch::PositionGM{});
            auto y = tla::MakeTensor(a[int64_t(start) * lda],
                                     tla::MakeLayout(tla::MakeShape(start, width), tla::MakeStride(tla::Int<1>{}, lda)),
                                     Arch::PositionGM{});
            mm(x, y, c, GemmCoord(m, width, start));
        }
        else
        {
            auto x = tla::MakeTensor(a[start + firstRow],
                                     tla::MakeLayout(tla::MakeShape(m, start), tla::MakeStride(tla::Int<1>{}, lda)),
                                     Arch::PositionGM{});
            auto y = tla::MakeTensor(a[start],
                                     tla::MakeLayout(tla::MakeShape(start, width), tla::MakeStride(lda, tla::Int<1>{})),
                                     Arch::PositionGM{});
            mm(x, y, c, GemmCoord(m, width, start));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

void LeftGram(const MatrixBatch &a, float *product, int productRows, int start, int *info, aclrtStream s)
{
    if (a.upper)
        LeftGramKernel<true><<<CubeCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                          a.count, (GM_ADDR)product, productRows, start, (GM_ADDR)info);
    else
        LeftGramKernel<false><<<CubeCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                           a.count, (GM_ADDR)product, productRows, start,
                                                           (GM_ADDR)info);
}
template <bool ColumnMajor>
__global__ __aicore__ void TriUpdateKernel(GM_ADDR gAddr, GM_ADDR bAddr, GM_ADDR pAddr, int lda, int ldb,
                                           int productPitch, int n, int nrhs, int start, int width, int reverse,
                                           GM_ADDR infoAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (g_coreType == AscendC::AIV) return;
    AscendC::GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    if (info.GetValue(0)) return;
    using BTag = std::conditional_t<ColumnMajor, layout::ColumnMajor, layout::RowMajor>;
    using TCopy = Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, float, layout::RowMajor, float, BTag, float,
                                                layout::RowMajor, void>;
    using TMmad = Gemm::Block::BlockMmadTla<Policy, L1Shape, L0Shape, float, float, float, void, TCopy>;
    Arch::Resource<Arch::Ascend950> resource;
    TMmad mm(resource);
    AscendC::GlobalTensor<float> g, b, p;
    g.SetGlobalBuffer((__gm__ float *)gAddr);
    b.SetGlobalBuffer((__gm__ float *)bAddr);
    p.SetGlobalBuffer((__gm__ float *)pAddr);
    int rows = reverse ? start : n - start - width;
    int rowFirst = reverse ? 0 : start + width;
    int mt = (rows + 127) / 128, nt = (nrhs + 127) / 128;
    for (int task = AscendC::GetBlockIdx(); task < mt * nt; task += AscendC::GetBlockNum())
    {
        int i = task / nt * 128, r = task % nt * 128;
        int m = rows - i < 128 ? rows - i : 128, cols = nrhs - r < 128 ? nrhs - r : 128;
        // Transpose the product so it is contiguous in column-major RHS storage.
        auto ta = tla::MakeTensor(b[int64_t(r) * ldb + start],
                                  tla::MakeLayout(tla::MakeShape(cols, width), tla::MakeStride(ldb, tla::Int<1>{})),
                                  Arch::PositionGM{});
        auto tc = tla::MakeTensor(
            p[int64_t(r) * productPitch + i],
            tla::MakeLayout(tla::MakeShape(cols, m), tla::MakeStride(productPitch, tla::Int<1>{})), Arch::PositionGM{});
        if constexpr (ColumnMajor)
        {
            auto tb = tla::MakeTensor(g[int64_t(rowFirst + i) * lda + start],
                                      tla::MakeLayout(tla::MakeShape(width, m), tla::MakeStride(tla::Int<1>{}, lda)),
                                      Arch::PositionGM{});
            mm(ta, tb, tc, GemmCoord(cols, m, width));
        }
        else
        {
            auto tb = tla::MakeTensor(g[int64_t(start) * lda + rowFirst + i],
                                      tla::MakeLayout(tla::MakeShape(width, m), tla::MakeStride(lda, tla::Int<1>{})),
                                      Arch::PositionGM{});
            mm(ta, tb, tc, GemmCoord(cols, m, width));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

void RhsUpdate(const MatrixBatch &a, const MatrixBatch &b, int nrhs, float *product, int productPitch, int start,
               int width, bool reverse, int *info, unsigned cores, aclrtStream s)
{
    if (a.upper != int(reverse))
        TriUpdateKernel<true><<<cores, nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)b.single, (GM_ADDR)product, a.ld, b.ld,
                                                     productPitch, a.n, nrhs, start, width, int(reverse),
                                                     (GM_ADDR)info);
    else
        TriUpdateKernel<false><<<cores, nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)b.single, (GM_ADDR)product, a.ld,
                                                      b.ld, productPitch, a.n, nrhs, start, width, int(reverse),
                                                      (GM_ADDR)info);
}

void TriUpdate(float *g, float *b, float *product, int pitch, int n, int start, int width, bool reverse, int *info,
               unsigned cores, aclrtStream s)
{
    RhsUpdate({g, nullptr, n, pitch, 1, 0, 1}, {b, nullptr, n, pitch, 0, 0, 1}, n, product, pitch, start, width,
              reverse, info, cores, s);
}

void Gram(float *w, float *p, int pitch, int n, int start, int width, int count, int *info, unsigned cores,
          aclrtStream s)
{
    GramKernel<<<cores, nullptr, s>>>((GM_ADDR)w, (GM_ADDR)p, pitch, n, start, width, count, (GM_ADDR)info);
}
template <bool Upper>
__global__ __aicore__ void MediumGramKernel(GM_ADDR aAddr, GM_ADDR aArrayAddr, int firstA, int n, int lda, int count,
                                            GM_ADDR productAddr, int productPitch, GM_ADDR infoAddr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (g_coreType == AscendC::AIV) return;
    using ATag = std::conditional_t<Upper, layout::RowMajor, layout::ColumnMajor>;
    using BTag = std::conditional_t<Upper, layout::ColumnMajor, layout::RowMajor>;
    using GCopy =
        Gemm::Tile::PackedTileCopyTla<Arch::Ascend950, float, ATag, float, BTag, float, layout::RowMajor, void>;
    using GMmad = Gemm::Block::BlockMmadTla<Policy, L1Shape, L0Shape, float, float, float, void, GCopy>;
    Arch::Resource<Arch::Ascend950> resource;
    GMmad mm(resource);
    AscendC::GlobalTensor<int> info;
    info.SetGlobalBuffer((__gm__ int *)infoAddr);
    int width = n - PANEL;
    for (int q = AscendC::GetBlockIdx(); q < count; q += AscendC::GetBlockNum())
    {
        if (info.GetValue(q)) continue;
        auto pa = MatrixPointer(aAddr, aArrayAddr, firstA + q);
        if (!pa) continue;
        AscendC::GlobalTensor<float> a, product;
        a.SetGlobalBuffer(pa + (Upper ? int64_t(PANEL) * lda : PANEL));
        product.SetGlobalBuffer((__gm__ float *)productAddr + int64_t(q) * productPitch * productPitch);
        auto c = tla::MakeTensor(
            product, tla::MakeLayout(tla::MakeShape(width, width), tla::MakeStride(productPitch, tla::Int<1>{})),
            Arch::PositionGM{});
        if constexpr (Upper)
        {
            auto x =
                tla::MakeTensor(a, tla::MakeLayout(tla::MakeShape(width, PANEL), tla::MakeStride(lda, tla::Int<1>{})),
                                Arch::PositionGM{});
            auto y =
                tla::MakeTensor(a, tla::MakeLayout(tla::MakeShape(PANEL, width), tla::MakeStride(tla::Int<1>{}, lda)),
                                Arch::PositionGM{});
            mm(x, y, c, GemmCoord(width, width, PANEL));
        }
        else
        {
            auto x =
                tla::MakeTensor(a, tla::MakeLayout(tla::MakeShape(width, PANEL), tla::MakeStride(tla::Int<1>{}, lda)),
                                Arch::PositionGM{});
            auto y =
                tla::MakeTensor(a, tla::MakeLayout(tla::MakeShape(PANEL, width), tla::MakeStride(lda, tla::Int<1>{})),
                                Arch::PositionGM{});
            mm(x, y, c, GemmCoord(width, width, PANEL));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

void MediumGram(const MatrixBatch &a, float *product, int productPitch, int *info, aclrtStream s)
{
    if (a.upper)
        MediumGramKernel<true><<<CubeCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                            a.count, (GM_ADDR)product, productPitch, (GM_ADDR)info);
    else
        MediumGramKernel<false><<<CubeCores(), nullptr, s>>>((GM_ADDR)a.single, (GM_ADDR)a.array, a.first, a.n, a.ld,
                                                             a.count, (GM_ADDR)product, productPitch, (GM_ADDR)info);
}

}  // namespace Cholesky
