/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#define ASCENDC_CUBE_ONLY
#include "complex_qr_matmul.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;
namespace
{
using MatrixType = MatmulType<TPosition::GM, CubeFormat::ND, float>;
constexpr MatmulConfig kAccumulatorConfig = GetNormalConfig(false, false, false, BatchMode::BATCH_LESS_THAN_L1, true,
                                                            IterateOrder::UNDEF, ScheduleType::INNER_PRODUCT, false);
template <bool TransposeB>
__aicore__ inline void CompensatedMatmulBody(__gm__ const float* ah, __gm__ const float* al, __gm__ const float* bh,
                                             __gm__ const float* bl, __gm__ float* c, uint32_t m, uint32_t n,
                                             uint32_t k, TCubeTiling tiling)
{
    TPipe pipe;
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, TransposeB>;
    Matmul<MatrixType, BType, MatrixType, MatrixType> mm;
    __gm__ uint8_t* workspace = nullptr;
    REGIST_MATMUL_OBJ(&pipe, workspace, mm, &tiling);
    uint32_t tileM = tiling.baseM, tileN = tiling.baseN;
    uint32_t columns = (n + tileN - 1) / tileN, tiles = ((m + tileM - 1) / tileM) * columns;
    for (uint32_t tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum())
    {
        uint32_t row = tile / columns * tileM, col = tile % columns * tileN;
        uint32_t rows = m - row < tileM ? m - row : tileM, cols = n - col < tileN ? n - col : tileN;
        if (rows == tileM && cols == tileN)
        {
            continue;
        }
        GlobalTensor<float> gah, gal, gbh, gbl, gc;
        gah.SetGlobalBuffer(const_cast<__gm__ float*>(ah) + uint64_t(row) * k);
        gal.SetGlobalBuffer(const_cast<__gm__ float*>(al) + uint64_t(row) * k);
        gbh.SetGlobalBuffer(const_cast<__gm__ float*>(bh) + (TransposeB ? uint64_t(col) * k : col));
        gbl.SetGlobalBuffer(const_cast<__gm__ float*>(bl) + (TransposeB ? uint64_t(col) * k : col));
        gc.SetGlobalBuffer(c + uint64_t(row) * n + col);
        mm.SetOrgShape(m, n, k, k, n);
        mm.SetSingleShape(rows, cols, k);
        mm.SetHF32(true, 0);
        // Accumulate small corrections before the leading product. Each output
        // tile has one owner and a fixed three-product order.
        mm.SetTensorA(gah, false);
        mm.SetTensorB(gbl, TransposeB);
        mm.IterateAll(gc);
        PipeBarrier<PIPE_ALL>();
        mm.SetTensorA(gal, false);
        mm.SetTensorB(gbh, TransposeB);
        mm.IterateAll(gc, 1);
        PipeBarrier<PIPE_ALL>();
        mm.SetTensorA(gah, false);
        mm.SetTensorB(gbh, TransposeB);
        mm.IterateAll(gc, 1);
        PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}

template <bool TransposeB>
__aicore__ inline void CompensatedAccumulatorBody(__gm__ const float* ah, __gm__ const float* al,
                                                  __gm__ const float* bh, __gm__ const float* bl, __gm__ float* c,
                                                  uint32_t m, uint32_t n, uint32_t k, TCubeTiling tiling,
                                                  uint32_t splitParts, uint32_t splitK)
{
    using Accumulator = MatmulType<TPosition::CO1, CubeFormat::NZ, float>;
    TPipe pipe;
    TBuf<TPosition::CO1> storage;
    pipe.InitBuffer(storage, 256 * 256 * sizeof(float));
    auto sum = storage.Get<float>();
    using BType = MatmulType<TPosition::GM, CubeFormat::ND, float, TransposeB>;
    Matmul<MatrixType, BType, Accumulator, MatrixType, kAccumulatorConfig> mm;
    __gm__ uint8_t* workspace = nullptr;
    REGIST_MATMUL_OBJ(&pipe, workspace, mm, &tiling);
    uint32_t tileM = tiling.baseM, tileN = tiling.baseN;
    uint32_t columns = n / tileN, tiles = (m / tileM) * columns;
    for (uint32_t job = GetBlockIdx(); job < tiles * splitParts; job += GetBlockNum())
    {
        uint32_t part = job / tiles, tile = job % tiles;
        uint32_t kStart = part * splitK, kCount = k - kStart < splitK ? k - kStart : splitK;
        uint32_t row = tile / columns * tileM, col = tile % columns * tileN;
        GlobalTensor<float> gah, gal, gbh, gbl, gc;
        gah.SetGlobalBuffer(const_cast<__gm__ float*>(ah) + uint64_t(row) * k + kStart);
        gal.SetGlobalBuffer(const_cast<__gm__ float*>(al) + uint64_t(row) * k + kStart);
        uint64_t bOffset = TransposeB ? uint64_t(col) * k + kStart : uint64_t(kStart) * n + col;
        gbh.SetGlobalBuffer(const_cast<__gm__ float*>(bh) + bOffset);
        gbl.SetGlobalBuffer(const_cast<__gm__ float*>(bl) + bOffset);
        gc.SetGlobalBuffer(c + uint64_t(part) * m * n + uint64_t(row) * n + col);
        mm.SetOrgShape(m, n, k, k, n);
        mm.SetSingleShape(tileM, tileN, kCount);
        mm.SetHF32(true, 0);
        mm.SetTensorA(gah, false);
        mm.SetTensorB(gbl, TransposeB);
        mm.Iterate(false, sum);
        mm.SetTensorA(gal, false);
        mm.SetTensorB(gbh, TransposeB);
        mm.Iterate(true, sum);
        mm.SetTensorA(gah, false);
        mm.SetTensorB(gbh, TransposeB);
        mm.Iterate(true, sum);
        PipeBarrier<PIPE_ALL>();
        FixpipeParamsV220 output;
        output.nSize = tileN;
        output.mSize = tileM;
        output.srcStride = tileM;
        output.dstStride = n;
        Fixpipe(gc, sum, output);
        PipeBarrier<PIPE_ALL>();
    }
    mm.End();
}

// Concrete launch entries allow the toolkit's precompile pass to identify both
// layouts as AIC-only. The implementation remains shared in the template bodies.
__global__ __aicore__ void CompensatedMatmulKernel(__gm__ const float* ah, __gm__ const float* al,
                                                   __gm__ const float* bh, __gm__ const float* bl, __gm__ float* c,
                                                   uint32_t m, uint32_t n, uint32_t k, TCubeTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CompensatedMatmulBody<false>(ah, al, bh, bl, c, m, n, k, tiling);
}

__global__ __aicore__ void CompensatedMatmulTransposeKernel(__gm__ const float* ah, __gm__ const float* al,
                                                            __gm__ const float* bh, __gm__ const float* bl,
                                                            __gm__ float* c, uint32_t m, uint32_t n, uint32_t k,
                                                            TCubeTiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CompensatedMatmulBody<true>(ah, al, bh, bl, c, m, n, k, tiling);
}

__global__ __aicore__ void CompensatedAccumulatorKernel(__gm__ const float* ah, __gm__ const float* al,
                                                        __gm__ const float* bh, __gm__ const float* bl, __gm__ float* c,
                                                        uint32_t m, uint32_t n, uint32_t k, TCubeTiling tiling,
                                                        uint32_t splitParts, uint32_t splitK)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CompensatedAccumulatorBody<false>(ah, al, bh, bl, c, m, n, k, tiling, splitParts, splitK);
}

__global__ __aicore__ void CompensatedAccumulatorTransposeKernel(__gm__ const float* ah, __gm__ const float* al,
                                                                 __gm__ const float* bh, __gm__ const float* bl,
                                                                 __gm__ float* c, uint32_t m, uint32_t n, uint32_t k,
                                                                 TCubeTiling tiling, uint32_t splitParts,
                                                                 uint32_t splitK)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    CompensatedAccumulatorBody<true>(ah, al, bh, bl, c, m, n, k, tiling, splitParts, splitK);
}
}  // namespace
void ComplexQrCompensatedMatmulLaunch(const float* ah, const float* al, const float* bh, const float* bl, float* c,
                                      uint32_t m, uint32_t n, uint32_t k, const void* data, bool interior,
                                      uint32_t splitParts, uint32_t splitK, void* stream, bool transposeB)
{
    auto tiling = static_cast<const TCubeTiling*>(data);
    uint32_t tileM = tiling->baseM, tileN = tiling->baseN;
    uint32_t count = interior ? (m / tileM) * (n / tileN) : ((m + tileM - 1) / tileM) * ((n + tileN - 1) / tileN);
    if (interior)
    {
        count *= splitParts;
    }
    uint32_t available = ComplexQrCubeCores();
    const uint32_t blocks = count < available ? count : available;
    if (interior && transposeB)
    {
        CompensatedAccumulatorTransposeKernel<<<blocks, nullptr, stream>>>(ah, al, bh, bl, c, m, n, k, *tiling,
                                                                           splitParts, splitK);
    }
    else if (interior)
    {
        CompensatedAccumulatorKernel<<<blocks, nullptr, stream>>>(ah, al, bh, bl, c, m, n, k, *tiling, splitParts,
                                                                  splitK);
    }
    else if (transposeB)
    {
        CompensatedMatmulTransposeKernel<<<blocks, nullptr, stream>>>(ah, al, bh, bl, c, m, n, k, *tiling);
    }
    else
    {
        CompensatedMatmulKernel<<<blocks, nullptr, stream>>>(ah, al, bh, bl, c, m, n, k, *tiling);
    }
}
