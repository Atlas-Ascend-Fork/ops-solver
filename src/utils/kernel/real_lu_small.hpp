/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "real_lu.hpp"
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
#include "real_lu_factor_vf.hpp"
#include "real_lu_fixed_vf.hpp"
#include "real_lu_refine_vf.hpp"
#endif

namespace RealLuKernel
{
// Element-major storage makes each vector lane an independent matrix.
class SmallInverse
{
   public:
    __aicore__ inline SmallInverse(TPipe &pipe, int order, int groupLanes, bool factor, bool paddedColumns = false);
    __aicore__ inline int Lanes() const;
    __aicore__ inline LocalTensor<float> L(int row, int col);
    __aicore__ inline LocalTensor<float> X(int row, int col);
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR pivot, GM_ADDR info, int first, int count,
                               bool factor, bool identityPivots, bool arrays = false, int lda = 0, int ldc = 0);

   private:
    __aicore__ inline void InitializeLayout(int order, int groupLanes, bool factor, bool paddedColumns);
    __aicore__ inline void InitializeBuffers(TPipe &pipe, bool factor);
    __aicore__ inline void BindTensors();
    __aicore__ inline void InitializeIndices();
    __aicore__ inline void InitializeBroadcast(bool factor);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void LoadSeparate(GM_ADDR input, int first, int count, bool arrays, int lda);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void StoreSeparate(GM_ADDR output, int first, int count, bool arrays, int ldc);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void CompactLoadTile(GM_ADDR input, int first, int count, bool arrays, int lda);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void CompactStoreTile(GM_ADDR output, int first, int count, bool separateMatrices, bool arrays,
                                            int ldc);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void LoadCompact(GM_ADDR input, int first, int count, bool arrays, int lda);
#endif
    __aicore__ inline void LoadRegular(GM_ADDR input, GlobalTensor<float> a, int first, int count, bool arrays,
                                       int lda);
    __aicore__ inline void LoadGroup(GM_ADDR input, GlobalTensor<float> a, int first, int count, bool arrays, int lda);
    __aicore__ inline void PreparePivots(GM_ADDR pivot, int first, int count, bool factor);
    __aicore__ inline void ClassifySingular();
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void SaveRefinementInput();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void InitialRefinement();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline bool RefinementNeedsCorrection();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    template <bool fusedResidual = false>
    __aicore__ inline void ScoreRefinement();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    template <bool fusedResidual = false>
    __aicore__ inline void DampedRefinement(float step);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void RefineInverse();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void PackInverse(int offset, int count, bool identityOutput);
#endif
    __aicore__ inline void StoreRegular(GM_ADDR output, GlobalTensor<float> c, int first, int count,
                                        bool separateMatrices, bool arrays, int ldc);
    __aicore__ inline void StoreGroup(GM_ADDR output, GlobalTensor<float> c, int first, int count, bool arrays, int ldc,
                                      bool separateMatrices, bool identityPivots, bool refine);
    __aicore__ inline void ComputeDiagonalReciprocals(LocalTensor<float> diagonalValues);
    __aicore__ inline void DiagonalReciprocals();
    __aicore__ inline void ReciprocalPair();
    __aicore__ inline void InverseTwo();
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
    __aicore__ inline void GenericUpperInverse();
#endif
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
    __aicore__ inline void GenericLowerInverse();
#endif
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
    __aicore__ inline void GenericInversePermutation();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void InplaceLowerInverse();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline bool InplaceUpperLow();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline bool InplaceUpperHigh();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline bool SeparateInverseLow();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline bool SeparateInverseMiddle();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline bool SeparateInverseHigh();
#endif
    __aicore__ inline void LapackInverse(bool identityPivots);
    __aicore__ inline void UpdateRow(LocalTensor<float> target, LocalTensor<float> source,
                                     LocalTensor<float> coefficient, int count);
    __aicore__ inline void SwapRows(LocalTensor<float> matrix, int k);
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
    __aicore__ inline void ChoosePivotGeneric(int k);
#endif
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
    __aicore__ inline void FactorGeneric();
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void ApplyLeftColumn(uint16_t k);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void ChoosePivotGroup(uint16_t k, bool leftLooking, bool rangeDivision,
                                            __ubuf__ float *groupFlags);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void SwapPivotGroup(uint16_t k, bool leftLooking);
#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    __aicore__ inline void EliminateColumnGroup(uint16_t k, bool leftLooking, bool rangeDivision, bool normalColumn,
                                                bool boundedColumn);
#endif
    __aicore__ inline void Factor();
    bool compact;
    int n, lanes, rowStride, packedStride, packedColumnStride, maskBytes;
    TBuf<TPosition::VECCALC> rawBuf, luBuf, invBuf, inIndexBuf, outIndexBuf, pivIndexBuf;
    TBuf<TPosition::VECCALC> pivBuf, tmpBuf, maskBuf, infoBuf, broadcastBuf;
    LocalTensor<float> raw, lu, inv, tmp, piv;
    LocalTensor<uint32_t> inIndex, outIndex, pivIndex, broadcastIndex;
    LocalTensor<uint8_t> mask;
    LocalTensor<int> infos;
};
}  // namespace RealLuKernel
#include "real_lu_small_dispatch.hpp"
#include "real_lu_small_inverse.hpp"

namespace RealLuKernel
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::LoadSeparate(GM_ADDR input, int first, int count, bool arrays, int lda)
{
    for (int lane = 0; lane < count; ++lane)
    {
        int index = first + lane;
        uint64_t address =
            arrays ? ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t *>(input) + index)
                   : reinterpret_cast<uint64_t>(input) + static_cast<uint64_t>(index) * lda * n * sizeof(float);
        GlobalTensor<float> matrix;
        matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
        uint64_t sourceGap = static_cast<uint64_t>(lda - n) * sizeof(float);
        if (sourceGap > 0xffffffffULL)
        {
            for (int col = 0; col < n; ++col)
            {
                Load(raw[lane * packedStride + col * packedColumnStride], matrix[static_cast<int64_t>(col) * lda], n);
            }
            continue;
        }
        DataCopyExtParams copy{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                               static_cast<uint32_t>(sourceGap),
                               static_cast<uint32_t>((packedColumnStride - Align(n)) / 8), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(Align(n) - n), 0.0f};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(raw[lane * packedStride], matrix, copy, pad);
        Fence<HardEvent::MTE2_V>();
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::StoreSeparate(GM_ADDR output, int first, int count, bool arrays, int ldc)
{
    for (int lane = 0; lane < count; ++lane)
    {
        int index = first + lane;
        uint64_t address =
            arrays ? ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t *>(output) + index)
                   : reinterpret_cast<uint64_t>(output) + static_cast<uint64_t>(index) * ldc * n * sizeof(float);
        GlobalTensor<float> matrix;
        matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
        uint64_t destinationGap = static_cast<uint64_t>(ldc - n) * sizeof(float);
        if (destinationGap > 0xffffffffULL)
        {
            for (int col = 0; col < n; ++col)
            {
                Store(matrix[static_cast<int64_t>(col) * ldc], raw[lane * packedStride + col * packedColumnStride], n);
            }
            continue;
        }
        DataCopyExtParams copy{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                               static_cast<uint32_t>((packedColumnStride - Align(n)) / 8),
                               static_cast<uint32_t>(destinationGap), 0};
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(matrix, raw[lane * packedStride], copy);
        Fence<HardEvent::MTE3_S>();
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::CompactLoadTile(GM_ADDR input, int first, int count, bool arrays, int lda)
{
    if (arrays || lda != 0)
    {
        LoadSeparate(input, first, count, arrays, lda);
    }
    else
    {
        GlobalTensor<float> matrices;
        matrices.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        DataCopyExtParams copy{static_cast<uint16_t>(count), static_cast<uint32_t>(n * n * sizeof(float)), 0,
                               static_cast<uint32_t>((packedStride - Align(n * n)) / 8), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(Align(n * n) - n * n), 0.0f};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(raw, matrices[static_cast<int64_t>(first) * n * n], copy, pad);
        Fence<HardEvent::MTE2_V>();
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::CompactStoreTile(GM_ADDR output, int first, int count, bool separateMatrices,
                                                      bool arrays, int ldc)
{
    if (separateMatrices)
    {
        StoreSeparate(output, first, count, arrays, ldc);
    }
    else
    {
        GlobalTensor<float> matrices;
        matrices.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        DataCopyExtParams copy{static_cast<uint16_t>(count), static_cast<uint32_t>(n * n * sizeof(float)),
                               static_cast<uint32_t>((packedStride - Align(n * n)) / 8), 0, 0};
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(matrices[static_cast<int64_t>(first) * n * n], raw, copy);
        Fence<HardEvent::MTE3_S>();
    }
}

#endif
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::LoadCompact(GM_ADDR input, int first, int count, bool arrays, int lda)
{
    Duplicate(lu, 0.0f, n * n * lanes);
    PipeBarrier<PIPE_V>();
    for (int tile = 0; tile < count; tile += 8)
    {
        int tileCount = count - tile < 8 ? count - tile : 8;
        CompactLoadTile(input, first + tile, tileCount, arrays, lda);
        Fence<HardEvent::S_V>();
        SmallPackedToSoAVF(reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr()) + tile,
                           reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()), n, lanes, packedStride,
                           packedColumnStride, tileCount);
    }
    if (count < lanes)
    {
        SmallCompleteDiagonalVF(reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr()), n, lanes, count);
    }
}

#endif
__aicore__ inline void SmallInverse::LoadRegular(GM_ADDR input, GlobalTensor<float> a, int first, int count,
                                                 bool arrays, int lda)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (arrays || lda != 0)
    {
        LoadSeparate(input, first, count, arrays, lda);
    }
    else
#endif
        if (packedStride == n * n)
    {
        Load(raw, a[static_cast<int64_t>(first) * n * n], count * n * n);
    }
    else
    {
        DataCopyExtParams copy{static_cast<uint16_t>(count), static_cast<uint32_t>(n * n * sizeof(float)), 0,
                               static_cast<uint32_t>((packedStride - Align(n * n)) / 8), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(Align(n * n) - n * n), 0.0f};
        Fence<HardEvent::V_MTE2>();
        DataCopyPad(raw, a[static_cast<int64_t>(first) * n * n], copy, pad);
        Fence<HardEvent::MTE2_V>();
    }
    Fence<HardEvent::S_V>();
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    Duplicate(lu, 0.0f, n * n * lanes);
    PipeBarrier<PIPE_V>();
    SmallPackedToSoAVF(reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr()),
                       reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr()), n, lanes, packedStride, packedColumnStride,
                       count);
    if (count < lanes)
    {
        SmallCompleteDiagonalVF(reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr()), n, lanes, count);
    }
#else
    Gather(lu, raw, inIndex, 0, n * n * lanes);
#endif
}

__aicore__ inline void SmallInverse::LoadGroup(GM_ADDR input, GlobalTensor<float> a, int first, int count, bool arrays,
                                               int lda)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (compact)
    {
        LoadCompact(input, first, count, arrays, lda);
        return;
    }
#endif
    LoadRegular(input, a, first, count, arrays, lda);
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SmallInverse::PackInverse(int offset, int count, bool identityOutput)
{
    auto packed = reinterpret_cast<__ubuf__ float *>(raw.GetPhyAddr());
    auto inverse = reinterpret_cast<__ubuf__ float *>(inv.GetPhyAddr()) + offset;
    auto permutation = reinterpret_cast<__ubuf__ uint32_t *>(pivIndex.GetPhyAddr()) + offset;
    if (identityOutput)
    {
        SmallSoAToPackedVF<true>(packed, inverse, permutation, n, lanes, packedStride, packedColumnStride, count);
    }
    else
    {
        SmallSoAToPackedVF<false>(packed, inverse, permutation, n, lanes, packedStride, packedColumnStride, count);
    }
}

#endif
__aicore__ inline void SmallInverse::StoreRegular(GM_ADDR output, GlobalTensor<float> c, int first, int count,
                                                  bool separateMatrices, bool arrays, int ldc)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (separateMatrices)
    {
        StoreSeparate(output, first, count, arrays, ldc);
    }
    else
#endif
        if (packedStride == n * n)
    {
        Store(c[static_cast<int64_t>(first) * n * n], raw, count * n * n);
    }
    else
    {
        DataCopyExtParams copy{static_cast<uint16_t>(count), static_cast<uint32_t>(n * n * sizeof(float)),
                               static_cast<uint32_t>((packedStride - Align(n * n)) / 8), 0, 0};
        Fence<HardEvent::V_MTE3>();
        DataCopyPad(c[static_cast<int64_t>(first) * n * n], raw, copy);
        Fence<HardEvent::MTE3_S>();
    }
}

__aicore__ inline void SmallInverse::StoreGroup(GM_ADDR output, GlobalTensor<float> c, int first, int count,
                                                bool arrays, int ldc, bool separateMatrices, bool identityPivots,
                                                bool refine)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    bool identityOutput = n == 2 || identityPivots || refine;
    if (!identityOutput)
    {
        SmallColumnPermutationVF(reinterpret_cast<__ubuf__ uint32_t *>(pivIndex.GetPhyAddr()),
                                 reinterpret_cast<__ubuf__ float *>(piv.GetPhyAddr()), n, lanes, count);
    }
    if (compact)
    {
        for (int tile = 0; tile < count; tile += 8)
        {
            int tileCount = count - tile < 8 ? count - tile : 8;
            PackInverse(tile, tileCount, identityOutput);
            Fence<HardEvent::V_S>();
            CompactStoreTile(output, first + tile, tileCount, separateMatrices, arrays, ldc);
        }
        return;
    }
    PackInverse(0, count, identityOutput);
#else
    Gather(raw, inv, outIndex, 0, n * n * lanes);
#endif
    Fence<HardEvent::V_S>();
    StoreRegular(output, c, first, count, separateMatrices, arrays, ldc);
}

__aicore__ inline SmallInverse::SmallInverse(TPipe &pipe, int order, int groupLanes, bool factor, bool paddedColumns)
{
    InitializeLayout(order, groupLanes, factor, paddedColumns);
    InitializeBuffers(pipe, factor);
    BindTensors();
    InitializeIndices();
    InitializeBroadcast(factor);
}

__aicore__ inline void SmallInverse::InitializeLayout(int order, int groupLanes, bool factor, bool paddedColumns)
{
    n = order;
    compact = false;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    // Contiguous inverse-from-LU batches can reuse the in-place n=32
    // storage; generic padded routes retain separate factors and output.
    compact = (factor && n >= 17 && n <= 32) || (!factor && n == 32 && !paddedColumns);
#endif
    lanes = groupLanes;
    rowStride = n * lanes;
    packedColumnStride = paddedColumns ? Align(n) : n;
    packedStride = n * packedColumnStride;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    // Use an odd number of UB blocks between matrices to distribute the
    // addresses used by per-element Gather/Scatter transposes.
    if (n != 2 || paddedColumns)
    {
        packedStride = (Align(n * packedColumnStride) / 8 | 1) * 8;
    }
#endif
}

__aicore__ inline void SmallInverse::InitializeBuffers(TPipe &pipe, bool factor)
{
    int elements = n * n * lanes;
    pipe.InitBuffer(rawBuf, packedStride * (compact ? 8 : lanes) * sizeof(float));
    pipe.InitBuffer(luBuf, elements * sizeof(float));
    pipe.InitBuffer(invBuf, compact ? 32 : elements * sizeof(float));
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    pipe.InitBuffer(inIndexBuf, 32);
    pipe.InitBuffer(outIndexBuf, 32);
#else
    pipe.InitBuffer(inIndexBuf, elements * sizeof(uint32_t));
    pipe.InitBuffer(outIndexBuf, elements * sizeof(uint32_t));
#endif
    pipe.InitBuffer(pivIndexBuf, n * lanes * sizeof(uint32_t));
    bool factorBroadcast = factor;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    factorBroadcast = false;
#endif
    pipe.InitBuffer(broadcastBuf, factorBroadcast ? rowStride * sizeof(uint32_t) : 32);
    pipe.InitBuffer(pivBuf, n * lanes * sizeof(float));
    pipe.InitBuffer(tmpBuf, (2 * rowStride + 3 * lanes) * sizeof(float));
    maskBytes = ((rowStride + 255) / 256) * 32;
    pipe.InitBuffer(maskBuf, 2 * maskBytes);
    pipe.InitBuffer(infoBuf, lanes * sizeof(int));
}

__aicore__ inline void SmallInverse::BindTensors()
{
    raw = rawBuf.Get<float>();
    lu = luBuf.Get<float>();
    inv = compact ? lu : invBuf.Get<float>();
    tmp = tmpBuf.Get<float>();
    piv = pivBuf.Get<float>();
    inIndex = inIndexBuf.Get<uint32_t>();
    outIndex = outIndexBuf.Get<uint32_t>();
    pivIndex = pivIndexBuf.Get<uint32_t>();
    mask = maskBuf.Get<uint8_t>();
    broadcastIndex = broadcastBuf.Get<uint32_t>();
    infos = infoBuf.Get<int>();
}

__aicore__ inline void SmallInverse::InitializeIndices()
{
#if !((__NPU_ARCH__ == 3510) && defined(__DAV_VEC__))
    for (int entry = 0; entry < n * n; ++entry)
    {
        for (int lane = 0; lane < lanes; ++lane)
        {
            inIndex.SetValue(entry * lanes + lane, (lane * n * n + (entry % n) * n + entry / n) * sizeof(float));
            outIndex.SetValue(lane * n * n + entry, (((entry % n) * n + entry / n) * lanes + lane) * sizeof(float));
        }
    }
    for (int k = 0; k < n; ++k)
    {
        for (int lane = 0; lane < lanes; ++lane)
        {
            pivIndex.SetValue(k * lanes + lane, (lane * n + k) * sizeof(int));
        }
    }
#endif
}

__aicore__ inline void SmallInverse::InitializeBroadcast(bool factor)
{
    bool factorBroadcast = factor;
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    factorBroadcast = false;
#endif
    if (factorBroadcast)
    {
        CreateVecIndex(broadcastIndex.ReinterpretCast<int>(), 0, rowStride);
        Duplicate(tmp.ReinterpretCast<int>(), lanes - 1, rowStride);
        PipeBarrier<PIPE_V>();
        And(broadcastIndex.ReinterpretCast<int>(), broadcastIndex.ReinterpretCast<int>(), tmp.ReinterpretCast<int>(),
            rowStride);
        PipeBarrier<PIPE_V>();
        ShiftLeft(broadcastIndex, broadcastIndex, 2u, rowStride);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline int SmallInverse::Lanes() const { return lanes; }

__aicore__ inline LocalTensor<float> SmallInverse::L(int row, int col) { return lu[(row * n + col) * lanes]; }

__aicore__ inline LocalTensor<float> SmallInverse::X(int row, int col) { return inv[(row * n + col) * lanes]; }
}  // namespace RealLuKernel
