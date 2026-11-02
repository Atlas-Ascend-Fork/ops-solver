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
#if __NPU_ARCH__ == 3510
namespace ComplexBatchVector
{
namespace A = AscendC;
__simd_vf__ inline void PrepareIndices(int n, int group, int columns, int pitch, __ubuf__ uint32_t* io,
                                       __ubuf__ uint32_t* rows, __ubuf__ uint32_t* batchIndices)
{
    A::Reg::RegTensor<int32_t> sequence, g, row, offset, temporary;
    A::Reg::Arange(sequence, 0);
    for (int first = 0; first < columns * n * group; first += 64)
    {
        uint32_t count = columns * n * group - first;
        auto mask = A::Reg::UpdateMask<int32_t>(count);
        A::Reg::Adds(offset, sequence, first, mask);
        A::Reg::Duplicate(temporary, group);
        A::Reg::Div(row, offset, temporary, mask);
        A::Reg::Muls(temporary, row, group, mask);
        A::Reg::Sub(g, offset, temporary, mask);
        A::Reg::Muls(temporary, g, pitch, mask);
        A::Reg::Muls(offset, row, 2, mask);
        A::Reg::Add(offset, temporary, offset, mask);
        A::Reg::Muls(offset, offset, 4, mask);
        A::Reg::StoreAlign(reinterpret_cast<__ubuf__ int32_t*>(io + first), offset, mask);
        if (first < n * group)
        {
            uint32_t live = n * group - first;
            auto rowMask = A::Reg::UpdateMask<int32_t>(live);
            A::Reg::Muls(row, row, n * group, rowMask);
            A::Reg::Add(row, row, g, rowMask);
            A::Reg::StoreAlign(reinterpret_cast<__ubuf__ int32_t*>(rows + first), row, rowMask);
            A::Reg::StoreAlign(reinterpret_cast<__ubuf__ int32_t*>(batchIndices + first), g, rowMask);
        }
    }
}
template <int G>
__simd_vf__ inline void RestoreColumns(int n, int live, __ubuf__ float* real, __ubuf__ float* imag,
                                       __ubuf__ int* pivots)
{
    uint32_t count = G;
    auto mask = A::Reg::UpdateMask<float>(count);
    A::Reg::RegTensor<int32_t> signedLanes;
    A::Reg::Arange(signedLanes, 0);
    auto lanes = reinterpret_cast<A::Reg::RegTensor<uint32_t>&>(signedLanes);
    A::Reg::RegTensor<uint32_t> p, offset;
    A::Reg::RegTensor<float> ar, ai, br, bi;
    for (int k = n - 2; k >= 0; --k)
    {
        A::Reg::LoadAlign(p, reinterpret_cast<__ubuf__ uint32_t*>(pivots + k * G));
        A::Reg::Muls(offset, p, static_cast<uint32_t>(n * G), mask);
        A::Reg::Add(offset, offset, lanes, mask);
        A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
        for (int row = 0; row < n; ++row)
        {
            A::Reg::LoadAlign(ar, real + (k * n + row) * G);
            A::Reg::LoadAlign(ai, imag + (k * n + row) * G);
            A::Reg::Gather(br, real + row * G, offset, mask);
            A::Reg::Gather(bi, imag + row * G, offset, mask);
            A::Reg::Scatter(real + row * G, ar, offset, mask);
            A::Reg::Scatter(imag + row * G, ai, offset, mask);
            A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_STORE>();
            A::Reg::StoreAlign(real + (k * n + row) * G, br, mask);
            A::Reg::StoreAlign(imag + (k * n + row) * G, bi, mask);
        }
    }
}
template <int G>
__simd_callee__ inline void SelectPivot(int n, int k, __ubuf__ float* real, __ubuf__ float* imag,
                                        A::Reg::RegTensor<uint32_t>& selected, A::Reg::MaskReg& mask)
{
    A::Reg::RegTensor<float> best, r, i, ar, ai, score;
    A::Reg::RegTensor<uint32_t> candidate;
    A::Reg::MaskReg better;
    A::Reg::Duplicate(best, -1.0f);
    A::Reg::Duplicate(selected, static_cast<uint32_t>(k));
    A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
    for (int row = k; row < n; ++row)
    {
        A::Reg::LoadAlign(r, real + (k * n + row) * G);
        A::Reg::LoadAlign(i, imag + (k * n + row) * G);
        A::Reg::Abs(ar, r, mask);
        A::Reg::Abs(ai, i, mask);
        A::Reg::Add(score, ar, ai, mask);
        A::Reg::Compare<float, A::CMPMODE::GT>(better, score, best, mask);
        A::Reg::Select(best, score, best, better);
        A::Reg::Duplicate(candidate, static_cast<uint32_t>(row));
        A::Reg::Select(selected, candidate, selected, better);
    }
}
template <int G>
__simd_callee__ inline void SwapPivotRows(int n, int k, __ubuf__ float* real, __ubuf__ float* imag,
                                          __ubuf__ int* pivots, A::Reg::RegTensor<uint32_t>& selected,
                                          A::Reg::MaskReg& mask)
{
    A::Reg::RegTensor<uint32_t> lanes, offset;
    A::Reg::RegTensor<int32_t> signedLanes;
    A::Reg::Arange(signedLanes, 0);
    lanes = reinterpret_cast<A::Reg::RegTensor<uint32_t>&>(signedLanes);
    A::Reg::RegTensor<float> r, i, ar, ai;
    A::Reg::StoreAlign(reinterpret_cast<__ubuf__ uint32_t*>(pivots + k * G), selected, mask);
    A::Reg::Muls(offset, selected, static_cast<uint32_t>(G), mask);
    A::Reg::Add(offset, offset, lanes, mask);
    if (k + 1 == n) return;
    for (int col = 0; col < n; ++col)
    {
        int base = col * n * G;
        A::Reg::LoadAlign(ar, real + base + k * G);
        A::Reg::LoadAlign(ai, imag + base + k * G);
        A::Reg::Gather(r, real + base, offset, mask);
        A::Reg::Gather(i, imag + base, offset, mask);
        A::Reg::Scatter(real + base, ar, offset, mask);
        A::Reg::Scatter(imag + base, ai, offset, mask);
        A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_STORE>();
        A::Reg::StoreAlign(real + base + k * G, r, mask);
        A::Reg::StoreAlign(imag + base + k * G, i, mask);
    }
}
template <int G>
__simd_callee__ inline void NormalizePivot(int n, int k, __ubuf__ float* real, __ubuf__ float* imag,
                                           __ubuf__ float* divisors, __ubuf__ int* status, A::Reg::MaskReg& mask)
{
    A::Reg::RegTensor<float> r, i, ar, ai, score;
    A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
    A::Reg::LoadAlign(r, real + (k * n + k) * G);
    A::Reg::LoadAlign(i, imag + (k * n + k) * G);
    A::Reg::Abs(ar, r, mask);
    A::Reg::Abs(ai, i, mask);
    A::Reg::Max(score, ar, ai, mask);
    A::Reg::MaskReg zero, unset, bad;
    A::Reg::Compares<float, A::CMPMODE::EQ>(zero, score, 0.0f, mask);
    A::Reg::RegTensor<int> state, error;
    A::Reg::LoadAlign(state, status);
    A::Reg::Compares<int, A::CMPMODE::EQ>(unset, state, 0, mask);
    A::Reg::And(bad, zero, unset, mask);
    A::Reg::Duplicate(error, k + 1);
    A::Reg::Select(state, error, state, bad);
    A::Reg::StoreAlign(status, state, mask);
    A::Reg::Duplicate(ar, 1.0f);
    A::Reg::Duplicate(ai, 0.0f);
    A::Reg::Select(score, ar, score, zero);
    A::Reg::Select(r, ar, r, zero);
    A::Reg::Select(i, ai, i, zero);
    A::Reg::RegTensor<float> one, scaleInverse, normInverse;
    A::Reg::Duplicate(one, 1.0f);
    A::Reg::Div(scaleInverse, one, score, mask);
    A::Reg::Mul(r, r, scaleInverse, mask);
    A::Reg::Mul(i, i, scaleInverse, mask);
    A::Reg::Mul(ar, r, r, mask);
    A::Reg::MulAddDst(ar, i, i, mask);
    A::Reg::Div(normInverse, one, ar, mask);
    A::Reg::Mul(r, r, normInverse, mask);
    A::Reg::Mul(i, i, normInverse, mask);
    A::Reg::Mul(r, r, scaleInverse, mask);
    A::Reg::Mul(i, i, scaleInverse, mask);
    A::Reg::Muls(i, i, -1.0f, mask);
    A::Reg::StoreAlign(divisors, r, mask);
    A::Reg::StoreAlign(divisors + G, i, mask);
    A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
}
template <int G>
__simd_callee__ inline void PreparePivotVector(int n, int k, __ubuf__ float* real, __ubuf__ float* imag,
                                               __ubuf__ float* divisors, __ubuf__ int* pivots, __ubuf__ int* status)
{
    uint32_t count = G;
    auto mask = A::Reg::UpdateMask<float>(count);
    A::Reg::RegTensor<uint32_t> selected;
    SelectPivot<G>(n, k, real, imag, selected, mask);
    SwapPivotRows<G>(n, k, real, imag, pivots, selected, mask);
    NormalizePivot<G>(n, k, real, imag, divisors, status, mask);
}

template <int G>
__simd_callee__ inline void ClearPivotVectors(__ubuf__ float* real, __ubuf__ float* imag, __ubuf__ uint32_t* rowIndex,
                                              int n, int k)
{
    A::Reg::RegTensor<float> zero;
    A::Reg::RegTensor<uint32_t> rows;
    A::Reg::LocalMemBar<A::Reg::MemType::VEC_LOAD, A::Reg::MemType::VEC_STORE>();
    A::Reg::Duplicate(zero, 0.0f);
    for (int off = 0; off < n * G; off += 64)
    {
        uint32_t remaining = n * G - off;
        auto mask = A::Reg::UpdateMask<float>(remaining);
        A::Reg::LoadAlign(rows, rowIndex + off);
        A::Reg::Scatter(real + k * G, zero, rows, mask);
        A::Reg::Scatter(imag + k * G, zero, rows, mask);
        A::Reg::StoreAlign(real + k * n * G + off, zero, mask);
        A::Reg::StoreAlign(imag + k * n * G + off, zero, mask);
    }
}
template <int G>
__simd_callee__ inline void PrepareVectors(__ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* cr,
                                           __ubuf__ float* ci, __ubuf__ float* ur, __ubuf__ float* ui,
                                           __ubuf__ float* divisors, __ubuf__ uint32_t* rowIndex,
                                           __ubuf__ uint32_t* batchIndex, int n, int k)
{
    A::Reg::RegTensor<float> rr, ri, ar, ai, ir, ii, negative, zero;
    A::Reg::RegTensor<uint32_t> rows, batches;
    A::Reg::LoadAlign(batches, batchIndex);
    uint32_t full = 64;
    auto all = A::Reg::UpdateMask<float>(full);
    A::Reg::Gather(ir, divisors, batches, all);
    A::Reg::Gather(ii, divisors + G, batches, all);
    for (int off = 0; off < n * G; off += 64)
    {
        uint32_t remaining = n * G - off;
        auto mask = A::Reg::UpdateMask<float>(remaining);
        A::Reg::LoadAlign(rows, rowIndex + off);
        A::Reg::Gather(ar, real + k * G, rows, mask);
        A::Reg::Gather(ai, imag + k * G, rows, mask);
        A::Reg::Mul(rr, ar, ir, mask);
        A::Reg::Muls(negative, ai, -1.0f, mask);
        A::Reg::MulAddDst(rr, negative, ii, mask);
        A::Reg::Mul(ri, ar, ii, mask);
        A::Reg::MulAddDst(ri, ai, ir, mask);
        A::Reg::LoadAlign(ar, real + k * n * G + off);
        A::Reg::LoadAlign(ai, imag + k * n * G + off);
        A::Reg::StoreAlign(ur + off, rr, mask);
        A::Reg::StoreAlign(ui + off, ri, mask);
        A::Reg::Muls(ar, ar, -1.0f, mask);
        A::Reg::Muls(ai, ai, -1.0f, mask);
        A::Reg::StoreAlign(cr + off, ar, mask);
        A::Reg::StoreAlign(ci + off, ai, mask);
    }
    ClearPivotVectors<G>(real, imag, rowIndex, n, k);
    A::Reg::Duplicate(zero, 0.0f);
    uint32_t live = G;
    auto mask = A::Reg::UpdateMask<float>(live);
    A::Reg::StoreAlign(ur + k * G, ir, mask);
    A::Reg::StoreAlign(ui + k * G, ii, mask);
    A::Reg::Duplicate(ar, 1.0f);
    A::Reg::StoreAlign(cr + k * G, ar, mask);
    A::Reg::StoreAlign(ci + k * G, zero, mask);
}
template <int G>
__simd_callee__ inline void UpdateMatrixVector(__ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* cr,
                                               __ubuf__ float* ci, __ubuf__ float* ur, __ubuf__ float* ui,
                                               __ubuf__ uint32_t* batchIndex, __ubuf__ uint32_t* rowIndex,
                                               __ubuf__ float* divisors, int n, int k)
{
    PrepareVectors<G>(real, imag, cr, ci, ur, ui, divisors, rowIndex, batchIndex, n, k);
    A::Reg::LocalMemBar<A::Reg::MemType::VEC_STORE, A::Reg::MemType::VEC_LOAD>();
    A::Reg::RegTensor<float> r, i, vr, vi, wr, wi, negative;
    A::Reg::RegTensor<uint32_t> index;
    A::Reg::LoadAlign(index, batchIndex);
    uint32_t full = 64;
    auto fullMask = A::Reg::UpdateMask<float>(full);
    for (int col = 0; col < n; ++col)
    {
        A::Reg::Gather(wr, ur + col * G, index, fullMask);
        A::Reg::Gather(wi, ui + col * G, index, fullMask);
        uint16_t chunks = static_cast<uint16_t>((n * G + 63) / 64);
        for (uint16_t chunk = 0; chunk < chunks; ++chunk)
        {
            int off = chunk * 64;
            uint32_t remaining = n * G - off;
            auto mask = A::Reg::UpdateMask<float>(remaining);
            int pos = col * n * G + off;
            A::Reg::LoadAlign(r, real + pos);
            A::Reg::LoadAlign(i, imag + pos);
            A::Reg::LoadAlign(vr, cr + off);
            A::Reg::LoadAlign(vi, ci + off);
            A::Reg::MulAddDst(r, vr, wr, mask);
            A::Reg::Muls(negative, vi, -1.0f, mask);
            A::Reg::MulAddDst(r, negative, wi, mask);
            A::Reg::MulAddDst(i, vi, wr, mask);
            A::Reg::MulAddDst(i, vr, wi, mask);
            A::Reg::StoreAlign(real + pos, r, mask);
            A::Reg::StoreAlign(imag + pos, i, mask);
        }
    }
}
template <int G>
__simd_vf__ inline void ComputeMatrixVector(__ubuf__ float* real, __ubuf__ float* imag, __ubuf__ float* cr,
                                            __ubuf__ float* ci, __ubuf__ float* ur, __ubuf__ float* ui,
                                            __ubuf__ uint32_t* batchIndex, __ubuf__ uint32_t* rowIndex,
                                            __ubuf__ float* divisors, __ubuf__ int* pivots, __ubuf__ int* status, int n)
{
    for (int k = 0; k < n; ++k)
    {
        PreparePivotVector<G>(n, k, real, imag, divisors, pivots, status);
        UpdateMatrixVector<G>(real, imag, cr, ci, ur, ui, batchIndex, rowIndex, divisors, n, k);
    }
}

template <int G>
class Inverse
{
    __ubuf__ float* storage;
    A::LocalTensor<float> real, imag, rowReal, rowImag, colReal, colImag, stage, divisors;
    A::LocalTensor<uint32_t> ioIndex, rowIndex, columnIndex;
    A::LocalTensor<int> pivots, status;
    int n, plane, vectorSize, tileSize, stagePitch, columns;
    uint64_t input[G], output[G];
    __aicore__ inline __ubuf__ float* Pointer(A::LocalTensor<float> x)
    {
        return reinterpret_cast<__ubuf__ float*>(x.GetPhyAddr());
    }
    __aicore__ inline void Allocate()
    {
        uint32_t address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(storage));
        A::LocalTensor<float> matrix(A::TPosition::VECCALC, address, 2 * plane + 64);
        real = matrix;
        imag = matrix[plane + 64];
        int scratchSize = 4 * vectorSize + 6 * G + 192;
        if (stagePitch * G > scratchSize) scratchSize = stagePitch * G;
        A::LocalTensor<float> work(A::TPosition::VECCALC, address + (2 * plane + 64) * sizeof(float), scratchSize);
        rowReal = work;
        rowImag = work[vectorSize + 64];
        colReal = work[2 * vectorSize + 128];
        colImag = work[3 * vectorSize + 192];
        // DMA staging and compute vectors have disjoint lifetimes.
        stage = work;
        divisors = work[4 * vectorSize + 192];
        uint32_t indexAddress = address + (2 * plane + 64 + scratchSize) * sizeof(float);
        A::LocalTensor<uint32_t> idx(A::TPosition::VECCALC, indexAddress, tileSize + 2 * vectorSize);
        ioIndex = idx;
        rowIndex = idx[tileSize];
        columnIndex = idx[tileSize + vectorSize];
        A::LocalTensor<int> state(A::TPosition::VECCALC, indexAddress + (tileSize + 2 * vectorSize) * sizeof(uint32_t),
                                  (n + 1) * G);
        pivots = state;
        status = pivots[n * G];
        asc_vf_call<PrepareIndices>(n, G, columns, stagePitch,
                                    reinterpret_cast<__ubuf__ uint32_t*>(ioIndex.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint32_t*>(rowIndex.GetPhyAddr()),
                                    reinterpret_cast<__ubuf__ uint32_t*>(columnIndex.GetPhyAddr()));
    }
    __aicore__ inline bool Pointers(__gm__ uint8_t* a, __gm__ uint8_t* c, int base, int live)
    {
        bool continuous = true;
        auto* aa = reinterpret_cast<__gm__ uint64_t*>(a);
        auto* cc = reinterpret_cast<__gm__ uint64_t*>(c);
        uint64_t step = static_cast<uint64_t>(n) * n * 2 * sizeof(float);
        for (int g = 0; g < live; ++g)
        {
            input[g] = aa[base + g];
            output[g] = cc[base + g];
            continuous = continuous && input[g] == input[0] + g * step && output[g] == output[0] + g * step;
        }
        return continuous;
    }
    __aicore__ inline void Load(int live, bool continuous)
    {
        // Every matrix element is assigned by the gathers below.
        A::Duplicate(status, 0, G);
        A::DataCopyPadExtParams<float> padding{false, 0, 0, 0};
        for (int col = 0; col < n; col += columns)
        {
            int width = n - col < columns ? n - col : columns, count = width * n * G;
            // Inactive lanes need zeros; live lanes are overwritten by DMA.
            if (live < G) A::Duplicate(stage, 0.0f, stagePitch * G);
            A::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            A::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            if (continuous && width == columns)
            {
                A::GlobalTensor<float> gm;
                gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input[0]));
                A::DataCopyExtParams cp{static_cast<uint16_t>(live), static_cast<uint32_t>(width * n * 8),
                                        static_cast<uint32_t>((n - width) * n * 8),
                                        static_cast<uint32_t>((stagePitch - (width * n * 2 + 7) / 8 * 8) / 8), 0};
                A::DataCopyPad(stage, gm[col * n * 2], cp, padding);
            }
            else
            {
                for (int g = 0; g < live; ++g)
                {
                    A::GlobalTensor<float> gm;
                    gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(input[g]));
                    A::DataCopyExtParams cp{1, static_cast<uint32_t>(width * n * 8), 0, 0, 0};
                    A::DataCopyPad(stage[g * stagePitch], gm[col * n * 2], cp, padding);
                }
            }
            A::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            A::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            A::Gather(real[col * vectorSize], stage, ioIndex, 0, count);
            A::Gather(imag[col * vectorSize], stage, ioIndex, sizeof(float), count);
            A::PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void Compute(int live)
    {
        asc_vf_call<ComputeMatrixVector<G>>(Pointer(real), Pointer(imag), Pointer(colReal), Pointer(colImag),
                                            Pointer(rowReal), Pointer(rowImag),
                                            reinterpret_cast<__ubuf__ uint32_t*>(columnIndex.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ uint32_t*>(rowIndex.GetPhyAddr()),
                                            Pointer(divisors), reinterpret_cast<__ubuf__ int*>(pivots.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ int*>(status.GetPhyAddr()), n);
        A::PipeBarrier<PIPE_V>();
    }
    __aicore__ inline void StoreSeparateBatches(int live, int col, int width)
    {
        for (int g = 0; g < live; ++g)
        {
            if (status.GetValue(g) != 0) continue;
            A::GlobalTensor<float> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output[g]));
            A::DataCopyExtParams cp{1, static_cast<uint32_t>(width * n * 8), 0, 0, 0};
            A::DataCopyPad(gm[col * n * 2], stage[g * stagePitch], cp);
        }
    }

    __aicore__ inline void Store(int live, bool continuous)
    {
        A::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        A::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
        bool allOk = true;
        for (int g = 0; g < live; ++g) allOk = allOk && status.GetValue(g) == 0;
        for (int col = 0; col < n; col += columns)
        {
            int width = n - col < columns ? n - col : columns, count = width * n * G;
            A::Scatter(stage, real[col * vectorSize], ioIndex, 0, count);
            A::Scatter(stage, imag[col * vectorSize], ioIndex, sizeof(float), count);
            A::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            A::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            if (continuous && allOk && width == columns)
            {
                A::GlobalTensor<float> gm;
                gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output[0]));
                A::DataCopyExtParams cp{static_cast<uint16_t>(live), static_cast<uint32_t>(width * n * 8),
                                        static_cast<uint32_t>((stagePitch - (width * n * 2 + 7) / 8 * 8) / 8),
                                        static_cast<uint32_t>((n - width) * n * 8), 0};
                A::DataCopyPad(gm[col * n * 2], stage, cp);
            }
            else
            {
                StoreSeparateBatches(live, col, width);
            }
            A::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            A::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

   public:
    __aicore__ inline void Run(ComplexSolverShape shape, __gm__ uint8_t* a, __gm__ uint8_t* c, __gm__ int* info,
                               __ubuf__ float* memory)
    {
        storage = memory;
        n = shape.n;
        columns = 4;
        for (;;)
        {
            stagePitch = (columns * n * 2 + 7) / 8 * 8;
            if ((stagePitch / 8) % 2 == 0) stagePitch += 8;
            int scratchSize = 4 * n * G + 6 * G + 192;
            if (stagePitch * G > scratchSize) scratchSize = stagePitch * G;
            int words = (2 * n * n + (3 + columns) * n + 1) * G + 64 + scratchSize;
            if (words * sizeof(float) <= 244 * 1024 || columns == 1) break;
            columns /= 2;
        }
        plane = n * n * G;
        vectorSize = n * G;
        tileSize = columns * vectorSize;
        Allocate();
        for (int base = A::GetBlockIdx() * G; base < shape.batch; base += A::GetBlockNum() * G)
        {
            int live = shape.batch - base < G ? shape.batch - base : G;
            bool continuous = Pointers(a, c, base, live);
            Load(live, continuous);
            Compute(live);
            asc_vf_call<RestoreColumns<G>>(n, live, Pointer(real), Pointer(imag),
                                           reinterpret_cast<__ubuf__ int*>(pivots.GetPhyAddr()));
            A::SetFlag<A::HardEvent::V_S>(EVENT_ID0);
            A::WaitFlag<A::HardEvent::V_S>(EVENT_ID0);
            for (int g = 0; g < live; ++g) info[base + g] = status.GetValue(g);
            Store(live, continuous);
        }
    }
};
}  // namespace ComplexBatchVector
#endif
