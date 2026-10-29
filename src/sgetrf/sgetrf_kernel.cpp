/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>

#include <algorithm>

#include "../utils/kernel/real_lu.hpp"
using namespace RealLuKernel;
extern "C" int real_lu_vector_cores();

namespace
{
__aicore__ inline void PrepareFiniteMask(LocalTensor<float> scratch, LocalTensor<uint8_t> mask, int lda, int m,
                                         int chunk)
{
    if (lda == 0)
    {
        return;
    }
    if (lda < 0 || lda == m)
    {
        return;
    }
    for (int i = 0; i < chunk; ++i)
    {
        scratch.SetValue(i, i % lda < m ? 1.0f : 0.0f);
    }
    Fence<HardEvent::S_V>();
    CompareScalar(mask, scratch, 0.5f, CMPMODE::GT, chunk);
    PipeBarrier<PIPE_V>();
}
__aicore__ inline int LoadFiniteColumn(LocalTensor<float> raw, GlobalTensor<uint64_t> pointers, GM_ADDR input,
                                       bool arrays, int64_t job, int m, int n, int lda)
{
    if (n == 0)
    {
        return 0;
    }
    if (n < 0)
    {
        return 0;
    }
    int index = job / n, column = job % n;
    GlobalTensor<float> matrix;
    uint64_t address = arrays
                           ? pointers.GetValue(index)
                           : reinterpret_cast<uint64_t>(input) + static_cast<uint64_t>(index) * lda * n * sizeof(float);
    matrix.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
    Load(raw, matrix[static_cast<int64_t>(column) * lda], m);
    return m;
}
__aicore__ inline bool NonFiniteChunk(LocalTensor<float> raw, LocalTensor<float> diff, LocalTensor<float> scratch,
                                      LocalTensor<float> sum, LocalTensor<uint8_t> mask, int count, bool padding)
{
    Fence<HardEvent::S_V>();
    Sub(diff, raw, raw, count);
    PipeBarrier<PIPE_V>();
    if (padding)
    {
        Duplicate(raw, 0.0f, count);
        PipeBarrier<PIPE_V>();
        Select(diff, mask, diff, raw, SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
        PipeBarrier<PIPE_V>();
    }
    ReduceSum(sum, diff, scratch, count);
    Fence<HardEvent::V_S>();
    float value = sum.GetValue(0);
    return value != value;
}
__aicore__ inline int LoadFiniteChunk(LocalTensor<float> raw, GlobalTensor<float> a, GlobalTensor<uint64_t> pointers,
                                      GM_ADDR input, bool arrays, bool linear, int64_t job, int64_t total, int chunk,
                                      int m, int n, int lda)
{
    if (linear)
    {
        int64_t first = job * chunk;
        int count = total - first < chunk ? total - first : chunk;
        Load(raw, a[first], count);
        return count;
    }
    return LoadFiniteColumn(raw, pointers, input, arrays, job, m, n, lda);
}
__aicore__ inline void StoreFiniteResult(GM_ADDR flags, LocalTensor<int> scratch, int bad)
{
    GlobalTensor<int> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(flags));
    StoreInt(result[GetBlockIdx()], scratch, bad);
}
}  // namespace

__global__ __aicore__ void real_lu_finite_kernel(GM_ADDR input, GM_ADDR flags, int m, int n, int lda, int batch,
                                                 bool arrays)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (lda == 0)
    {
        return;
    }
    if (m <= 0 || n <= 0 || lda < 0 || batch <= 0)
    {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuf, diffBuf, scratchBuf, sumBuf, maskBuf;
    pipe.InitBuffer(rawBuf, 8192 * sizeof(float));
    pipe.InitBuffer(diffBuf, 8192 * sizeof(float));
    pipe.InitBuffer(scratchBuf, 8192 * sizeof(float));
    pipe.InitBuffer(sumBuf, 32);
    pipe.InitBuffer(maskBuf, 1024);
    auto raw = rawBuf.Get<float>(), diff = diffBuf.Get<float>(), scratch = scratchBuf.Get<float>();
    auto sum = sumBuf.Get<float>();
    auto mask = maskBuf.Get<uint8_t>();
    GlobalTensor<float> a;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
    GlobalTensor<uint64_t> pointers;
    pointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input));
    bool linear = !arrays && lda <= 8192;
    int chunk = linear ? 8192 / lda * lda : m;
    if (chunk <= 0)
    {
        return;
    }
    if (linear)
    {
        PrepareFiniteMask(scratch, mask, lda, m, chunk);
    }
    int64_t total = static_cast<int64_t>(batch) * lda * n;
    int64_t jobs = linear ? (total + chunk - 1) / chunk : static_cast<int64_t>(batch) * n;
    int bad = 0;
    for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum())
    {
        int count = LoadFiniteChunk(raw, a, pointers, input, arrays, linear, job, total, chunk, m, n, lda);
        if (NonFiniteChunk(raw, diff, scratch, sum, mask, count, linear && lda != m))
        {
            bad = 1;
            break;
        }
    }
    StoreFiniteResult(flags, sumBuf.Get<int>(), bad);
}
extern "C" void real_lu_finite_do(GM_ADDR a, GM_ADDR flags, int m, int n, int lda, int batch, bool arrays, int blocks,
                                  aclrtStream stream)
{
    real_lu_finite_kernel<<<blocks, nullptr, stream>>>(a, flags, m, n, lda, batch, arrays);
}

__global__ __aicore__ void real_lu_fill_info_kernel(GM_ADDR info, int count, int value)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 4096);
    auto local = buffer.Get<int>();
    GlobalTensor<int> output;
    output.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    Duplicate(local, value, 1024);
    Fence<HardEvent::V_MTE3>();
    for (int64_t begin = static_cast<int64_t>(GetBlockIdx()) * 1024; begin < count;
         begin += static_cast<int64_t>(GetBlockNum()) * 1024)
    {
        int size = count - begin < 1024 ? count - begin : 1024;
        DataCopyExtParams copy{1, static_cast<uint32_t>(size * sizeof(int)), 0, 0, 0};
        DataCopyPad(output[begin], local, copy);
    }
    Fence<HardEvent::MTE3_S>();
}
namespace
{
__aicore__ inline void ConfigureFactorLayout(int m, int n, int steps, int &stride, bool &localMatrix)
{
#if defined(REAL_LU_REGISTER_LAYOUT_950)
    // The full local matrix needs one scratch column. The unused GM-path
    // second column is a 32-byte stub; account for it and the info buffer.
    // A failed VF-capacity check keeps the original local/GM fallback.
    constexpr int FACTOR_950_LOCAL_BYTES = 237568;
    int vectorStride = (m + 63) / 64 * 64;
    int64_t vectorLocalBytes = (static_cast<int64_t>(vectorStride) * (n + 1) + Align(steps) + 16) * sizeof(float);
    if (vectorLocalBytes <= FACTOR_950_LOCAL_BYTES)
    {
        stride = vectorStride;
        localMatrix = true;
    }
#else
    (void)m;
    (void)n;
    (void)steps;
    (void)stride;
    (void)localMatrix;
#endif
}
__aicore__ inline void UpdateGlobalFactorColumns(GlobalTensor<float> a, LocalTensor<float> col,
                                                 LocalTensor<float> other, int m, int n, int lda, int k, int p,
                                                 float diagonal)
{
    for (int j = 0; j < n; ++j)
    {
        if (j == k || (j < k && p == k))
        {
            continue;
        }
        Load(other, a[static_cast<int64_t>(j) * lda], m);
        Swap(other, k, p);
        if (j > k && diagonal != 0 && k + 1 < m)
        {
            UpdateColumn(other, col, k, m, other.GetValue(k));
        }
        Store(a[static_cast<int64_t>(j) * lda], other, m);
    }
}
__aicore__ inline int FactorGlobal(GlobalTensor<float> a, LocalTensor<float> col, LocalTensor<float> other,
                                   LocalTensor<float> reduction, LocalTensor<int> piv, int m, int n, int lda, int steps,
                                   bool pivoting)
{
    int state = 0;
    for (int k = 0; k < steps; ++k)
    {
        Load(col, a[static_cast<int64_t>(k) * lda], m);
        int p = PivotRF(col, other, reduction, k, m, pivoting);
        piv.SetValue(k, p + 1);
        Swap(col, k, p);
        float diagonal = col.GetValue(k);
        if (diagonal == 0 && state == 0)
        {
            state = k + 1;
        }
        if (diagonal != 0 && k + 1 < m)
        {
            ScaleColumnRF(col, other, reduction, k, m, diagonal);
        }
        Store(a[static_cast<int64_t>(k) * lda], col, m);
        UpdateGlobalFactorColumns(a, col, other, m, n, lda, k, p, diagonal);
    }
    return state;
}
__aicore__ inline int FactorLoadedMatrix(GlobalTensor<float> a, LocalTensor<float> mat, LocalTensor<float> col,
                                         LocalTensor<int> piv, LocalTensor<float> reduction, int m, int n, int lda,
                                         int stride, bool pivoting)
{
    for (int j = 0; j < n; ++j)
    {
        Load(mat[j * stride], a[static_cast<int64_t>(j) * lda], m);
    }
    int state = FactorLocalRF(mat, col, m, n, stride, piv, pivoting, reduction);
    for (int j = 0; j < n; ++j)
    {
        Store(a[static_cast<int64_t>(j) * lda], mat[j * stride], m);
    }
    return state;
}
}  // namespace

__global__ __aicore__ void real_lu_factor_kernel(GM_ADDR matrix, int m, int n, int lda, GM_ADDR pivot, GM_ADDR info)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    int stride = Align(m), steps = m < n ? m : n;
    bool localMatrix = static_cast<int64_t>(stride) * n <= 32768;
    ConfigureFactorLayout(m, n, steps, stride, localMatrix);
    TBuf<TPosition::VECCALC> matrixBuf, colBuf, otherBuf, pivBuf, infoBuf;
    pipe.InitBuffer(matrixBuf, localMatrix ? stride * n * sizeof(float) : 32);
    pipe.InitBuffer(colBuf, stride * sizeof(float));
    pipe.InitBuffer(otherBuf, localMatrix ? 32 : stride * sizeof(float));
    pipe.InitBuffer(pivBuf, Align(steps) * sizeof(int));
    pipe.InitBuffer(infoBuf, 32);
    auto mat = matrixBuf.Get<float>();
    auto col = colBuf.Get<float>();
    auto other = otherBuf.Get<float>();
    auto piv = pivBuf.Get<int>();
    auto tmp = infoBuf.Get<int>();
    GlobalTensor<float> a;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matrix));
    int state = localMatrix
                    ? FactorLoadedMatrix(a, mat, col, piv, infoBuf.Get<float>(), m, n, lda, stride, pivot != nullptr)
                    : FactorGlobal(a, col, other, infoBuf.Get<float>(), piv, m, n, lda, steps, pivot != nullptr);
    if (pivot != nullptr)
    {
        GlobalTensor<int> out;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivot));
        Fence<HardEvent::S_MTE3>();
        DataCopyExtParams copy{1, static_cast<uint32_t>(steps * sizeof(int)), 0, 0, 0};
        DataCopyPad(out, piv, copy);
        Fence<HardEvent::MTE3_S>();
    }
    GlobalTensor<int> result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    StoreInt(result, tmp, state);
}
extern "C" void real_lu_fill_info_do(GM_ADDR info, int count, int value, aclrtStream stream)
{
    int blocks = std::min<int64_t>(real_lu_vector_cores(), (static_cast<int64_t>(count) + 1023) / 1024);
    real_lu_fill_info_kernel<<<blocks, nullptr, stream>>>(info, count, value);
}
extern "C" void real_lu_factor_do(GM_ADDR a, int m, int n, int lda, GM_ADDR piv, GM_ADDR info, aclrtStream stream)
{
    real_lu_factor_kernel<<<1, nullptr, stream>>>(a, m, n, lda, piv, info);
}

// A short first panel determines the pivots and unit-lower factor. Independent
// trailing column tiles then apply P and solve L*U=P*A in parallel.
__global__ __aicore__ void real_lu_thin_panel_kernel(GM_ADDR matrix, GM_ADDR workspace, GM_ADDR pivot, GM_ADDR info,
                                                     int m, int lda)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    int stride = 8;
    while (stride < m)
    {
        stride *= 2;
    }
    TBuf<TPosition::VECCALC> factorBuf, columnBuf, pivBuf, infoBuf;
    pipe.InitBuffer(factorBuf, stride * m * sizeof(float));
    pipe.InitBuffer(columnBuf, stride * sizeof(float));
    pipe.InitBuffer(pivBuf, stride * sizeof(int));
    pipe.InitBuffer(infoBuf, 32);
    auto factor = factorBuf.Get<float>(), column = columnBuf.Get<float>();
    auto piv = pivBuf.Get<int>();
    GlobalTensor<float> a, w;
    a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matrix));
    w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
    for (int col = 0; col < m; ++col)
    {
        Load(factor[col * stride], a[static_cast<int64_t>(col) * lda], m);
    }
    int state = FactorLocal(factor, column, m, m, stride, piv, pivot != nullptr);
    Store(w, factor, stride * m);
    for (int col = 0; col < m; ++col)
    {
        Store(a[static_cast<int64_t>(col) * lda], factor[col * stride], m);
    }
    GlobalTensor<int> internal, external, result;
    internal.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(workspace) + stride * m);
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{1, static_cast<uint32_t>(m * sizeof(int)), 0, 0, 0};
    DataCopyPad(internal, piv, copy);
    if (pivot != nullptr)
    {
        external.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivot));
        DataCopyPad(external, piv, copy);
    }
    Fence<HardEvent::MTE3_S>();
    StoreInt(result, infoBuf.Get<int>(), state);
}

namespace
{
class ThinColumns
{
   public:
    __aicore__ inline ThinColumns(TPipe &pipe, GM_ADDR matrix, GM_ADDR workspace, int m, int lda) : m(m), lda(lda)
    {
        while (stride < m)
        {
            stride *= 2;
        }
        entries = stride * lanes;
        for (int s = stride; s > 1; s /= 2)
        {
            ++shift;
        }
        pipe.InitBuffer(rawBuf, entries * sizeof(float));
        pipe.InitBuffer(rowsBuf, entries * sizeof(float));
        pipe.InitBuffer(indexBuf, entries * sizeof(int));
        pipe.InitBuffer(factorBuf, stride * m * sizeof(float));
        pipe.InitBuffer(tmpBuf, lanes * sizeof(float));
        pipe.InitBuffer(pivBuf, stride * sizeof(int));
        raw = rawBuf.Get<float>();
        rows = rowsBuf.Get<float>();
        factor = factorBuf.Get<float>();
        tmp = tmpBuf.Get<float>();
        index = indexBuf.Get<uint32_t>();
        piv = pivBuf.Get<int>();
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(matrix));
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace));
        p.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(workspace) + stride * m);
        Load(factor, w, stride * m);
        DataCopyExtParams pivotCopy{1, static_cast<uint32_t>(m * sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int> pivotPad{true, 0, static_cast<uint8_t>(Align(m) - m), 0};
        DataCopyPad(piv, p, pivotCopy, pivotPad);
        Fence<HardEvent::MTE2_S>();
    }
    __aicore__ inline void Run(int first, int count)
    {
        LoadRows(first, count);
        ApplyPivots();
        SolveRows();
        StoreColumns(first, count);
    }

   private:
    __aicore__ inline void LoadRows(int first, int count)
    {
        CreateVecIndex(index.ReinterpretCast<int>(), 0, lanes);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index, index, static_cast<uint32_t>(shift + 2), lanes);
        PipeBarrier<PIPE_V>();
        for (int row = 1; row < stride; ++row)
        {
            Adds(index[row * lanes].ReinterpretCast<int>(), index.ReinterpretCast<int>(), row * 4, lanes);
        }
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams copy{static_cast<uint16_t>(count), static_cast<uint32_t>(m * sizeof(float)),
                               static_cast<uint32_t>((lda - m) * sizeof(float)),
                               static_cast<uint32_t>((stride - Align(m)) / 8), 0};
        DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(Align(m) - m), 0};
        DataCopyPad(raw, a[static_cast<int64_t>(first) * lda], copy, pad);
        Fence<HardEvent::MTE2_V>();
        Gather(rows, raw, index, 0, entries);
        Fence<HardEvent::V_S>();
    }
    __aicore__ inline void ApplyPivots()
    {
        for (int k = 0; k < m; ++k)
        {
            int selected = piv.GetValue(k) - 1;
            if (selected == k)
            {
                continue;
            }
            Fence<HardEvent::S_V>();
            Adds(tmp, rows[k * lanes], 0.0f, lanes);
            PipeBarrier<PIPE_V>();
            Adds(rows[k * lanes], rows[selected * lanes], 0.0f, lanes);
            Adds(rows[selected * lanes], tmp, 0.0f, lanes);
            Fence<HardEvent::V_S>();
        }
    }
    __aicore__ inline void SolveRows()
    {
        for (int k = 0; k < m; ++k)
        {
            if (factor.GetValue(k * stride + k) == 0.0f)
            {
                continue;
            }
            for (int row = k + 1; row < m; ++row)
            {
                float coefficient = -factor.GetValue(k * stride + row);
                Fence<HardEvent::S_V>();
                Axpy(rows[row * lanes], rows[k * lanes], coefficient, lanes);
                Fence<HardEvent::V_S>();
            }
        }
    }
    __aicore__ inline void StoreColumns(int first, int count)
    {
        // Build packed-column output indices without overwriting the row data.
        Fence<HardEvent::S_V>();
        CreateVecIndex(index.ReinterpretCast<int>(), 0, entries);
        Duplicate(tmp.ReinterpretCast<int>(), stride - 1, lanes);
        PipeBarrier<PIPE_V>();
        ShiftRight(raw.ReinterpretCast<uint32_t>(), index, static_cast<uint32_t>(shift), entries);
        for (int start = 0; start < entries; start += lanes)
        {
            And(index[start].ReinterpretCast<int>(), index[start].ReinterpretCast<int>(), tmp.ReinterpretCast<int>(),
                lanes);
        }
        PipeBarrier<PIPE_V>();
        ShiftLeft(index, index, 9u, entries);
        PipeBarrier<PIPE_V>();
        Add(index.ReinterpretCast<int>(), index.ReinterpretCast<int>(), raw.ReinterpretCast<int>(), entries);
        PipeBarrier<PIPE_V>();
        ShiftLeft(index, index, 2u, entries);
        PipeBarrier<PIPE_V>();
        Gather(raw, rows, index, 0, entries);
        Fence<HardEvent::V_MTE3>();
        DataCopyExtParams store{static_cast<uint16_t>(count), static_cast<uint32_t>(m * sizeof(float)),
                                static_cast<uint32_t>((stride - Align(m)) / 8),
                                static_cast<uint32_t>((lda - m) * sizeof(float)), 0};
        DataCopyPad(a[static_cast<int64_t>(first) * lda], raw, store);
        Fence<HardEvent::MTE3_S>();
    }
    static constexpr int lanes = 512;
    int m, lda, stride = 8, entries = 0, shift = 0;
    TBuf<TPosition::VECCALC> rawBuf, rowsBuf, indexBuf, factorBuf, tmpBuf, pivBuf;
    LocalTensor<float> raw, rows, factor, tmp;
    LocalTensor<uint32_t> index;
    LocalTensor<int> piv;
    GlobalTensor<float> a, w;
    GlobalTensor<int> p;
};
}  // namespace

__global__ __aicore__ void real_lu_thin_columns_kernel(GM_ADDR matrix, GM_ADDR workspace, int m, int n, int lda)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    ThinColumns op(pipe, matrix, workspace, m, lda);
    constexpr int lanes = 512;
    int tiles = (n - m + lanes - 1) / lanes;
    for (int tile = GetBlockIdx(); tile < tiles; tile += GetBlockNum())
    {
        int first = m + tile * lanes;
        int count = n - first < lanes ? n - first : lanes;
        op.Run(first, count);
    }
}

extern "C" void real_lu_thin_factor_do(GM_ADDR a, GM_ADDR workspace, int m, int n, int lda, GM_ADDR piv, GM_ADDR info,
                                       aclrtStream stream)
{
    real_lu_thin_panel_kernel<<<1, nullptr, stream>>>(a, workspace, piv, info, m, lda);
    real_lu_thin_columns_kernel<<<std::min(real_lu_vector_cores(), (n - m + 511) / 512), nullptr, stream>>>(
        a, workspace, m, n, lda);
}
