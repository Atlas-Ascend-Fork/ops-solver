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
extern "C" int real_lu_gemm_do(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR info, int pitch, int rows, int columns,
                               int depth, int64_t matrixStride, int batch, int64_t offsetA, int64_t offsetB,
                               int64_t offsetC, bool transposeB, aclrtStream stream);

__aicore__ inline void BuildInversePermutation(LocalTensor<int> permutation, LocalTensor<int> inverse,
                                               GlobalTensor<int> pivots, int n, int matrix)
{
    for (int i = 0; i < n; ++i)
    {
        permutation.SetValue(i, i);
    }
    for (int k = 0; k < n; ++k)
    {
        int selected = pivots.GetValue(static_cast<int64_t>(matrix) * n + k) - 1;
        int value = permutation.GetValue(k);
        permutation.SetValue(k, permutation.GetValue(selected));
        permutation.SetValue(selected, value);
    }
    for (int i = 0; i < n; ++i)
    {
        inverse.SetValue(permutation.GetValue(i), i);
    }
}

struct InversePackOperation
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> columnBuf, permutationBuf, inverseBuf;
    LocalTensor<float> column;
    LocalTensor<int> permutation, inverse;
    GlobalTensor<uint64_t> pointers;
    GlobalTensor<int> p;
    GlobalTensor<float> a, w;
    GM_ADDR input;
    GM_ADDR workspace;
    int n, lda, pitch, batch, tileColumns, columnTiles, previous = -1;
    int64_t matrixStride, span, jobs;
    uint64_t sourceGap;
    bool bulk, tiled, arrays, identityPivots;

    __aicore__ inline bool Init(GM_ADDR in, GM_ADDR pivots, GM_ADDR work, int order, int leading, int padded,
                                int64_t storage, int count, bool pointerArray, bool identity)
    {
        input = in;
        workspace = work;
        n = order;
        lda = leading;
        pitch = padded;
        matrixStride = storage;
        batch = count;
        arrays = pointerArray;
        identityPivots = identity;
        if (pitch == 0)
        {
            return false;
        }
        if (n <= 0 || pitch < 0 || batch <= 0 || lda < n)
        {
            return false;
        }
        sourceGap = static_cast<uint64_t>(lda - n) * sizeof(float);
        bulk = (n <= 128 || (n <= 192 && batch > 1)) && sourceGap <= 0xffffffffULL;
        tiled = n > 192 && batch > 1 && sourceGap <= 0xffffffffULL;
        tileColumns = 24576 / pitch;
        tileColumns = tileColumns > 32 ? 32 : tileColumns;
        if (tileColumns == 0)
        {
            return false;
        }
        if (tileColumns < 0)
        {
            return false;
        }
        columnTiles = (pitch + tileColumns - 1) / tileColumns;
        int elements = bulk ? pitch * pitch : (tiled ? tileColumns * pitch : pitch);
        pipe.InitBuffer(columnBuf, elements * sizeof(float));
        pipe.InitBuffer(permutationBuf, pitch * sizeof(int));
        pipe.InitBuffer(inverseBuf, pitch * sizeof(int));
        column = columnBuf.Get<float>();
        permutation = permutationBuf.Get<int>();
        inverse = inverseBuf.Get<int>();
        pointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(input));
        p.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivots));
        jobs = bulk ? batch : static_cast<int64_t>(batch) * (tiled ? columnTiles : pitch);
        span = static_cast<int64_t>(pitch) * pitch;
        return true;
    }

    __aicore__ inline void PrepareMatrix(int matrix)
    {
        if (matrix == previous)
        {
            return;
        }
        uint64_t address =
            arrays ? pointers.GetValue(matrix)
                   : reinterpret_cast<uint64_t>(input) + static_cast<uint64_t>(matrix) * lda * n * sizeof(float);
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + matrixStride * matrix);
        if (!identityPivots)
        {
            BuildInversePermutation(permutation, inverse, p, n, matrix);
        }
        previous = matrix;
    }

    __aicore__ inline void PackColumns(int col, int columns, int realColumns)
    {
        Duplicate(column, 0.0f, columns * pitch);
        Fence<HardEvent::V_MTE2>();
        if (realColumns > 0)
        {
            DataCopyExtParams copy{static_cast<uint16_t>(realColumns), static_cast<uint32_t>(n * sizeof(float)),
                                   static_cast<uint32_t>(sourceGap), static_cast<uint32_t>((pitch - Align(n)) / 8), 0};
            DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(Align(n) - n), 0.0f};
            DataCopyPad(column, a[static_cast<int64_t>(col) * lda], copy, pad);
        }
        Fence<HardEvent::MTE2_S>();
        for (int localCol = realColumns; localCol < columns; ++localCol)
        {
            column.SetValue(localCol * pitch + col + localCol, 1.0f);
        }
        Store(w[static_cast<int64_t>(col) * pitch], column, columns * pitch);
        Duplicate(column, 0.0f, columns * pitch);
        Fence<HardEvent::V_S>();
        for (int localCol = 0; localCol < realColumns; ++localCol)
        {
            int actualCol = col + localCol;
            column.SetValue(localCol * pitch + (identityPivots ? actualCol : inverse.GetValue(actualCol)), 1.0f);
        }
        Store(w[span + static_cast<int64_t>(col) * pitch], column, columns * pitch);
    }

    __aicore__ inline void PackSingle(int col)
    {
        Duplicate(column, 0.0f, pitch);
        Fence<HardEvent::V_S>();
        if (col < n)
        {
            Load(column, a[static_cast<int64_t>(col) * lda], n);
        }
        else
        {
            column.SetValue(col, 1.0f);
        }
        Store(w[static_cast<int64_t>(col) * pitch], column, pitch);
        Duplicate(column, 0.0f, pitch);
        Fence<HardEvent::V_S>();
        if (col < n)
        {
            column.SetValue(identityPivots ? col : inverse.GetValue(col), 1.0f);
        }
        Store(w[span + static_cast<int64_t>(col) * pitch], column, pitch);
    }

    __aicore__ inline void Run()
    {
        if (columnTiles == 0)
        {
            return;
        }
        if (pitch == 0)
        {
            return;
        }
        for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum())
        {
            int matrix = bulk ? job : (tiled ? job / columnTiles : job / pitch);
            int col = bulk ? 0 : (tiled ? (job % columnTiles) * tileColumns : job % pitch);
            PrepareMatrix(matrix);
            if (bulk)
            {
                PackColumns(0, pitch, n);
            }
            else if (tiled)
            {
                int columns = pitch - col < tileColumns ? pitch - col : tileColumns;
                int realColumns = col < n ? (n - col < columns ? n - col : columns) : 0;
                PackColumns(col, columns, realColumns);
            }
            else
            {
                PackSingle(col);
            }
        }
    }
};

__global__ __aicore__ void real_lu_pack_inverse_kernel(GM_ADDR input, GM_ADDR pivots, GM_ADDR workspace, int n, int lda,
                                                       int pitch, int64_t matrixStride, int batch, bool arrays,
                                                       bool identityPivots)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    InversePackOperation operation;
    if (operation.Init(input, pivots, workspace, n, lda, pitch, matrixStride, batch, arrays, identityPivots))
    {
        operation.Run();
    }
}

__aicore__ inline void InitializeIdentity(LocalTensor<float> rhs)
{
    Duplicate(rhs, 0.0f, 4096);
    Fence<HardEvent::V_S>();
    for (int i = 0; i < 64; ++i)
    {
        rhs.SetValue(i * 64 + i, 1.0f);
    }
    Fence<HardEvent::S_V>();
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__aicore__ inline void SolveDiagonalVector(LocalTensor<float> right, LocalTensor<float> factors,
                                           LocalTensor<float> reciprocals, bool upper, int active)
{
    auto rhs = reinterpret_cast<__ubuf__ float *>(right.GetPhyAddr());
    auto lu = reinterpret_cast<__ubuf__ float *>(factors.GetPhyAddr());
    auto reciprocal = reinterpret_cast<__ubuf__ float *>(reciprocals.GetPhyAddr());
    uint16_t count = static_cast<uint16_t>(active);
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<float> source, negative, coefficient, target;
        auto valid = MicroAPI::CreateMask<float>();
        for (uint16_t step = 0; step < count; ++step)
        {
            uint16_t k = upper ? static_cast<uint16_t>(count - 1 - step) : step;
            uint16_t begin = upper ? 0 : static_cast<uint16_t>(k + 1), end = upper ? k : count;
            MicroAPI::LoadAlign<float>(source, rhs + k * 64);
            if (upper)
            {
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(coefficient, reciprocal + k);
                MicroAPI::Mul<float>(source, source, coefficient, valid);
            }
            else
            {
                MicroAPI::Muls<float, float>(source, source, 1.0f, valid);
            }
            MicroAPI::StoreAlign<float>(rhs + k * 64, source, valid);
            MicroAPI::Neg<float>(negative, source, valid);
            for (uint16_t row = begin; row < end; ++row)
            {
                MicroAPI::LoadAlign<float>(target, rhs + row * 64);
                MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_BRC_B32>(coefficient, lu + k * 64 + row);
                MicroAPI::MulAddDst<float>(target, negative, coefficient, valid);
                MicroAPI::StoreAlign<float>(rhs + row * 64, target, valid);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
        }
    }
}
#else
__aicore__ inline void SolveDiagonalGeneric(LocalTensor<float> right, LocalTensor<float> factors,
                                            LocalTensor<float> reciprocals, bool upper, int active)
{
    for (int step = 0; step < active; ++step)
    {
        int k = upper ? active - 1 - step : step;
        float scale = upper ? reciprocals.GetValue(k) : 1.0f;
        uint16_t begin = upper ? 0 : k + 1, end = upper ? k : active;
        Muls(right[k * 64], right[k * 64], scale, 64);
        PipeBarrier<PIPE_V>();
        for (int row = begin; row < end; ++row)
        {
            Axpy(right[row * 64], right[k * 64], -factors.GetValue(k * 64 + row), 64);
        }
        PipeBarrier<PIPE_V>();
    }
}
#endif
__aicore__ inline void SolveDiagonal(LocalTensor<float> right, LocalTensor<float> factors,
                                     LocalTensor<float> reciprocals, bool upper, int active)
{
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    SolveDiagonalVector(right, factors, reciprocals, upper, active);
#else
    SolveDiagonalGeneric(right, factors, reciprocals, upper, active);
#endif
    Fence<HardEvent::V_S>();
}

__aicore__ inline int ReadDiagonalReciprocals(LocalTensor<float> lu, LocalTensor<float> reciprocals,
                                              GlobalTensor<int> flags, LocalTensor<int> infoBuffer, int first,
                                              int block, bool &normal)
{
    int state = 0;
    normal = true;
    for (int k = 0; k < 64; ++k)
    {
        float value = lu.GetValue(k * 64 + k), magnitude = value < 0 ? -value : value;
        if (value == 0 && state == 0)
        {
            state = first + k + 1;
        }
        if (magnitude < 1.1754943508222875e-38f || magnitude > 1.0e37f)
        {
            normal = false;
        }
        reciprocals.SetValue(64 + k, value);
    }
    StoreInt(flags[block], infoBuffer, state);

    return state;
}

__aicore__ inline void StoreDiagonalInverses(GlobalTensor<float> w, LocalTensor<float> rhs, LocalTensor<float> lu,
                                             LocalTensor<float> reciprocals, int pitch, int64_t span, int first,
                                             int active)
{
    DataCopyExtParams store{64, 64 * sizeof(float), 0, static_cast<uint32_t>((pitch - 64) * sizeof(float)), 0};
    InitializeIdentity(rhs);
    SolveDiagonal(rhs, lu, reciprocals, false, active);
    Fence<HardEvent::S_MTE3>();
    DataCopyPad(w[3 * span + static_cast<int64_t>(first) * pitch + first], rhs, store);
    Fence<HardEvent::MTE3_S>();
    InitializeIdentity(rhs);
    SolveDiagonal(rhs, lu, reciprocals, true, active);
    Fence<HardEvent::S_MTE3>();
    DataCopyPad(w[4 * span + static_cast<int64_t>(first) * pitch + first], rhs, store);
    Fence<HardEvent::MTE3_S>();
}

__aicore__ inline void ReciprocateInverseDiagonal(LocalTensor<float> reciprocals, bool normal)
{
    Fence<HardEvent::S_V>();
    Duplicate(reciprocals, 1.0f, 64);
    PipeBarrier<PIPE_V>();
#if (__NPU_ARCH__ == 3510)
    if (normal)
    {
        Div<float, NORMAL_DIVIDE>(reciprocals, reciprocals, reciprocals[64], 64);
    }
    else
    {
        Divide(reciprocals, reciprocals, reciprocals[64], 64);
    }
#else
    Divide(reciprocals, reciprocals, reciprocals[64], 64);
#endif
    Fence<HardEvent::V_S>();
}

__global__ __aicore__ void real_lu_diagonal_inverse_kernel(GM_ADDR workspace, int n, int pitch, int64_t matrixStride,
                                                           int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (batch == 0)
    {
        return;
    }
    if (batch < 0 || pitch <= 0)
    {
        return;
    }
    TPipe pipe;
    TBuf<TPosition::VECCALC> luBuf, rhsBuf, reciprocalBuf, infoBuf;
    pipe.InitBuffer(luBuf, 4096 * sizeof(float));
    pipe.InitBuffer(rhsBuf, 4096 * sizeof(float));
    pipe.InitBuffer(reciprocalBuf, 128 * sizeof(float));
    pipe.InitBuffer(infoBuf, 32);
    auto lu = luBuf.Get<float>(), rhs = rhsBuf.Get<float>(), reciprocals = reciprocalBuf.Get<float>();
    int blocks = pitch / 64;
    int64_t span = static_cast<int64_t>(pitch) * pitch, jobs = static_cast<int64_t>(batch) * blocks;
    GlobalTensor<float> w;
    GlobalTensor<int> flags;
    for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum())
    {
        int matrix = job % batch, block = job / batch, first = block * 64;
        int active = n - first < 64 ? n - first : 64;
        w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace) + matrixStride * matrix);
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(workspace) + matrixStride * matrix + 5 * span);
        Fence<HardEvent::V_MTE2>();
        DataCopyExtParams load{64, 64 * sizeof(float), static_cast<uint32_t>((pitch - 64) * sizeof(float)), 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(lu, w[static_cast<int64_t>(first) * pitch + first], load, pad);
        Fence<HardEvent::MTE2_S>();
        bool normal = true;
        if (ReadDiagonalReciprocals(lu, reciprocals, flags, infoBuf.Get<int>(), first, block, normal) != 0)
        {
            continue;
        }
        ReciprocateInverseDiagonal(reciprocals, normal);
        StoreDiagonalInverses(w, rhs, lu, reciprocals, pitch, span, first, active);
    }
}

__global__ __aicore__ void real_lu_inverse_info_kernel(GM_ADDR workspace, GM_ADDR info, int pitch, int64_t matrixStride,
                                                       int batch)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    pipe.InitBuffer(buffer, 32);
    GlobalTensor<int> flags, result;
    result.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
    int64_t span = static_cast<int64_t>(pitch) * pitch;
    for (int matrix = GetBlockIdx(); matrix < batch; matrix += GetBlockNum())
    {
        flags.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(workspace) + matrixStride * matrix + 5 * span);
        int state = 0;
        for (int block = 0; block < pitch / 64; ++block)
        {
            int candidate = flags.GetValue(block);
            if (candidate > 0 && (state == 0 || candidate < state))
            {
                state = candidate;
            }
        }
        StoreInt(result[matrix], buffer.Get<int>(), state);
    }
}

__aicore__ inline void SubtractInverseColumns(GlobalTensor<float> w, LocalTensor<float> a, LocalTensor<float> b,
                                              int64_t span, int64_t offset, int pitch, int rows, int localStride,
                                              int columns)
{
    DataCopyExtParams copy{static_cast<uint16_t>(columns), static_cast<uint32_t>(rows * sizeof(float)),
                           static_cast<uint32_t>((pitch - rows) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(localStride - rows), 0.0f};
    Fence<HardEvent::V_MTE2>();
    DataCopyPad(a, w[span + offset], copy, pad);
    DataCopyPad(b, w[2 * span + offset], copy, pad);
    Fence<HardEvent::MTE2_V>();
    Sub(a, a, b, columns * localStride);
    Fence<HardEvent::V_MTE3>();
    DataCopyExtParams store{static_cast<uint16_t>(columns), static_cast<uint32_t>(rows * sizeof(float)), 0,
                            static_cast<uint32_t>((pitch - rows) * sizeof(float)), 0};
    DataCopyPad(w[span + offset], a, store);
    Fence<HardEvent::MTE3_S>();
}
template <typename Operation>
__aicore__ inline void RunInverseColumnJobs(Operation &operation)
{
    const int columnTiles = operation.columnTiles;
    if (columnTiles == 0)
    {
        return;
    }
    const int n = operation.n;
    if (n == 0)
    {
        return;
    }
    for (int64_t job = GetBlockIdx(); job < operation.jobs; job += GetBlockNum())
    {
        int matrix = operation.bulk ? job : (operation.tiled ? job / columnTiles : job / n);
        int col = operation.bulk ? 0 : (operation.tiled ? (job % columnTiles) * operation.tileColumns : job % n);
        if (operation.state.GetValue(matrix) != 0)
        {
            continue;
        }
        operation.w.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(operation.workspace) +
                                    operation.matrixStride * matrix);
        operation.RunColumn(matrix, col);
    }
}

struct InverseUpdateOperation
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> aBuf, bBuf;
    LocalTensor<float> a, b;
    GlobalTensor<float> w;
    GlobalTensor<int> state;
    GM_ADDR workspace;
    int n, pitch, batch, first, rows, localStride, tileColumns, columnTiles;
    int64_t matrixStride, span, jobs;
    bool bulk, tiled;
    __aicore__ inline bool Init(GM_ADDR work, GM_ADDR info, int order, int padded, int64_t storage, int count,
                                int start, int activeRows)
    {
        workspace = work;
        n = order;
        pitch = padded;
        matrixStride = storage;
        batch = count;
        first = start;
        rows = activeRows;

        if (n <= 0 || rows <= 0 || pitch <= 0 || batch <= 0)
        {
            return false;
        }
        bulk = (n <= 128 || (n <= 192 && batch > 1));
        tiled = (batch == 1 && n > 128) || (batch > 1 && n > 192);
        localStride = Align(rows);
        if (localStride == 0)
        {
            return false;
        }
        if (localStride < 0)
        {
            return false;
        }
        tileColumns = 24576 / localStride;
        int tileLimit = batch == 1 ? 16 : 32;
        tileColumns = tileColumns < 1 ? 1 : (tileColumns > tileLimit ? tileLimit : tileColumns);
        if (tileColumns == 0)
        {
            return false;
        }
        columnTiles = (n + tileColumns - 1) / tileColumns;
        int elements = bulk ? n * localStride : (tiled ? tileColumns * localStride : localStride);
        pipe.InitBuffer(aBuf, elements * sizeof(float));
        pipe.InitBuffer(bBuf, elements * sizeof(float));
        a = aBuf.Get<float>();
        b = bBuf.Get<float>();
        state.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
        span = static_cast<int64_t>(pitch) * pitch;
        jobs = bulk ? batch : static_cast<int64_t>(batch) * (tiled ? columnTiles : n);

        return true;
    }
    __aicore__ inline void RunColumn(int matrix, int col)
    {
        int64_t offset = static_cast<int64_t>(col) * pitch + first;
        if (bulk || tiled)
        {
            int columns = bulk ? n : (n - col < tileColumns ? n - col : tileColumns);
            SubtractInverseColumns(w, a, b, span, offset, pitch, rows, localStride, columns);
            return;
        }
        Load(a, w[span + offset], rows);
        Load(b, w[2 * span + offset], rows);
        Fence<HardEvent::S_V>();
        Sub(a, a, b, rows);
        Fence<HardEvent::V_S>();
        Store(w[span + offset], a, rows);
    }
};

__global__ __aicore__ void real_lu_inverse_update_kernel(GM_ADDR workspace, GM_ADDR info, int n, int pitch,
                                                         int64_t matrixStride, int batch, int first, int rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    InverseUpdateOperation operation;
    if (operation.Init(workspace, info, n, pitch, matrixStride, batch, first, rows))
    {
        RunInverseColumnJobs(operation);
    }
}

__aicore__ inline void UnpackInverseBulk(GlobalTensor<float> w, GlobalTensor<float> c, LocalTensor<float> column,
                                         int64_t span, int pitch, int n, uint64_t destinationGap)
{
    Load(column, w[span], pitch * pitch);
    Fence<HardEvent::S_MTE3>();
    DataCopyExtParams copy{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                           static_cast<uint32_t>((pitch - Align(n)) / 8), static_cast<uint32_t>(destinationGap), 0};
    DataCopyPad(c, column, copy);
    Fence<HardEvent::MTE3_S>();
}

__aicore__ inline void UnpackInverseTile(GlobalTensor<float> w, GlobalTensor<float> c, LocalTensor<float> column,
                                         int64_t span, int pitch, int n, int ldc, int col, int localStride, int columns,
                                         uint64_t destinationGap)
{
    DataCopyExtParams copy{static_cast<uint16_t>(columns), static_cast<uint32_t>(n * sizeof(float)),
                           static_cast<uint32_t>((pitch - n) * sizeof(float)), 0, 0};
    DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(localStride - n), 0.0f};
    Fence<HardEvent::V_MTE2>();
    DataCopyPad(column, w[span + static_cast<int64_t>(col) * pitch], copy, pad);
    Fence<HardEvent::MTE2_S>();
    DataCopyExtParams store{static_cast<uint16_t>(columns), static_cast<uint32_t>(n * sizeof(float)), 0,
                            static_cast<uint32_t>(destinationGap), 0};
    Fence<HardEvent::S_MTE3>();
    DataCopyPad(c[static_cast<int64_t>(col) * ldc], column, store);
    Fence<HardEvent::MTE3_S>();
}

struct InverseUnpackOperation
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> buffer;
    LocalTensor<float> column;
    GlobalTensor<float> w, c;
    GlobalTensor<int> state;
    GlobalTensor<uint64_t> pointers;
    GM_ADDR workspace;
    GM_ADDR output;
    int n, ldc, pitch, batch, localStride, tileColumns, columnTiles;
    int64_t matrixStride, span, jobs;
    uint64_t destinationGap;
    bool bulk, tiled, arrays;
    __aicore__ inline bool Init(GM_ADDR work, GM_ADDR out, GM_ADDR info, int order, int leading, int padded,
                                int64_t storage, int count, bool pointerArray)
    {
        workspace = work;
        output = out;
        n = order;
        ldc = leading;
        pitch = padded;
        matrixStride = storage;
        batch = count;
        arrays = pointerArray;

        if (n <= 0 || pitch <= 0 || batch <= 0 || ldc < n)
        {
            return false;
        }
        destinationGap = static_cast<uint64_t>(ldc - n) * sizeof(float);
        bulk = (n <= 128 || (n <= 192 && batch > 1)) && destinationGap <= 0xffffffffULL;
        tiled = n > 192 && batch > 1 && destinationGap <= 0xffffffffULL;
        localStride = Align(n);
        if (localStride == 0)
        {
            return false;
        }
        if (localStride < 0)
        {
            return false;
        }
        tileColumns = 24576 / localStride;
        tileColumns = tileColumns > 32 ? 32 : tileColumns;
        if (tileColumns == 0)
        {
            return false;
        }
        if (tileColumns < 0)
        {
            return false;
        }
        columnTiles = (n + tileColumns - 1) / tileColumns;
        int elements = bulk ? pitch * pitch : (tiled ? tileColumns * localStride : localStride);
        pipe.InitBuffer(buffer, elements * sizeof(float));
        column = buffer.Get<float>();
        state.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
        pointers.SetGlobalBuffer(reinterpret_cast<__gm__ uint64_t *>(output));
        span = static_cast<int64_t>(pitch) * pitch;
        jobs = bulk ? batch : static_cast<int64_t>(batch) * (tiled ? columnTiles : n);

        return true;
    }
    __aicore__ inline void RunColumn(int matrix, int col)
    {
        uint64_t address =
            arrays ? pointers.GetValue(matrix)
                   : reinterpret_cast<uint64_t>(output) + static_cast<uint64_t>(matrix) * ldc * n * sizeof(float);
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
        if (bulk)
        {
            UnpackInverseBulk(w, c, column, span, pitch, n, destinationGap);
            return;
        }
        if (tiled)
        {
            int columns = n - col < tileColumns ? n - col : tileColumns;
            UnpackInverseTile(w, c, column, span, pitch, n, ldc, col, localStride, columns, destinationGap);
            return;
        }
        Load(column, w[span + static_cast<int64_t>(col) * pitch], n);
        Store(c[static_cast<int64_t>(col) * ldc], column, n);
    }
};

__global__ __aicore__ void real_lu_unpack_inverse_kernel(GM_ADDR workspace, GM_ADDR output, GM_ADDR info, int n,
                                                         int ldc, int pitch, int64_t matrixStride, int batch,
                                                         bool arrays)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    InverseUnpackOperation operation;
    if (operation.Init(workspace, output, info, n, ldc, pitch, matrixStride, batch, arrays))
    {
        RunInverseColumnJobs(operation);
    }
}

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void VectorInversePermutationVF(
    __ubuf__ int32_t *pivots, __ubuf__ int32_t *permutation, uint16_t n, uint16_t group, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> lane, currentIndex, selectedIndex, pivotIndex;
    MicroAPI::RegTensor<int32_t> label, selected, first, second;
    auto valid = MicroAPI::UpdateMask<int32_t>(count);
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    for (uint16_t row = 0; row < n; ++row)
    {
        MicroAPI::Duplicate<int32_t>(label, static_cast<int32_t>(row), valid);
        MicroAPI::Adds(currentIndex, lane, static_cast<uint32_t>(row) * group, valid);
        MicroAPI::Scatter(permutation, label, currentIndex, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    MicroAPI::Muls(pivotIndex, lane, static_cast<uint32_t>(n), valid);
    for (uint16_t k = 0; k < n; ++k)
    {
        MicroAPI::Gather(selected, pivots + k, pivotIndex, valid);
        MicroAPI::Adds(selected, selected, -1, valid);
        MicroAPI::Muls(selectedIndex, reinterpret_cast<MicroAPI::RegTensor<uint32_t> &>(selected),
                       static_cast<uint32_t>(group), valid);
        MicroAPI::Add(selectedIndex, selectedIndex, lane, valid);
        MicroAPI::Adds(currentIndex, lane, static_cast<uint32_t>(k) * group, valid);
        MicroAPI::Gather(first, permutation, currentIndex, valid);
        MicroAPI::Gather(second, permutation, selectedIndex, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_LOAD, MicroAPI::MemType::VEC_STORE>();
        MicroAPI::Scatter(permutation, first, selectedIndex, valid);
        MicroAPI::Scatter(permutation, second, currentIndex, valid);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void VectorInverseDiagonalVF(
    __ubuf__ float *lu, __ubuf__ float *diagonal, __ubuf__ int32_t *info, uint16_t n, uint16_t colStride,
    uint16_t group, uint32_t matrixStride, uint32_t count)
{
    MicroAPI::RegTensor<uint32_t> lane, matrixOffset, diagonalIndex;
    MicroAPI::RegTensor<float> value;
    MicroAPI::RegTensor<int32_t> state, firstZero;
    MicroAPI::MaskReg zero, unset, replace;
    auto valid = MicroAPI::UpdateMask<int32_t>(count);
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    MicroAPI::Muls(matrixOffset, lane, matrixStride, valid);
    MicroAPI::Duplicate<int32_t>(state, 0, valid);
    for (uint16_t row = 0; row < n; ++row)
    {
        MicroAPI::Gather(value, lu + static_cast<uint32_t>(row) * colStride + row, matrixOffset, valid);
        MicroAPI::Adds(diagonalIndex, lane, static_cast<uint32_t>(row) * group, valid);
        MicroAPI::Scatter(diagonal, value, diagonalIndex, valid);
        MicroAPI::CompareScalar<float, CMPMODE::EQ>(zero, value, 0.0f, valid);
        MicroAPI::CompareScalar<int32_t, CMPMODE::EQ>(unset, state, 0, valid);
        MicroAPI::MaskAnd(replace, zero, unset, valid);
        MicroAPI::Duplicate<int32_t>(firstZero, static_cast<int32_t>(row + 1), valid);
        MicroAPI::Select<int32_t>(state, firstZero, state, replace);
    }
    MicroAPI::Scatter(info, state, lane, valid);
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

__simd_callee__ inline void VectorInverseUpdateRows(
    __ubuf__ float *lu, __ubuf__ float *rhs, uint16_t k, uint16_t begin, uint16_t end, uint16_t colStride,
    MicroAPI::RegTensor<uint32_t> &matrixOffset, MicroAPI::MaskReg &valid, MicroAPI::RegTensor<float> &negative,
    MicroAPI::RegTensor<float> &coefficient, MicroAPI::RegTensor<float> &target,
    MicroAPI::RegTensor<float> &coefficient1, MicroAPI::RegTensor<float> &target1)
{
    uint16_t pairCount = (end - begin) / 2;
    uint16_t pairedEnd = begin + pairCount * 2;
    for (uint16_t pair = 0; pair < pairCount; ++pair)
    {
        uint16_t row = begin + pair * 2;
        MicroAPI::Gather(coefficient, lu + static_cast<uint32_t>(k) * colStride + row, matrixOffset, valid);
        MicroAPI::Gather(coefficient1, lu + static_cast<uint32_t>(k) * colStride + row + 1, matrixOffset, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target, rhs + static_cast<uint32_t>(row) * 64, 1u, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target1, rhs + static_cast<uint32_t>(row + 1) * 64, 1u, valid);
        MicroAPI::MulAddDst<float>(target, negative, coefficient, valid);
        MicroAPI::MulAddDst<float>(target1, negative, coefficient1, valid);
        MicroAPI::StoreAlign<float>(rhs + static_cast<uint32_t>(row) * 64, target, valid);
        MicroAPI::StoreAlign<float>(rhs + static_cast<uint32_t>(row + 1) * 64, target1, valid);
    }
    if (pairedEnd < end)
    {
        MicroAPI::Gather(coefficient, lu + static_cast<uint32_t>(k) * colStride + pairedEnd, matrixOffset, valid);
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target, rhs + static_cast<uint32_t>(pairedEnd) * 64, 1u, valid);
        MicroAPI::MulAddDst<float>(target, negative, coefficient, valid);
        MicroAPI::StoreAlign<float>(rhs + static_cast<uint32_t>(pairedEnd) * 64, target, valid);
    }
}

__simd_callee__ inline void VectorInverseInitializeRhs(
    __ubuf__ int32_t *permutation, __ubuf__ float *rhs, uint16_t n, uint16_t group, bool identityPivots,
    MicroAPI::RegTensor<uint32_t> &matrixId, MicroAPI::RegTensor<uint32_t> &globalColumn, MicroAPI::MaskReg &valid,
    MicroAPI::RegTensor<int32_t> &label, MicroAPI::RegTensor<float> &one, MicroAPI::RegTensor<float> &zero,
    MicroAPI::RegTensor<float> &target, MicroAPI::MaskReg &isOne)
{
    MicroAPI::Duplicate<float>(one, 1.0f, valid);
    MicroAPI::Duplicate<float>(zero, 0.0f, valid);
    for (uint16_t row = 0; row < n; ++row)
    {
        if (identityPivots)
        {
            MicroAPI::Duplicate<int32_t>(label, static_cast<int32_t>(row), valid);
        }
        else
        {
            MicroAPI::Gather(label, permutation + static_cast<uint32_t>(row) * group, matrixId, valid);
        }
        MicroAPI::Compare<int32_t, CMPMODE::EQ>(isOne, label,
                                                reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(globalColumn), valid);
        MicroAPI::Select<float>(target, one, zero, isOne);
        MicroAPI::StoreAlign<float>(rhs + static_cast<uint32_t>(row) * 64, target, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}

template <bool upper>
__simd_callee__ inline void VectorInverseSolveRhs(
    __ubuf__ float *lu, __ubuf__ float *reciprocals, __ubuf__ float *rhs, uint16_t n, uint16_t colStride,
    uint16_t group, MicroAPI::RegTensor<uint32_t> &matrixId, MicroAPI::RegTensor<uint32_t> &matrixOffset,
    MicroAPI::MaskReg &valid, MicroAPI::RegTensor<float> &source, MicroAPI::RegTensor<float> &negative,
    MicroAPI::RegTensor<float> &coefficient, MicroAPI::RegTensor<float> &target,
    MicroAPI::RegTensor<float> &coefficient1, MicroAPI::RegTensor<float> &target1)
{
    for (uint16_t step = 0; step < n; ++step)
    {
        uint16_t k = upper ? static_cast<uint16_t>(n - 1 - step) : step;
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(source, rhs + static_cast<uint32_t>(k) * 64,
                                                                            1u, valid);
        if constexpr (upper)
        {
            MicroAPI::Gather(coefficient, reciprocals + static_cast<uint32_t>(k) * group, matrixId, valid);
            MicroAPI::Mul<float>(source, source, coefficient, valid);
            MicroAPI::StoreAlign<float>(rhs + static_cast<uint32_t>(k) * 64, source, valid);
        }
        MicroAPI::Neg<float>(negative, source, valid);
        uint16_t begin = upper ? 0 : static_cast<uint16_t>(k + 1);
        uint16_t end = upper ? k : n;
        VectorInverseUpdateRows(lu, rhs, k, begin, end, colStride, matrixOffset, valid, negative, coefficient, target,
                                coefficient1, target1);
        MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
    }
}

__no_simd_vf_fusion__ __simd_vf__ __attribute__((noinline)) void VectorInverseRhsVF(
    __ubuf__ float *lu, __ubuf__ int32_t *permutation, __ubuf__ float *reciprocals, __ubuf__ float *rhs,
    __ubuf__ float *stage, uint16_t n, uint16_t colStride, uint16_t group, uint16_t rhsCols, uint16_t rhsLog,
    uint16_t firstCol, uint32_t matrixStride, uint32_t count, bool identityPivots)
{
    MicroAPI::RegTensor<uint32_t> lane, matrixId, rhsId, matrixOffset, index, globalColumn;
    MicroAPI::RegTensor<int32_t> label;
    MicroAPI::RegTensor<float> one, zero, source, negative, coefficient, target, coefficient1, target1;
    MicroAPI::MaskReg isOne;
    uint32_t lanes = count * rhsCols;
    auto valid = MicroAPI::UpdateMask<float>(lanes);
    MicroAPI::Arange(reinterpret_cast<MicroAPI::RegTensor<int32_t> &>(lane), 0);
    MicroAPI::ShiftRights(matrixId, lane, static_cast<int16_t>(rhsLog), valid);
    MicroAPI::Muls(index, matrixId, static_cast<uint32_t>(rhsCols), valid);
    MicroAPI::Sub(rhsId, lane, index, valid);
    MicroAPI::Muls(matrixOffset, matrixId, matrixStride, valid);
    MicroAPI::Adds(globalColumn, rhsId, static_cast<uint32_t>(firstCol), valid);
    VectorInverseInitializeRhs(permutation, rhs, n, group, identityPivots, matrixId, globalColumn, valid, label, one,
                               zero, target, isOne);
    VectorInverseSolveRhs<false>(lu, reciprocals, rhs, n, colStride, group, matrixId, matrixOffset, valid, source,
                                 negative, coefficient, target, coefficient1, target1);
    VectorInverseSolveRhs<true>(lu, reciprocals, rhs, n, colStride, group, matrixId, matrixOffset, valid, source,
                                negative, coefficient, target, coefficient1, target1);
    MicroAPI::Muls(index, lane, static_cast<uint32_t>(colStride), valid);
    for (uint16_t row = 0; row < n; ++row)
    {
        MicroAPI::LoadAlign<float, MicroAPI::DataCopyMode::DATA_BLOCK_COPY>(
            target, rhs + static_cast<uint32_t>(row) * 64, 1u, valid);
        MicroAPI::Scatter(stage + row, target, index, valid);
    }
    MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
}
#endif

#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
struct VectorInverseOperation
{
    TPipe pipe;
    TBuf<TPosition::VECCALC> luBuf, rhsBuf, stageBuf, pivotBuf, permutationBuf, diagonalBuf, infoBuf;
    LocalTensor<float> lu, rhs, stage, diagonal, ones;
    LocalTensor<int32_t> pivot, permutation, localInfo;
    GlobalTensor<int32_t> globalPivots, globalInfo;
    __ubuf__ float *rawLu, *rawRhs, *rawStage, *rawDiagonal, *rawReciprocals;
    __ubuf__ int32_t *rawPivot, *rawPermutation, *rawInfo;
    GM_ADDR input;
    GM_ADDR output;
    int n, lda, ldc, batch, group, rhsCols, rhsLog, colStride, matrixStride, auxiliaryElements;
    uint64_t sourceGap, destinationGap;
    bool identityPivots;

    __aicore__ inline void InitBuffers()
    {
        colStride = Align(n);
        matrixStride = ((colStride * n / 8) | 1) * 8;
        auxiliaryElements = Align(n * group);
        pipe.InitBuffer(luBuf, group * matrixStride * sizeof(float));
        pipe.InitBuffer(rhsBuf, n * 64 * sizeof(float));
        pipe.InitBuffer(stageBuf, colStride * 64 * sizeof(float));
        pipe.InitBuffer(pivotBuf, auxiliaryElements * sizeof(int32_t));
        pipe.InitBuffer(permutationBuf, auxiliaryElements * sizeof(int32_t));
        pipe.InitBuffer(diagonalBuf, auxiliaryElements * sizeof(float));
        pipe.InitBuffer(infoBuf, Align(group) * sizeof(int32_t));
        lu = luBuf.Get<float>();
        rhs = rhsBuf.Get<float>();
        stage = stageBuf.Get<float>();
        pivot = pivotBuf.Get<int32_t>();
        permutation = permutationBuf.Get<int32_t>();
        diagonal = diagonalBuf.Get<float>();
        ones = pivotBuf.Get<float>();
        localInfo = infoBuf.Get<int32_t>();
        rawLu = reinterpret_cast<__ubuf__ float *>(lu.GetPhyAddr());
        rawRhs = reinterpret_cast<__ubuf__ float *>(rhs.GetPhyAddr());
        rawStage = reinterpret_cast<__ubuf__ float *>(stage.GetPhyAddr());
        rawPivot = reinterpret_cast<__ubuf__ int32_t *>(pivot.GetPhyAddr());
        rawPermutation = reinterpret_cast<__ubuf__ int32_t *>(permutation.GetPhyAddr());
        rawDiagonal = reinterpret_cast<__ubuf__ float *>(diagonal.GetPhyAddr());
        rawReciprocals = reinterpret_cast<__ubuf__ float *>(ones.GetPhyAddr());
        rawInfo = reinterpret_cast<__ubuf__ int32_t *>(localInfo.GetPhyAddr());
    }
    __aicore__ inline void Init(GM_ADDR in, GM_ADDR out, GM_ADDR pivots, GM_ADDR info, int order, int leadingA,
                                int leadingC, int count, int matrices, int columns, int columnLog, bool identity)
    {
        input = in;
        output = out;
        n = order;
        lda = leadingA;
        ldc = leadingC;
        batch = count;
        group = matrices;
        rhsCols = columns;
        rhsLog = columnLog;
        identityPivots = identity;
        InitBuffers();
        globalPivots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(pivots));
        globalInfo.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(info));
        sourceGap = static_cast<uint64_t>(lda - n) * sizeof(float);
        destinationGap = static_cast<uint64_t>(ldc - n) * sizeof(float);
    }
    __aicore__ inline void LoadGroup(int64_t first, int count)
    {
        for (int matrix = 0; matrix < count; ++matrix)
        {
            uint64_t address =
                ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t *>(input) + first + matrix);
            GlobalTensor<float> a;
            a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
            if (sourceGap <= 0xffffffffULL)
            {
                Fence<HardEvent::V_MTE2>();
                DataCopyExtParams copy{static_cast<uint16_t>(n), static_cast<uint32_t>(n * sizeof(float)),
                                       static_cast<uint32_t>(sourceGap), 0, 0};
                DataCopyPadExtParams<float> pad{true, 0, static_cast<uint8_t>(colStride - n), 0.0f};
                DataCopyPad(lu[matrix * matrixStride], a, copy, pad);
                Fence<HardEvent::MTE2_S>();
            }
            else
            {
                for (int col = 0; col < n; ++col)
                {
                    Load(lu[matrix * matrixStride + col * colStride], a[static_cast<int64_t>(col) * lda], n);
                }
            }
        }
    }
    __aicore__ inline void PreparePermutation(int64_t first, int count)
    {
        if (!identityPivots)
        {
            Fence<HardEvent::V_MTE2>();
            int elements = count * n;
            DataCopyExtParams copy{1, static_cast<uint32_t>(elements * sizeof(int32_t)), 0, 0, 0};
            DataCopyPadExtParams<int32_t> pad{true, 0, static_cast<uint8_t>(Align(elements) - elements), 0};
            DataCopyPad(pivot, globalPivots[first * n], copy, pad);
            Fence<HardEvent::MTE2_S>();
            Fence<HardEvent::S_V>();
            VectorInversePermutationVF(rawPivot, rawPermutation, n, group, count);
        }
    }
    __aicore__ inline void PrepareDiagonal(int64_t first, int count)
    {
        Duplicate(diagonal, 1.0f, auxiliaryElements);
        PipeBarrier<PIPE_V>();
        Fence<HardEvent::S_V>();
        VectorInverseDiagonalVF(rawLu, rawDiagonal, rawInfo, n, colStride, group, matrixStride, count);
        // Pivot storage is dead after the permutation; reuse it as the IEEE division numerator.
        PipeBarrier<PIPE_V>();
        Duplicate(ones, 1.0f, auxiliaryElements);
        PipeBarrier<PIPE_V>();
        Divide(ones, ones, diagonal, auxiliaryElements);
        Fence<HardEvent::V_S>();
        Fence<HardEvent::S_MTE3>();
        DataCopyExtParams infoCopy{1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(globalInfo[first], localInfo, infoCopy);
        Fence<HardEvent::MTE3_S>();
    }
    __aicore__ inline void StoreMatrix(int64_t first, int matrix, int firstCol, int columns)
    {
        if (localInfo.GetValue(matrix) != 0)
        {
            return;
        }
        uint64_t address = ReadGmByPassDCache<uint64_t>(reinterpret_cast<__gm__ uint64_t *>(output) + first + matrix);
        GlobalTensor<float> c;
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(address));
        int stageOffset = matrix * rhsCols * colStride;
        if (destinationGap <= 0xffffffffULL)
        {
            Fence<HardEvent::S_MTE3>();
            DataCopyExtParams copy{static_cast<uint16_t>(columns), static_cast<uint32_t>(n * sizeof(float)), 0,
                                   static_cast<uint32_t>(destinationGap), 0};
            DataCopyPad(c[static_cast<int64_t>(firstCol) * ldc], stage[stageOffset], copy);
            Fence<HardEvent::MTE3_S>();
        }
        else
        {
            for (int col = 0; col < columns; ++col)
            {
                Store(c[static_cast<int64_t>(firstCol + col) * ldc], stage[stageOffset + col * colStride], n);
            }
        }
    }
    __aicore__ inline void SolveColumns(int64_t first, int count)
    {
        for (int firstCol = 0; firstCol < n; firstCol += rhsCols)
        {
            int columns = n - firstCol < rhsCols ? n - firstCol : rhsCols;
            Fence<HardEvent::S_V>();
            VectorInverseRhsVF(rawLu, rawPermutation, rawReciprocals, rawRhs, rawStage, n, colStride, group, rhsCols,
                               rhsLog, firstCol, matrixStride, count, identityPivots);
            Fence<HardEvent::V_S>();
            for (int matrix = 0; matrix < count; ++matrix)
            {
                StoreMatrix(first, matrix, firstCol, columns);
            }
        }
    }
    __aicore__ inline void Run()
    {
        if (group == 0)
        {
            return;
        }
        if (group < 0)
        {
            return;
        }
        int64_t jobs = (static_cast<int64_t>(batch) + group - 1) / group;
        for (int64_t job = GetBlockIdx(); job < jobs; job += GetBlockNum())
        {
            int64_t first = job * group;
            int count = batch - first < group ? static_cast<int>(batch - first) : group;
            LoadGroup(first, count);
            PreparePermutation(first, count);
            PrepareDiagonal(first, count);
            SolveColumns(first, count);
        }
    }
};
#endif

__global__ __aicore__ void real_lu_vector_inverse_kernel(GM_ADDR input, GM_ADDR output, GM_ADDR pivots, GM_ADDR info,
                                                         int n, int lda, int ldc, int batch, int group, int rhsCols,
                                                         int rhsLog, bool identityPivots)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if (__NPU_ARCH__ == 3510) && defined(__DAV_VEC__)
    if (n <= 0 || batch <= 0 || group <= 0 || rhsCols <= 0)
    {
        return;
    }
    VectorInverseOperation operation;
    operation.Init(input, output, pivots, info, n, lda, ldc, batch, group, rhsCols, rhsLog, identityPivots);
    operation.Run();
#endif
}

#if defined(REAL_LU_REGISTER_LAYOUT_950)
static int VectorInverseGroup(int n, int batch)
{
    int colStride = (n + 7) / 8 * 8;
    int matrixStride = ((colStride * n / 8) | 1) * 8;
    for (int group = 32; group >= 1; group /= 2)
    {
        int auxiliaryElements = (n * group + 7) / 8 * 8;
        int infoElements = (group + 7) / 8 * 8;
        int64_t bytes = (static_cast<int64_t>(group) * matrixStride + n * 64 + colStride * 64 + 3 * auxiliaryElements +
                         infoElements) *
                        sizeof(float);
        if (group <= batch && bytes <= 237568)
        {
            return group;
        }
    }
    return 1;
}
#endif

#if defined(REAL_LU_REGISTER_LAYOUT_950)
static int LaunchVectorInverse(GM_ADDR a, GM_ADDR c, GM_ADDR piv, GM_ADDR info, int n, int lda, int ldc, int batch,
                               bool identityPivots, aclrtStream stream)
{
    int group = VectorInverseGroup(n, batch);
    if (group == 0)
    {
        return -1;
    }
    if (group < 0)
    {
        return -1;
    }
    int rhsCols = 64 / group, rhsLog = 0;
    if (rhsCols <= 0)
    {
        return -1;
    }
    for (int columns = rhsCols; columns > 1; columns /= 2)
    {
        ++rhsLog;
    }
    int64_t jobs = (static_cast<int64_t>(batch) + group - 1) / group;
    int blocks = static_cast<int>(std::min<int64_t>(real_lu_vector_cores(), jobs));
    real_lu_vector_inverse_kernel<<<blocks, nullptr, stream>>>(a, c, piv, info, n, lda, ldc, batch, group, rhsCols,
                                                               rhsLog, identityPivots);
    return 0;
}
#endif

struct InverseLaunchContext
{
    GM_ADDR a;
    GM_ADDR c;
    GM_ADDR piv;
    GM_ADDR info;
    GM_ADDR workspace;
    int n, lda, ldc, batch, pitch, cores;
    int64_t span, matrixStride;
    bool arrays, identityPivots, bulk;
    aclrtStream stream;

    bool Init(GM_ADDR input, GM_ADDR output, GM_ADDR pivots, GM_ADDR result, GM_ADDR work, int order, int leadingA,
              int leadingC, int count, bool pointerArray, bool identity, aclrtStream current)
    {
        a = input;
        c = output;
        piv = pivots;
        info = result;
        workspace = work;
        n = order;
        lda = leadingA;
        ldc = leadingC;
        batch = count;
        arrays = pointerArray;
        identityPivots = identity;
        stream = current;
        pitch = (n + 63) / 64 * 64;
        cores = real_lu_vector_cores();
        if (pitch <= 0 || cores <= 0)
        {
            return false;
        }
        span = static_cast<int64_t>(pitch) * pitch;
        matrixStride = 5 * span + (pitch / 64 + 7) / 8 * 8;
        bulk = (n <= 128 || (n <= 192 && batch > 1));
        return true;
    }

    int64_t TransferJobs(bool packing) const
    {
        uint64_t gap = static_cast<uint64_t>((packing ? lda : ldc) - n) * sizeof(float);
        bool whole = bulk && gap <= 0xffffffffULL;
        bool tiled = n > 192 && batch > 1 && gap <= 0xffffffffULL;
        int stride = packing ? pitch : (n + 7) / 8 * 8;
        if (stride == 0)
        {
            return 0;
        }
        if (stride < 0)
        {
            return 0;
        }
        int tileColumns = std::min(32, 24576 / stride);
        if (tileColumns == 0)
        {
            return 0;
        }
        if (tileColumns < 0)
        {
            return 0;
        }
        int columns = packing ? pitch : n;
        int columnTiles = (columns + tileColumns - 1) / tileColumns;
        return whole ? batch : static_cast<int64_t>(batch) * (tiled ? columnTiles : columns);
    }

    int Prepare() const
    {
        int64_t jobs = TransferJobs(true);
        if (jobs <= 0)
        {
            return -1;
        }
        real_lu_pack_inverse_kernel<<<std::min<int64_t>(cores, jobs), nullptr, stream>>>(
            a, piv, workspace, n, lda, pitch, matrixStride, batch, arrays, identityPivots);
        real_lu_diagonal_inverse_kernel<<<std::min<int64_t>(cores, static_cast<int64_t>(batch) * (pitch / 64)), nullptr,
                                          stream>>>(workspace, n, pitch, matrixStride, batch);
        real_lu_inverse_info_kernel<<<std::min(cores, batch), nullptr, stream>>>(workspace, info, pitch, matrixStride,
                                                                                 batch);
        return 0;
    }

    int64_t UpdateJobs(int rows) const
    {
        if (rows <= 0)
        {
            return 0;
        }
        int stride = (rows + 7) / 8 * 8;
        if (stride == 0)
        {
            return 0;
        }
        if (stride < 0)
        {
            return 0;
        }
        int updateTileColumns = std::min(batch == 1 ? 16 : 32, std::max(1, 24576 / stride));
        if (updateTileColumns == 0)
        {
            return 0;
        }
        int updateColumnTiles = (n + updateTileColumns - 1) / updateTileColumns;
        bool updateTiled = (batch == 1 && n > 128) || (batch > 1 && n > 192);
        return bulk ? batch : static_cast<int64_t>(batch) * (updateTiled ? updateColumnTiles : n);
    }

    int SolveStep(int direction, int step) const
    {
        int first = direction == 0 ? step * 64 : pitch - (step + 1) * 64;
        int active = std::min(64, n - first);
        int64_t diagonal = (direction == 0 ? 3 : 4) * span + static_cast<int64_t>(first) * pitch + first;
        if (real_lu_gemm_do(workspace, workspace, workspace, info, pitch, n, active, active, matrixStride, batch,
                            span + first, diagonal, span + first, true, stream) != 0)
        {
            return -1;
        }
        int begin = direction == 0 ? first + 64 : 0;
        int rows = direction == 0 ? std::max(0, n - begin) : first;
        if (rows == 0)
        {
            return 0;
        }
        if (real_lu_gemm_do(workspace, workspace, workspace, info, pitch, n, rows, active, matrixStride, batch,
                            span + first, static_cast<int64_t>(first) * pitch + begin, 2 * span + begin, false,
                            stream) != 0)
        {
            return -1;
        }
        int64_t updateJobs = UpdateJobs(rows);
        real_lu_inverse_update_kernel<<<std::min<int64_t>(cores, updateJobs), nullptr, stream>>>(
            workspace, info, n, pitch, matrixStride, batch, begin, rows);
        return 0;
    }

    int Solve() const
    {
        for (int direction = 0; direction < 2; ++direction)
        {
            for (int step = 0; step < pitch / 64; ++step)
            {
                if (SolveStep(direction, step) != 0)
                {
                    return -1;
                }
            }
        }
        return 0;
    }

    int Unpack() const
    {
        int64_t jobs = TransferJobs(false);
        if (jobs <= 0)
        {
            return -1;
        }
        real_lu_unpack_inverse_kernel<<<std::min<int64_t>(cores, jobs), nullptr, stream>>>(
            workspace, c, info, n, ldc, pitch, matrixStride, batch, arrays);
        return 0;
    }
};

extern "C" int real_lu_blocked_inverse_do(GM_ADDR a, GM_ADDR c, GM_ADDR piv, GM_ADDR info, GM_ADDR workspace, int n,
                                          int lda, int ldc, int batch, bool arrays, bool identityPivots,
                                          aclrtStream stream)
{
    if (n <= 0 || batch <= 0)
    {
        return 0;
    }
#if defined(REAL_LU_REGISTER_LAYOUT_950)
    if (arrays && n >= 33 && n <= 128 && batch > 1)
    {
        return LaunchVectorInverse(a, c, piv, info, n, lda, ldc, batch, identityPivots, stream);
    }
#endif
    InverseLaunchContext operation;
    if (!operation.Init(a, c, piv, info, workspace, n, lda, ldc, batch, arrays, identityPivots, stream) ||
        operation.Prepare() != 0 || operation.Solve() != 0)
    {
        return -1;
    }
    return operation.Unpack();
}
