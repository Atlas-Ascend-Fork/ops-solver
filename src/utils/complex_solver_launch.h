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

#include <stdint.h>

struct ComplexSolverShape
{
    int m;
    int n;
    int lda;
    int ldc;
    int batch;
};

void ComplexSolverFillInfo(int* info, int count, int value, uint32_t blocks, void* stream);
void ComplexSolverGetrf(ComplexSolverShape shape, void* a, void* workspace, int* pivots, int* info, uint32_t blocks,
                        void* stream);
void ComplexSolverGetri(ComplexSolverShape shape, const void* a, const int* pivots, const void* c, int* info,
                        bool batched, uint32_t blocks, void* stream);
void ComplexSolverCopyInverse(int n, int lda, const void* workspace, void* a, const int* info, uint32_t blocks,
                              void* stream);
void ComplexSolverMatinv(ComplexSolverShape shape, const void* a, const void* c, int* info, uint32_t blocks,
                         void* stream);

void ComplexSolverBlockedGetri(ComplexSolverShape shape, const void* a, const int* pivots, const void* c, int* info,
                               bool batched, uint32_t blocks, void* stream);

constexpr int COMPLEX_SOLVER_CUBE_ROWS = 128;
constexpr int COMPLEX_SOLVER_CUBE_COLUMNS = 128;
constexpr int COMPLEX_SOLVER_UPDATE_ROWS = 64;
constexpr int COMPLEX_SOLVER_UPDATE_COLUMNS = 32;
constexpr int COMPLEX_SOLVER_PANEL = 32;
constexpr int COMPLEX_SOLVER_INVERSE_PANEL = 64;
constexpr int COMPLEX_SOLVER_INVERSE_COLUMNS = 64;
constexpr int COMPLEX_SOLVER_SMALL_RESIDENT_COLUMNS = 16;
constexpr int COMPLEX_SOLVER_DIAGONAL_RHS = 8;
constexpr int COMPLEX_SOLVER_PANEL_CAPACITY = 24 * 1024;

inline bool ComplexSolverUseDistributedPanel(int m, int n)
{
    return m > COMPLEX_SOLVER_PANEL_CAPACITY / COMPLEX_SOLVER_PANEL && n > 8;
}

// Candidate slots hold at most 64 cores; local panel buffers hold 160 aligned rows.
inline uint32_t ComplexSolverDistributedBlocks(int rows, uint32_t blocks)
{
    uint32_t used = blocks < 64 ? blocks : 64;
    if (used == 0) return 0;
    uint32_t localRows = (static_cast<uint32_t>(rows) + used - 1) / used;
    return (localRows + 7) / 8 * 8 <= 160 ? used : 0;
}

inline int ComplexSolverGetrfWorkspace(int m, int n)
{
    return ComplexSolverUseDistributedPanel(m, n) ? ((m + 7) / 8 * 8 * COMPLEX_SOLVER_PANEL + 8192) : 0;
}

void ComplexSolverDistributedPanel(ComplexSolverShape shape, void* matrix, void* workspace, void* pivots, void* info,
                                   int begin, int end, uint32_t blocks, void* stream);

inline int ComplexSolverPanelWidth(int remainingRows)
{
    int width = COMPLEX_SOLVER_PANEL;
    while (width > 1 && static_cast<uint64_t>(remainingRows) * width > COMPLEX_SOLVER_PANEL_CAPACITY) width /= 2;
    return width;
}
constexpr int COMPLEX_SOLVER_TINY_THREADS = 512;
constexpr int COMPLEX_SOLVER_RESIDENT_BATCH_GROUPS = 6;
constexpr int COMPLEX_SOLVER_GROUPED_BATCH_GROUPS = 7;
constexpr int COMPLEX_SOLVER_GROUPED_PANEL_COLUMNS = 4;
constexpr int COMPLEX_SOLVER_GROUPED_UB_BUDGET = 253952 - 32768;
constexpr int COMPLEX_SOLVER_GROUPED_DENSE_COLUMNS =
    COMPLEX_SOLVER_GROUPED_UB_BUDGET / (COMPLEX_SOLVER_GROUPED_BATCH_GROUPS * 65 * 8);
inline int ComplexSolverResidentGroups(int n)
{
    return n <= COMPLEX_SOLVER_GROUPED_DENSE_COLUMNS ? COMPLEX_SOLVER_GROUPED_BATCH_GROUPS
                                                     : COMPLEX_SOLVER_RESIDENT_BATCH_GROUPS;
}

struct ComplexSolverUpdateGeometry
{
    int firstRow;
    int lastRow;
    int firstColumn;
    int rowTiles;
    int columnTiles;
    uint64_t tasks;
};

inline uint32_t ComplexSolverUsedBlocks(uint64_t tasks, uint32_t blocks)
{
    return tasks < blocks ? static_cast<uint32_t>(tasks) : blocks;
}

template <uint32_t Threads>
inline uint32_t ComplexSolverLinearBlocks(uint64_t tasks, uint32_t blocks)
{
    static_assert(Threads > 0, "linear launch requires positive threads");
    return ComplexSolverUsedBlocks((tasks + Threads - 1) / Threads, blocks);
}

template <int Width>
inline constexpr int COMPLEX_SOLVER_GROUP_THREADS = Width >= 8 ? 512 : 256;

inline constexpr int ComplexSolverInverseThreads(int width) { return width >= 32 ? 768 : width >= 4 ? 1024 : 256; }

template <int Width, bool Direct>
inline constexpr int COMPLEX_SOLVER_BATCH_THREADS =
    Direct ? (Width >= 16 ? 512 : COMPLEX_SOLVER_GROUP_THREADS<Width>) : ComplexSolverInverseThreads(Width);

inline int ComplexSolverGroupThreads(int width)
{
    return width >= 32   ? COMPLEX_SOLVER_GROUP_THREADS<32>
           : width >= 16 ? COMPLEX_SOLVER_GROUP_THREADS<16>
           : width >= 8  ? COMPLEX_SOLVER_GROUP_THREADS<8>
                         : COMPLEX_SOLVER_GROUP_THREADS<2>;
}

inline int ComplexSolverDirectThreads(int width) { return ComplexSolverGroupThreads(width); }

inline int ComplexSolverRegisterWidth(int n)
{
    if (n <= 2) return 2;
    if (n <= 4) return 4;
    if (n <= 8) return 8;
    if (n <= 16) return 16;
    return 32;
}

constexpr uint64_t COMPLEX_SOLVER_LU_VECTOR_BYTES = 208 * 1024;
inline uint64_t ComplexSolverLuVectorBytes(int n, int rows)
{
    uint64_t order = static_cast<uint64_t>(n);
    uint64_t pitch = (order + 7) / 8 * 8;
    uint64_t plane = order * pitch, result = order * rows;
    uint64_t shared = plane > result ? plane : result;
    return (2 * shared + 2 * result + 4 * pitch + 65) * sizeof(float);
}
inline int ComplexSolverLuVectorRows(int n)
{
    return n <= 96 && ComplexSolverLuVectorBytes(n, 96) <= COMPLEX_SOLVER_LU_VECTOR_BYTES ? 96 : 0;
}

inline uint64_t ComplexSolverInverseBatchTasks(int n, int batch)
{
    int groups = n <= 2    ? COMPLEX_SOLVER_TINY_THREADS
                 : n <= 4  ? 512 / 4
                 : n <= 8  ? 256 / 8
                 : n <= 16 ? 1024 / 16
                 : n <= 24 ? 1024 / 32
                 : n <= 32 ? ComplexSolverInverseThreads(32) / 32
                           : ComplexSolverResidentGroups(n);
    return (static_cast<uint64_t>(batch) + groups - 1) / groups;
}

inline ComplexSolverUpdateGeometry ComplexSolverUpdateRegion(ComplexSolverShape shape, bool inverse, int begin, int end,
                                                             bool forward)
{
    int firstRow = inverse && !forward ? 0 : end;
    int lastRow = inverse ? (forward ? shape.n : begin) : shape.m;
    int firstColumn = inverse ? 0 : end;
    int rows = (lastRow - firstRow + COMPLEX_SOLVER_UPDATE_ROWS - 1) / COMPLEX_SOLVER_UPDATE_ROWS;
    int columns = (shape.n - firstColumn + COMPLEX_SOLVER_UPDATE_COLUMNS - 1) / COMPLEX_SOLVER_UPDATE_COLUMNS;
    return {firstRow, lastRow, firstColumn, rows, columns, static_cast<uint64_t>(rows) * columns * shape.batch};
}

inline bool ComplexSolverUseCube(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry, int begin, int end)
{
    return end - begin >= 4 && geometry.lastRow - geometry.firstRow >= 32 && shape.n - geometry.firstColumn >= 32 &&
           static_cast<uint64_t>(geometry.lastRow - geometry.firstRow) * (shape.n - geometry.firstColumn) >= 4096;
}

void ComplexSolverCubeUpdate(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry, const void* a,
                             const void* c, const int* info, bool batched, bool inverse, int begin, int end,
                             uint32_t blocks, void* stream);
