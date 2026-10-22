/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    static constexpr int RowWorkers = 32;
    static constexpr int Participants = RowWorkers + 1;
    static constexpr int LocalQ = MatrixN * RowsPerWorker;
    static constexpr int GatherTile = 64;
    static constexpr int GatherShift = RowsPerWorker == 16 ? 6 : 7;
    static constexpr int BarrierInts = Participants * 8;
    static constexpr int CommandInts = 32;
    static constexpr int MaxIterations = 256;
    static constexpr float Epsilon = 1.1920928955078125e-7f;
    static constexpr float SafeMin = 1.0e-30f;
    static constexpr int RotateCommand = 1;
    static constexpr int SwapCommand = 2;
    static constexpr int StopCommand = 3;

   public:
    __aicore__ inline void Init(AscendC::TPipe *pipe, GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR eigenvalues,
                                GM_ADDR eigenvectors, GM_ADDR info, GM_ADDR commandWorkspace, GM_ADDR barrierWorkspace)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        diagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(diagonal));
        offDiagonalGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(offDiagonal));
        eigenvaluesGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvalues));
        eigenvectorsGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(eigenvectors));
        infoGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(info));
        commandGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(commandWorkspace));
        barrierGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(barrierWorkspace));

        pipe->InitBuffer(diagonalBuf, MatrixN * sizeof(float));
        pipe->InitBuffer(offDiagonalBuf, MatrixN * sizeof(float));
        pipe->InitBuffer(localQBuf, LocalQ * sizeof(float));
        pipe->InitBuffer(scratch0Buf, MatrixN * sizeof(float));
        pipe->InitBuffer(scratch1Buf, MatrixN * sizeof(float));
        pipe->InitBuffer(gatherIndexBuf, MatrixN * sizeof(uint32_t));
        pipe->InitBuffer(commandBuf, CommandInts * sizeof(int32_t));
        pipe->InitBuffer(barrierBuf, BarrierInts * sizeof(int32_t));
#else
        (void)pipe;
        (void)diagonal;
        (void)offDiagonal;
        (void)eigenvalues;
        (void)eigenvectors;
        (void)info;
        (void)commandWorkspace;
        (void)barrierWorkspace;
#endif
    }

    __aicore__ inline void Process()
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        const int block = static_cast<int>(AscendC::GetBlockIdx());
        if (AscendC::GetBlockNum() != Participants || block < 0 ||
            block >= Participants)
        {
            return;
        }

        auto barrierLocal = barrierBuf.Get<int32_t>();
        AscendC::Duplicate(barrierLocal, static_cast<int32_t>(0), BarrierInts);
        AscendC::PipeBarrier<PIPE_ALL>();

        float scale = 0.0f;
        if (block == 0)
        {
            LoadTridiagonal();
            scale = ScaleTridiagonal();
        }
        else
        {
            InitializeWorkerIdentity((block - 1) * RowsPerWorker);
        }

        // Nobody may publish or consume the first rotation until all worker
        // slabs contain identity and the controller has scaled d/e.
        CommandBarrier(barrierLocal);
        if (block == 0)
        {
            ControllerProcess(scale, barrierLocal);
        }
        else
        {
            WorkerProcess((block - 1) * RowsPerWorker, barrierLocal);
        }
#endif
    }

   private:
#define CHEEVJ_TRIDIAG_COMMON_SECTION 1
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION

#define CHEEVJ_TRIDIAG_COMMON_SECTION 5
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        AscendC::SyncAll<true>(barrierGlobal, barrierLocal, Participants);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadTridiagonal()
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        AscendC::DataCopyExtParams diagonalCopy{1, MatrixN * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> diagonalPad{false, 0, 0, 0.0f};
        AscendC::DataCopyPad(diagonal, diagonalGlobal, diagonalCopy, diagonalPad);
        AscendC::DataCopyExtParams offDiagonalCopy{1, (MatrixN - 1) * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> offDiagonalPad{true, 0, 1, 0.0f};
        AscendC::DataCopyPad(offDiagonal, offDiagonalGlobal, offDiagonalCopy, offDiagonalPad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline float ScaleTridiagonal()
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        float scale = 0.0f;
        // These are recurrence controls, not Q matrix traffic.
        for (int index = 0; index < MatrixN; ++index)
        {
            scale = Max(scale, Abs(diagonal.GetValue(index)));
        }
        for (int index = 0; index + 1 < MatrixN; ++index)
        {
            scale = Max(scale, Abs(offDiagonal.GetValue(index)));
        }
        if (scale == 0.0f)
        {
            return 0.0f;
        }
        const float inverseScale = 1.0f / scale;
        AscendC::Muls(diagonal, diagonal, inverseScale, MatrixN);
        AscendC::Muls(offDiagonal, offDiagonal, inverseScale, MatrixN);
        AscendC::PipeBarrier<PIPE_V>();
        return scale;
    }

    __aicore__ inline void InitializeWorkerIdentity(int rowBegin)
    {
        auto localQ = localQBuf.Get<float>();
        auto rowIndex = scratch0Buf.Get<int32_t>();
        auto columnValue = scratch1Buf.Get<int32_t>();

        AscendC::CreateVecIndex(rowIndex, static_cast<int32_t>(rowBegin),
                                static_cast<uint32_t>(RowsPerWorker));
        AscendC::PipeBarrier<PIPE_V>();
        for (int column = 0; column < MatrixN; ++column)
        {
            AscendC::Adds(columnValue, rowIndex, static_cast<int32_t>(-column), RowsPerWorker);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(columnValue, columnValue, columnValue, RowsPerWorker);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mins(columnValue, columnValue, static_cast<int32_t>(1), RowsPerWorker);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Muls(columnValue, columnValue, static_cast<int32_t>(-1), RowsPerWorker);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(columnValue, columnValue, static_cast<int32_t>(1), RowsPerWorker);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(localQ[column * RowsPerWorker], columnValue,
                          AscendC::RoundMode::CAST_NONE, RowsPerWorker);
            AscendC::PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline int FindSplit(int first)
    {
        auto diagonal = diagonalBuf.Get<float>();
        auto offDiagonal = offDiagonalBuf.Get<float>();
        for (int split = first; split + 1 < MatrixN; ++split)
        {
            const float edge = Abs(offDiagonal.GetValue(split));
            const float adjacent = Abs(diagonal.GetValue(split)) + Abs(diagonal.GetValue(split + 1));
            const float tolerance = Max(4.0f * Epsilon * adjacent, SafeMin);
            if (edge <= tolerance)
            {
                offDiagonal.SetValue(split, 0.0f);
                return split;
            }
        }
        return MatrixN - 1;
    }
#define CHEEVJ_TRIDIAG_FIND_SPLIT_CLOSED
#define CHEEVJ_TRIDIAG_COMMON_SECTION 4
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
#undef CHEEVJ_TRIDIAG_FIND_SPLIT_CLOSED
        AscendC::DataCopyExtParams copy{1, CommandInts * sizeof(int32_t), 0, 0, 0};
#define CHEEVJ_TRIDIAG_COMMON_SECTION 2
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        for (int first = 0; first < MatrixN; ++first)
        {
            for (int iteration = 0; iteration <= MaxIterations; ++iteration)
            {
                const int split = FindSplit(first);
                if (split == first)
                {
                    break;
                }
                if (iteration == MaxIterations)
#define CHEEVJ_TRIDIAG_COMMON_SECTION 7
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        if (Abs(denominator) < SafeMin)
        {
            denominator = CopySign(SafeMin, denominator);
#define CHEEVJ_TRIDIAG_COMMON_SECTION 3
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
            if (radius <= SafeMin)
#define CHEEVJ_TRIDIAG_COMMON_SECTION 6
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
            PublishCommand(RotateCommand, index, cosine, sine, barrierLocal);
        }
        diagonal.SetValue(first, diagonal.GetValue(first) - correction);
        offDiagonal.SetValue(first, shift);
        offDiagonal.SetValue(split, 0.0f);
    }

    __aicore__ inline void SortEigenpairs(const AscendC::LocalTensor<int32_t> &barrierLocal)
    {
        auto diagonal = diagonalBuf.Get<float>();
        for (int output = 0; output + 1 < MatrixN; ++output)
        {
            int minimumIndex = output;
            float minimum = diagonal.GetValue(output);
            for (int candidate = output + 1; candidate < MatrixN; ++candidate)
#define CHEEVJ_TRIDIAG_COMMON_SECTION 10
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
                PublishCommand(SwapCommand, packed, 0.0f, 0.0f, barrierLocal);
            }
        }
    }

    __aicore__ inline void StoreControllerResults(float scale, int convergenceInfo)
    {
        auto diagonal = diagonalBuf.Get<float>();
        if (scale != 0.0f)
        {
            AscendC::Muls(diagonal, diagonal, scale, MatrixN);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopyExtParams eigenvalueCopy{1, MatrixN * sizeof(float), 0, 0, 0};
        AscendC::DataCopyPad(eigenvaluesGlobal, diagonal, eigenvalueCopy);

        auto metadata = commandBuf.Get<int32_t>();
#define CHEEVJ_TRIDIAG_COMMON_SECTION 8
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
        PublishCommand(StopCommand, 0, 0.0f, 0.0f, barrierLocal);
    }

    __aicore__ inline void LoadCommand()
    {
        auto command = commandBuf.Get<int32_t>();
        AscendC::DataCopyExtParams copy{1, CommandInts * sizeof(int32_t), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(command, commandGlobal, copy, pad);
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void RotateWorkerColumns(int column, float cosine, float sine)
    {
        auto localQ = localQBuf.Get<float>();
        auto scratch0 = scratch0Buf.Get<float>();
        auto scratch1 = scratch1Buf.Get<float>();
        auto current = localQ[column * RowsPerWorker];
        auto next = localQ[(column + 1) * RowsPerWorker];

        AscendC::Muls(scratch0, next, cosine, RowsPerWorker);
        AscendC::Muls(scratch1, next, -sine, RowsPerWorker);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(next, current, sine, RowsPerWorker);
        AscendC::Muls(current, current, cosine, RowsPerWorker);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(next, next, scratch0, RowsPerWorker);
        AscendC::Add(current, current, scratch1, RowsPerWorker);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void SwapWorkerColumns(int lhs, int rhs)
    {
        auto localQ = localQBuf.Get<float>();
        auto scratch = scratch0Buf.Get<float>();
        auto lhsColumn = localQ[lhs * RowsPerWorker];
        auto rhsColumn = localQ[rhs * RowsPerWorker];
        AscendC::Adds(scratch, lhsColumn, 0.0f, RowsPerWorker);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(lhsColumn, rhsColumn, 0.0f, RowsPerWorker);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(rhsColumn, scratch, 0.0f, RowsPerWorker);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void PrepareGatherIndex()
    {
        auto signedIndex = gatherIndexBuf.Get<int32_t>();
        auto index = signedIndex.template ReinterpretCast<uint32_t>();
        AscendC::CreateVecIndex(signedIndex, static_cast<int32_t>(0),
                                static_cast<uint32_t>(GatherTile));
        AscendC::PipeBarrier<PIPE_V>();
        // Gather offsets follow the compile-time worker-local column width.
        AscendC::ShiftLeft(index, index, static_cast<uint32_t>(GatherShift), GatherTile);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreWorkerRows(int rowBegin)
    {
        auto localQ = localQBuf.Get<float>();
        auto row = scratch0Buf.Get<float>();
        auto index = gatherIndexBuf.Get<uint32_t>();
        PrepareGatherIndex();
        for (int localRow = 0; localRow < RowsPerWorker; ++localRow)
        {
            // The 64-lane tiled form is deliberate: long-count Gather was
            // unreliable with a whole-leaf transpose at this tile width.
            for (int column = 0; column < MatrixN; column += GatherTile)
            {
                AscendC::Gather(row[column], localQ[column * RowsPerWorker + localRow], index, 0,
                                GatherTile);
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams copy{1, MatrixN * sizeof(float), 0, 0, 0};
            AscendC::DataCopyPad(eigenvectorsGlobal[(rowBegin + localRow) * MatrixN], row, copy);
#define CHEEVJ_TRIDIAG_COMMON_SECTION 9
#include "cheevj_tridiag_common.inc"
#undef CHEEVJ_TRIDIAG_COMMON_SECTION
            if (operation == RotateCommand)
            {
                RotateWorkerColumns(index, commandFloat.GetValue(16), commandFloat.GetValue(24));
            }
            else if (operation == SwapCommand)
            {
                const int lhs = index & 0xffff;
                const int rhs = (index >> 16) & 0xffff;
                SwapWorkerColumns(lhs, rhs);
            }
            else if (operation == StopCommand)
            {
                StoreWorkerRows(rowBegin);
            }
            CommandBarrier(barrierLocal);
            running = operation != StopCommand;
        }
    }
