/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// CANN 9.0 native UB/L1 channel configuration.
#define ENABLE_CV_COMM_VIA_SSBUF 1
#include "kernel_operator.h"

// Device API declarations precede the implementation fragments below.
#include "complex_solver_kernel.hpp"
#include "complex_solver_launch.h"

#if __NPU_ARCH__ == 3510
namespace ComplexSolverKernel
{
template <int Depth>
struct CubePipeline
{
    static constexpr int CM = COMPLEX_SOLVER_CUBE_ROWS;
    static constexpr int CN = COMPLEX_SOLVER_CUBE_COLUMNS;
    static constexpr int CK = Depth;
    static constexpr uint32_t CUBE_THREADS = 1024;
    static_assert(CM == CN, "Transposed update uses square output tiles");
    static constexpr int HALF_VALUES = CM * CN / 2;
    static constexpr int A_BYTES = CM * CK * sizeof(float);
    static constexpr int B_BYTES = CK * CN * sizeof(float);
    static constexpr int COMPONENT_BYTES = A_BYTES + B_BYTES;
    static constexpr int C_BYTES = CM * CN * sizeof(float);
    static constexpr int RESULT_OFFSET = COMPONENT_BYTES + A_BYTES;
    static constexpr int STAGE_OFFSET = RESULT_OFFSET / sizeof(float) + 2 * HALF_VALUES;
    static constexpr int STAGE_FLOATS = CM * 64;
    static_assert(2 * A_BYTES <= 65536 && 2 * B_BYTES <= 65536, "L0A/B capacity");
    static_assert(2 * C_BYTES <= 262144, "L0C capacity");
    static_assert(RESULT_OFFSET + 2 * HALF_VALUES * sizeof(float) <= 253952, "UB capacity");
    static constexpr AscendC::FixpipeConfig TO_UB = {AscendC::CO2Layout::ROW_MAJOR, true};

    static __simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void PackCubeVf(ComplexSolverShape shape,
                                                                           ComplexSolverUpdateGeometry geometry,
                                                                           int firstRow, int firstColumn, int begin,
                                                                           int end, int component, bool rightPanel,
                                                                           int part, __ubuf__ float* local)
    {
        const __ubuf__ float* right = local + STAGE_OFFSET;
        const __ubuf__ float* left = right;
        if (rightPanel)
            for (int index = threadIdx.x; index < CN * CK / 2; index += blockDim.x)
            {
                int k = index % CK, column = index / CK + part * CN / 2;
                int address = (k / 8) * CN * 8 + column * 8 + k % 8;
                float value = firstColumn + column < shape.n && begin + k < end ? right[2 * index + component] : 0.0f;
                local[address] = value;
                if (component == 1) local[COMPONENT_BYTES / sizeof(float) + address] = -value;
            }
        else
            for (int index = threadIdx.x; index < CK * CM / 2; index += blockDim.x)
            {
                int row = index % CM, k = index / CM + part * CK / 2;
                float value = firstRow + row < geometry.lastRow && begin + k < end ? left[2 * index + component] : 0.0f;
                local[CN * CK + (row / 8) * CK * 8 + k * 8 + row % 8] = value;
            }
    }

    static __aicore__ inline __gm__ aclFloatComplex* CubeMatrix(__gm__ uint8_t* array, int batch, bool batched)
    {
        if (!batched) return reinterpret_cast<__gm__ aclFloatComplex*>(array);
        return reinterpret_cast<__gm__ aclFloatComplex*>(reinterpret_cast<__gm__ const uint64_t*>(array)[batch]);
    }

    static __aicore__ inline void CopyCubePanel(AscendC::LocalTensor<float> stage, __gm__ aclFloatComplex* source,
                                                int firstRow, int firstColumn, int rows, int columns, int leading,
                                                int capacity)
    {
        AscendC::GlobalTensor<float> gm;
        gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(source) +
                           (static_cast<uint64_t>(firstColumn) * leading + firstRow) * 2);
        AscendC::DataCopyExtParams cp{static_cast<uint16_t>(columns), static_cast<uint32_t>(rows * 8),
                                      static_cast<uint32_t>((leading - rows) * 8),
                                      static_cast<uint32_t>((capacity - (rows + 3) / 4 * 4) / 4), 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::DataCopyPad(stage, gm, cp, pad);
    }

    static __aicore__ inline void PrepareCubeInput(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry,
                                                   __gm__ uint8_t* a, __gm__ uint8_t* c, bool batched, int batch,
                                                   int firstRow, int firstColumn, int begin, int end, int component,
                                                   __ubuf__ float* storage, bool inverse)
    {
        AscendC::LocalTensor<float> stage(AscendC::TPosition::VECIN,
                                          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(storage + STAGE_OFFSET)),
                                          STAGE_FLOATS);
        auto* input = CubeMatrix(a, batch, batched);
        auto* output = CubeMatrix(c, batch, batched);
        if (inverse)
        {
            auto* swap = input;
            input = output;
            output = swap;
        }
        int columns = shape.n - firstColumn < CN ? shape.n - firstColumn : CN;
        int rows = geometry.lastRow - firstRow < CM ? geometry.lastRow - firstRow : CM;
        for (int part = 0; part < 2; ++part)
        {
            int live = columns - part * (CN / 2) < CN / 2 ? columns - part * (CN / 2) : CN / 2;
            if (live <= 0) continue;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopyCubePanel(stage, output, begin, firstColumn + part * (CN / 2), end - begin, live,
                          inverse ? shape.lda : shape.ldc, CK);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<PackCubeVf>(dim3{256, 1, 1}, shape, geometry, firstRow, firstColumn, begin, end, component,
                                    true, part, storage);
        }
        for (int part = 0; part < 2; ++part)
        {
            int count = end - begin - part * (CK / 2) < CK / 2 ? end - begin - part * (CK / 2) : CK / 2;
            if (count <= 0) count = 1;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopyCubePanel(stage, input, firstRow, begin + (part * (CK / 2) < end - begin ? part * (CK / 2) : 0), rows,
                          count, inverse ? shape.ldc : shape.lda, CM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<PackCubeVf>(dim3{256, 1, 1}, shape, geometry, firstRow, firstColumn, begin, end, component,
                                    false, part, storage);
        }
    }

    static __simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void StoreCubeVf(__ubuf__ float* stage,
                                                                            __ubuf__ const float* result, int rows,
                                                                            int columns)
    {
        for (int index = threadIdx.x; index < CM * columns; index += blockDim.x)
            if (index % CM < rows)
            {
                stage[2 * index] -= result[index];
                stage[2 * index + 1] -= result[HALF_VALUES + index];
            }
    }

    static __aicore__ inline void WriteCubeResult(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry,
                                                  __gm__ uint8_t* c, bool batched, int batch, int firstRow,
                                                  int firstColumn, int component, __ubuf__ float* storage)
    {
        int column = firstColumn + component * (CN / 2);
        int rows = geometry.lastRow - firstRow < CM ? geometry.lastRow - firstRow : CM;
        auto* output = CubeMatrix(c, batch, batched);
        AscendC::LocalTensor<float> stage(AscendC::TPosition::VECIN,
                                          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(storage + STAGE_OFFSET)),
                                          STAGE_FLOATS);
        for (int part = 0; part < CN / 2; part += 32)
        {
            int columns = shape.n - column - part < 32 ? shape.n - column - part : 32;
            if (columns <= 0) break;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            CopyCubePanel(stage, output, firstRow, column + part, rows, columns, shape.ldc, CM);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            asc_vf_call<StoreCubeVf>(dim3{256, 1, 1}, storage + STAGE_OFFSET,
                                     storage + RESULT_OFFSET / sizeof(float) + part * CM, rows, columns);
            AscendC::GlobalTensor<float> gm;
            gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output) +
                               (static_cast<uint64_t>(column + part) * shape.ldc + firstRow) * 2);
            AscendC::DataCopyExtParams cp{static_cast<uint16_t>(columns), static_cast<uint32_t>(rows * 8),
                                          static_cast<uint32_t>((CM - (rows + 3) / 4 * 4) / 4),
                                          static_cast<uint32_t>((shape.ldc - rows) * 8), 0};
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopyPad(gm, stage, cp);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        }
    }

    static __aicore__ inline void ComputeCube(uint32_t resultAddress, bool previousOutput)
    {
        AscendC::LocalTensor<float> ar(AscendC::TPosition::A1, 0, CM * CK);
        AscendC::LocalTensor<float> ai(AscendC::TPosition::A1, COMPONENT_BYTES, CM * CK);
        AscendC::LocalTensor<float> br(AscendC::TPosition::B1, A_BYTES, CK * CN);
        AscendC::LocalTensor<float> bi(AscendC::TPosition::B1, COMPONENT_BYTES + A_BYTES, CK * CN);
        AscendC::LocalTensor<float> negativeAi(AscendC::TPosition::A1, 2 * COMPONENT_BYTES, CM * CK);
        AscendC::LocalTensor<float> ar0(AscendC::TPosition::A2, 0, CM * CK);
        AscendC::LocalTensor<float> ai0(AscendC::TPosition::A2, A_BYTES, CM * CK);
        AscendC::LocalTensor<float> br0(AscendC::TPosition::B2, 0, CK * CN);
        AscendC::LocalTensor<float> bi0(AscendC::TPosition::B2, B_BYTES, CK * CN);
        AscendC::LocalTensor<float> negativeAi0 = ar0;
        AscendC::LoadData2DParamsV2 pa{0, 0, CM / 16, CK / 8, CM / 16, CM / 16, false, 0};
        AscendC::LoadData2DParamsV2 pb{0, 0, CK / 16, CN / 8, CK / 16, CN / 16, true, 0};
        AscendC::LoadData(ar0, ar, pa);
        AscendC::LoadData(ai0, ai, pa);
        AscendC::LoadData(br0, br, pb);
        AscendC::LoadData(bi0, bi, pb);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::LocalTensor<float> c0(AscendC::TPosition::CO1, 0, CM * CN);
        AscendC::LocalTensor<float> c1(AscendC::TPosition::CO1, C_BYTES, CM * CN);
        AscendC::MmadParams initialize{CM, CN, CK, 0, false, true};
        AscendC::MmadParams accumulate{CM, CN, CK, 0, false, false};
        AscendC::Mmad(c0, ar0, br0, initialize);
        AscendC::Mmad(c1, ar0, bi0, initialize);
        AscendC::Mmad(c1, ai0, br0, accumulate);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::LoadData(negativeAi0, negativeAi, pa);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::Mmad(c0, negativeAi0, bi0, accumulate);
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
        if (previousOutput) AscendC::CrossCoreWaitFlag<2>(10);
        AscendC::FixpipeParamsC310<AscendC::CO2Layout::ROW_MAJOR> fix{CN, CM, CM, CN};
        fix.dualDstCtl = 1;
        AscendC::LocalTensor<float> result(AscendC::TPosition::VECIN, resultAddress, 2 * HALF_VALUES);
        AscendC::Fixpipe<float, float, TO_UB>(result, c0, fix);
        AscendC::Fixpipe<float, float, TO_UB>(result[HALF_VALUES], c1, fix);
    }
    static __aicore__ inline void QueueCubeInput(AscendC::LocalTensor<float> input, int component)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::LocalTensor<float> target(AscendC::TPosition::A1, component * COMPONENT_BYTES, CM * CK + CK * CN);
        AscendC::DataCopy(target, input, CM * CK + CK * CN);
        if (component == 1)
        {
            AscendC::LocalTensor<float> negativeAi(AscendC::TPosition::A1, 2 * COMPONENT_BYTES, CM * CK);
            AscendC::DataCopy(negativeAi, input[COMPONENT_BYTES / sizeof(float)], CM * CK);
        }
        AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(8);
    }
    static __aicore__ inline void PrefetchCubeInput(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry,
                                                    __gm__ uint8_t* a, __gm__ uint8_t* c, bool batched, uint64_t next,
                                                    uint64_t perMatrix, int begin, int end, int component,
                                                    __ubuf__ float* storage, bool inverse)
    {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        uint64_t tile = next % perMatrix;
        int row = geometry.firstRow + (tile % geometry.rowTiles) * CM;
        int column = geometry.firstColumn + (tile / geometry.rowTiles) * CN;
        PrepareCubeInput(shape, geometry, a, c, batched, static_cast<int>(next / perMatrix), row, column, begin, end,
                         component, storage, inverse);
    }

    static __aicore__ inline void ProcessVectorCubeTask(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry,
                                                        __gm__ uint8_t* a, __gm__ uint8_t* c, __gm__ int* info,
                                                        bool batched, bool inverse, uint64_t task, uint64_t perMatrix,
                                                        int batch, int firstRow, int firstColumn, int begin, int end,
                                                        __ubuf__ float* storage, bool& prefetched, bool& queued)
    {
        auto* resultBuffer = storage + RESULT_OFFSET / sizeof(float);

        int component = AscendC::GetSubBlockIdx();
        AscendC::LocalTensor<float> input(AscendC::TPosition::VECIN,
                                          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(storage)),
                                          2 * CM * CK + CK * CN);
        if (!prefetched)
            PrepareCubeInput(shape, geometry, a, c, batched, batch, firstRow, firstColumn, begin, end, component,
                             storage, inverse);
        if (!queued) QueueCubeInput(input, component);
        uint64_t next = task + AscendC::GetBlockNum();
        while (next < geometry.tasks && inverse && info[next / perMatrix] != 0) next += AscendC::GetBlockNum();
        prefetched = next < geometry.tasks;
        if (prefetched)
            PrefetchCubeInput(shape, geometry, a, c, batched, next, perMatrix, begin, end, component, storage, inverse);
        AscendC::CrossCoreWaitFlag<2>(9);
        queued = prefetched;
        if (queued) QueueCubeInput(input, component);
        WriteCubeResult(shape, geometry, c, batched, batch, firstRow, firstColumn, component, storage);
        AscendC::CrossCoreSetFlag<2, PIPE_V>(10);
    }
};
}  // namespace ComplexSolverKernel
#endif

template <int Depth>
__global__ __mix__(1, 2) void complex_cube_update(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry,
                                                  __gm__ uint8_t* a, __gm__ uint8_t* c, __gm__ int* info, bool batched,
                                                  bool inverse, int begin, int end)
{
#if __NPU_ARCH__ == 3510
    using Pipeline = ComplexSolverKernel::CubePipeline<Depth>;
    AscendC::InitSocState();
    if ASCEND_IS_AIC
    {
        AscendC::SetHF32Mode(AscendC::HF32Mode::DISABLE);
    }
    // One declared region gives AIC and AIV the same compiler-managed UB layout.
    alignas(32) __ubuf__ float storage[Pipeline::STAGE_OFFSET + Pipeline::STAGE_FLOATS];
    auto* resultBuffer = storage + Pipeline::RESULT_OFFSET / sizeof(float);
    int core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV
    {
        core /= 2;
    }
    bool prefetched = false;
    bool queued = false;
    bool previousOutput = false;
    uint64_t perMatrix = static_cast<uint64_t>(geometry.rowTiles) * geometry.columnTiles;
    for (uint64_t task = core; task < geometry.tasks; task += AscendC::GetBlockNum())
    {
        int batch = task / perMatrix;
        if (inverse && info[batch] != 0) continue;
        uint64_t tile = task % perMatrix;
        int firstRow = geometry.firstRow + (tile % geometry.rowTiles) * COMPLEX_SOLVER_CUBE_ROWS;
        int firstColumn = geometry.firstColumn + (tile / geometry.rowTiles) * COMPLEX_SOLVER_CUBE_COLUMNS;
        if ASCEND_IS_AIV
        {
            Pipeline::ProcessVectorCubeTask(shape, geometry, a, c, info, batched, inverse, task, perMatrix, batch,
                                            firstRow, firstColumn, begin, end, storage, prefetched, queued);
        }
        if ASCEND_IS_AIC
        {
            AscendC::CrossCoreWaitFlag<2>(8);
            Pipeline::ComputeCube(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(resultBuffer)), previousOutput);
            previousOutput = true;
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(9);
        }
    }
    if ASCEND_IS_AIC
    {
        if (previousOutput) AscendC::CrossCoreWaitFlag<2>(10);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
#endif
}

void ComplexSolverCubeUpdate(ComplexSolverShape shape, ComplexSolverUpdateGeometry geometry, const void* a,
                             const void* c, const int* info, bool batched, bool inverse, int begin, int end,
                             uint32_t blocks, void* stream)
{
    geometry.rowTiles =
        (geometry.lastRow - geometry.firstRow + COMPLEX_SOLVER_CUBE_ROWS - 1) / COMPLEX_SOLVER_CUBE_ROWS;
    geometry.columnTiles =
        (shape.n - geometry.firstColumn + COMPLEX_SOLVER_CUBE_COLUMNS - 1) / COMPLEX_SOLVER_CUBE_COLUMNS;
    geometry.tasks = static_cast<uint64_t>(geometry.rowTiles) * geometry.columnTiles * shape.batch;
    if (end - begin <= COMPLEX_SOLVER_PANEL)
    {
        complex_cube_update<32><<<ComplexSolverUsedBlocks(geometry.tasks, blocks / 2), nullptr, stream>>>(
            shape, geometry, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
            reinterpret_cast<__gm__ int*>(reinterpret_cast<uintptr_t>(info)), batched, inverse, begin, end);
    }
    else
    {
        complex_cube_update<64><<<ComplexSolverUsedBlocks(geometry.tasks, blocks / 2), nullptr, stream>>>(
            shape, geometry, reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a)),
            reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(c)),
            reinterpret_cast<__gm__ int*>(reinterpret_cast<uintptr_t>(info)), batched, inverse, begin, end);
    }
}
