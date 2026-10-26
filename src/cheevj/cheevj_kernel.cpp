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
 * \file cheevj_kernel.cpp
 * \brief Device entry for the complex Hermitian Jacobi eigensolver.
 */

#include <cstdint>

#include "cheevj_launchers.hpp"
#include "kernel/cheevj_e2e_fixed.hpp"
#include "kernel/cheevj_her2k.hpp"
#include "kernel/cheevj_householder_fixed.hpp"
#include "kernel/cheevj_jobzv_assembly.hpp"
#include "kernel/cheevj_tridiag_vectors.hpp"
#include "kernel_operator.h"

#define CHEEVJ_ENABLE_MIXED_AIV_WORKER_COUNT
#define CHEEVJ_ENABLE_STAGE_PANEL_UPDATE
#include "kernel/jacobi.hpp"

// kernel 编译单元不可 include utils/gm_addr.h：CCE tikcfw 内置
// #define GM_ADDR __gm__ uint8_t*（带地址空间标注），与公共头的
// plain define 冲突（expected unqualified-id）。kernel 侧各文件本地
// 定义为有意为之，勿收敛到公共头；host 侧才使用 gm_addr.h。
#ifndef GM_ADDR
#define GM_ADDR uint8_t*
#endif

namespace
{
constexpr uint32_t CHEEVJ_TILING_N = 0;
constexpr uint32_t CHEEVJ_TILING_LDA = 1;
constexpr uint32_t CHEEVJ_TILING_JOBZ = 2;
constexpr uint32_t CHEEVJ_TILING_UPLO = 3;
constexpr uint32_t CHEEVJ_TILING_IS_DIAGONAL = 4;
constexpr uint32_t CHEEVJ_TILING_MAX_SWEEPS = 10;

#ifdef __DAV_C220_CUBE__
__aicore__ inline void RunLargeCubeStage(Cheevj::CheevjStageBlockJacobi16* panelService,
                                         Cheevj::CheevjStageBlockPairSchedule16* schedule, CMatmulCustom<float>* opmm,
                                         int stage, int panelSlotIndex, int panelSlotCount,
                                         bool useDirectSegmentedRight, bool computeVectors)
{
    int pairOrdinal = 0;
    if (useDirectSegmentedRight)
    {
        (void)panelService->ProcessSinglePairDirectRightPanelCube(stage, panelSlotIndex, opmm);
    }
    else
    {
        for (int pairIndex = panelSlotIndex; pairIndex < schedule->PairCountPerStage();
             pairIndex += panelSlotCount, ++pairOrdinal)
        {
            (void)panelService->ProcessSinglePairRightPanelCubeBuffered(stage, pairIndex, panelSlotIndex,
                                                                        pairOrdinal & 1, opmm);
        }
        if (computeVectors)
        {
            pairOrdinal = 0;
            for (int pairIndex = panelSlotIndex; pairIndex < schedule->PairCountPerStage();
                 pairIndex += panelSlotCount, ++pairOrdinal)
            {
                (void)panelService->ProcessSinglePairRightPanelCubeBuffered(stage, pairIndex, panelSlotIndex,
                                                                            pairOrdinal & 1, opmm);
            }
        }
    }
    if (panelSlotIndex < panelSlotCount)
    {
        pairOrdinal = 0;
        for (int pairIndex = panelSlotIndex; pairIndex < schedule->PairCountPerStage();
             pairIndex += panelSlotCount, ++pairOrdinal)
        {
            (void)panelService->ProcessSinglePairLeftPanelCubeBuffered(stage, pairIndex, panelSlotIndex,
                                                                       pairOrdinal & 1, opmm);
        }
    }
}

__aicore__ inline void RunLargeCubeJacobi(GM_ADDR workspace, int n, int jobz, int maxSweeps)
{
    const bool computeVectors = Cheevj::IsVectorMode(jobz);
    const int scratchPlanes =
        computeVectors ? Cheevj::CHEEVJ_LARGE_SCRATCH_PLANES_V : Cheevj::CHEEVJ_LARGE_SCRATCH_PLANES_N;
    Cheevj::CheevjPlanarWorkspace planar;
    planar.Bind(workspace, n, computeVectors, scratchPlanes);

    Cheevj::CheevjStageBlockJacobi16 panelService;
    panelService.SetWorkspace(planar);
    panelService.SetComputeVectors(computeVectors);
    Cheevj::CheevjStageBlockPairSchedule16 schedule;
    schedule.Reset(n, computeVectors);
    const int panelSlotCount = panelService.PanelServiceSlotCount();
    const bool useDirectSegmentedRight = panelService.CanUseDirectSegmentedRightPanel();
    const int directSlotCount = useDirectSegmentedRight ? schedule.PairCountPerStage() : 0;
    const int cubeServiceSlotCount = panelSlotCount > directSlotCount ? panelSlotCount : directSlotCount;
    const int panelSlotIndex = GetBlockIdx();
    if (cubeServiceSlotCount <= 0 || panelSlotIndex >= cubeServiceSlotCount)
    {
        return;
    }

    TPipe cubePipe;
    CMatmulCustom<float> opmm;
    opmm.Init(&cubePipe);
    if (useDirectSegmentedRight)
    {
        opmm.InitSegmentedEvents();
    }
    const int outerSweeps = maxSweeps > 0 ? maxSweeps : Cheevj::CHEEVJ_STAGE_BLOCK16_OUTER_SWEEPS;
    for (int sweep = 0; sweep < outerSweeps; ++sweep)
    {
        for (int stage = 0; stage < schedule.StageCount(); ++stage)
        {
            RunLargeCubeStage(&panelService, &schedule, &opmm, stage, panelSlotIndex, panelSlotCount,
                              useDirectSegmentedRight, computeVectors);
        }
    }
    cubePipe.Destroy();
}
#endif
}  // namespace

__global__ __aicore__ void cheevj_kernel(GM_ADDR sync, GM_ADDR a, GM_ADDR w, GM_ADDR info, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    SetSyncBaseAddr((unsigned long)sync);

    auto tilingData = reinterpret_cast<__gm__ uint32_t*>(tiling);
    const int n = static_cast<int>(tilingData[CHEEVJ_TILING_N]);
    const int jobz = static_cast<int>(tilingData[CHEEVJ_TILING_JOBZ]);
    const bool isDiagonal = tilingData[CHEEVJ_TILING_IS_DIAGONAL] != 0;
    const int maxSweeps = static_cast<int>(tilingData[CHEEVJ_TILING_MAX_SWEEPS]);

#ifdef __DAV_C220_CUBE__
    if (isDiagonal || n <= Cheevj::CHEEVJ_DEVICE_MAX_N)
    {
        return;
    }

    RunLargeCubeJacobi(workspace, n, jobz, maxSweeps);
    return;
#else
    const int lda = static_cast<int>(tilingData[CHEEVJ_TILING_LDA]);
    const int uplo = static_cast<int>(tilingData[CHEEVJ_TILING_UPLO]);

    Cheevj::CheevjDeviceKernel op;
    op.Process(a, w, info, workspace, n, lda, jobz, uplo, isDiagonal, maxSweeps);
#endif
}

void cheevj_kernel_do(GM_ADDR sync, GM_ADDR a, GM_ADDR w, GM_ADDR info, GM_ADDR workspace, GM_ADDR tiling,
                      uint32_t numBlocks, void* stream)
{
    cheevj_kernel<<<numBlocks, nullptr, stream>>>(sync, a, w, info, workspace, tiling);
}

namespace
{

constexpr int kN512 = 512;
constexpr int kN1024 = 1024;
constexpr int kN2048 = 2048;
constexpr int kN512PanelSize = 4;
constexpr int kDefaultPanelSize = 8;
constexpr int kN2048PanelSize = 16;
constexpr uint32_t kPrepareBlocks = 32;
constexpr uint32_t kCubeBlocks = 16;
constexpr uint32_t kEpilogueBlocks = 48;
constexpr uint32_t kTridiagBlocks = 33;

bool IsFixedShape(int n) { return n == kN512 || n == kN1024 || n == kN2048; }

}  // namespace

__global__ __aicore__ void cheevj_fixed_prepare_kernel(GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR packedReal,
                                                       GM_ADDR packedImag, GM_ADDR panelVReal, GM_ADDR panelVImag,
                                                       GM_ADDR panelWReal, GM_ADDR panelWImag, GM_ADDR wHReal,
                                                       GM_ADDR wHImag, GM_ADDR wHImagNeg, GM_ADDR updateReal,
                                                       GM_ADDR updateImag, GM_ADDR panelWorkspace,
                                                       GM_ADDR barrierWorkspace, int panelStart, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        Cheevj::CheevjE2EPreparePacked<kN512> prepare;
        prepare.Init(&pipe, matrixReal, matrixImag, packedReal, packedImag, panelVReal, panelVImag, panelWReal,
                     panelWImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag, panelWorkspace, barrierWorkspace,
                     panelStart);
        prepare.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjE2EPreparePacked<kN1024> prepare;
        prepare.Init(&pipe, matrixReal, matrixImag, packedReal, packedImag, panelVReal, panelVImag, panelWReal,
                     panelWImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag, panelWorkspace, barrierWorkspace,
                     panelStart);
        prepare.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjE2EPreparePacked<kN2048> prepare;
        prepare.Init(&pipe, matrixReal, matrixImag, packedReal, packedImag, panelVReal, panelVImag, panelWReal,
                     panelWImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag, panelWorkspace, barrierWorkspace,
                     panelStart);
        prepare.Process();
    }
    pipe.Destroy();
#endif
}

__global__ __aicore__ void cheevj_fixed_archive_kernel(GM_ADDR panelVReal, GM_ADDR panelVImag, GM_ADDR reflectorReal,
                                                       GM_ADDR reflectorImag, int panelStart, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        Cheevj::CheevjE2EN512ArchivePanelReflectors archive;
        archive.Init(&pipe, panelVReal, panelVImag, reflectorReal, reflectorImag, panelStart);
        archive.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjE2EN1024ArchivePanelReflectors archive;
        archive.Init(&pipe, panelVReal, panelVImag, reflectorReal, reflectorImag, panelStart);
        archive.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjE2EN2048ArchivePanelReflectors archive;
        archive.Init(&pipe, panelVReal, panelVImag, reflectorReal, reflectorImag, panelStart);
        archive.Process();
    }
    pipe.Destroy();
#endif
}

__global__ __aicore__ void cheevj_fixed_build_update_kernel(GM_ADDR panelVReal, GM_ADDR panelVImag, GM_ADDR wHReal,
                                                            GM_ADDR wHImag, GM_ADDR wHImagNeg, GM_ADDR updateReal,
                                                            GM_ADDR updateImag, int activeN, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (n == kN512)
    {
        Cheevj::CheevjHer2kBuildD<kN512>(panelVReal, panelVImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag,
                                         activeN);
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjHer2kBuildD<kN1024>(panelVReal, panelVImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag,
                                          activeN);
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjHer2kBuildD<kN2048>(panelVReal, panelVImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag,
                                          activeN);
    }
}

__global__ __aicore__ void cheevj_fixed_apply_update_kernel(GM_ADDR packedReal, GM_ADDR packedImag, GM_ADDR updateReal,
                                                            GM_ADDR updateImag, GM_ADDR panelVReal, GM_ADDR panelVImag,
                                                            GM_ADDR reflectorReal, GM_ADDR reflectorImag,
                                                            GM_ADDR matrixReal, GM_ADDR matrixImag, int panelStart,
                                                            int computeVectors, int activeN, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        if (computeVectors != 0)
        {
            Cheevj::CheevjE2EN512ArchivePanelReflectors archive;
            archive.Init(&pipe, panelVReal, panelVImag, reflectorReal, reflectorImag, panelStart,
                         kEpilogueBlocks - kN512PanelSize);
            archive.Process();
        }
        Cheevj::CheevjHer2kEpilogue<kN512> epilogue;
        epilogue.Init(&pipe, packedReal, packedImag, updateReal, updateImag, matrixReal, matrixImag, panelStart,
                      activeN);
        epilogue.Process();
    }
    else if (n == kN1024)
    {
        if (computeVectors != 0)
        {
            Cheevj::CheevjE2EN1024ArchivePanelReflectors archive;
            archive.Init(&pipe, panelVReal, panelVImag, reflectorReal, reflectorImag, panelStart,
                         kEpilogueBlocks - kDefaultPanelSize);
            archive.Process();
        }
        Cheevj::CheevjHer2kEpilogue<kN1024> epilogue;
        epilogue.Init(&pipe, packedReal, packedImag, updateReal, updateImag, matrixReal, matrixImag, panelStart,
                      activeN);
        epilogue.Process();
    }
    else if (n == kN2048)
    {
        if (computeVectors != 0)
        {
            Cheevj::CheevjE2EN2048ArchivePanelReflectors archive;
            archive.Init(&pipe, panelVReal, panelVImag, reflectorReal, reflectorImag, panelStart,
                         kEpilogueBlocks - kN2048PanelSize);
            archive.Process();
        }
        Cheevj::CheevjHer2kEpilogue<kN2048> epilogue;
        epilogue.Init(&pipe, packedReal, packedImag, updateReal, updateImag, matrixReal, matrixImag, panelStart,
                      activeN);
        epilogue.Process();
    }
    pipe.Destroy();
#endif
}

__global__ __aicore__ void cheevj_fixed_tridiag_reset_kernel(GM_ADDR commandWorkspace, GM_ADDR barrierWorkspace, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        Cheevj::CheevjTridiag512VectorsReset reset;
        reset.Init(&pipe, commandWorkspace, barrierWorkspace);
        reset.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjTridiag1024VectorsReset reset;
        reset.Init(&pipe, commandWorkspace, barrierWorkspace);
        reset.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjTridiag2048VectorsReset reset;
        reset.Init(&pipe, commandWorkspace, barrierWorkspace);
        reset.Process();
    }
    pipe.Destroy();
#endif
}

__global__ __aicore__ void cheevj_fixed_tridiag_vectors_kernel(GM_ADDR diagonal, GM_ADDR offDiagonal,
                                                               GM_ADDR eigenvalues, GM_ADDR eigenvectors, GM_ADDR info,
                                                               GM_ADDR commandWorkspace, GM_ADDR barrierWorkspace,
                                                               int rowWave, int n)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#ifdef __DAV_C220_VEC__
    AscendC::TPipe pipe;
    if (n == kN512)
    {
        Cheevj::CheevjTridiag512Vectors solver;
        solver.Init(&pipe, diagonal, offDiagonal, eigenvalues, eigenvectors, info, commandWorkspace, barrierWorkspace);
        solver.Process();
    }
    else if (n == kN1024)
    {
        Cheevj::CheevjTridiag1024Vectors solver;
        solver.Init(&pipe, diagonal, offDiagonal, eigenvalues, eigenvectors, info, commandWorkspace, barrierWorkspace);
        solver.Process();
    }
    else if (n == kN2048)
    {
        Cheevj::CheevjTridiag2048Vectors solver;
        solver.Init(&pipe, diagonal, offDiagonal, eigenvalues, eigenvectors, info, commandWorkspace, barrierWorkspace,
                    rowWave);
        solver.Process();
    }
    pipe.Destroy();
#endif
}

void cheevj_fixed_kernel_do(GM_ADDR matrixReal, GM_ADDR matrixImag, GM_ADDR packedReal, GM_ADDR packedImag,
                            GM_ADDR panelVReal, GM_ADDR panelVImag, GM_ADDR panelWReal, GM_ADDR panelWImag,
                            GM_ADDR wHReal, GM_ADDR wHImag, GM_ADDR wHImagNeg, GM_ADDR updateReal, GM_ADDR updateImag,
                            GM_ADDR diagonal, GM_ADDR offDiagonal, GM_ADDR tauReal, GM_ADDR tauImag,
                            GM_ADDR panelWorkspace, GM_ADDR barrierWorkspace, GM_ADDR reflectorReal,
                            GM_ADDR reflectorImag, GM_ADDR bounds, GM_ADDR eigenvalues, GM_ADDR lowWorkspace,
                            GM_ADDR highWorkspace, GM_ADDR tridiagonalEigenvectorsRowMajor, GM_ADDR commandWorkspace,
                            GM_ADDR eigenvectorReal, GM_ADDR eigenvectorImag, GM_ADDR info, int n, bool computeVectors,
                            void* stream)
{
    if (!IsFixedShape(n))
    {
        return;
    }

    constexpr int floatBytes = sizeof(float);
    const int panelSize = n == kN512 ? kN512PanelSize : (n == kN2048 ? kN2048PanelSize : kDefaultPanelSize);
    for (int panelStart = 0; panelStart < n; panelStart += panelSize)
    {
        cheevj_fixed_prepare_kernel<<<kPrepareBlocks, nullptr, stream>>>(
            matrixReal, matrixImag, packedReal, packedImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal,
            wHImag, wHImagNeg, updateReal, updateImag, panelWorkspace, barrierWorkspace, panelStart, n);
        cheevj_fixed_panel_do(packedReal, packedImag, panelVReal, panelVImag, panelWReal, panelWImag, wHReal, wHImag,
                              wHImagNeg, diagonal + panelStart * floatBytes, offDiagonal + panelStart * floatBytes,
                              tauReal + panelStart * floatBytes, tauImag + panelStart * floatBytes, panelWorkspace,
                              barrierWorkspace, n, n - panelStart, stream);
        if (computeVectors && n == kN512)
        {
            cheevj_fixed_archive_kernel<<<panelSize, nullptr, stream>>>(panelVReal, panelVImag, reflectorReal,
                                                                        reflectorImag, panelStart, n);
        }
        cheevj_fixed_build_update_kernel<<<kCubeBlocks, nullptr, stream>>>(
            panelVReal, panelVImag, wHReal, wHImag, wHImagNeg, updateReal, updateImag, n - panelStart, n);
        cheevj_fixed_apply_update_kernel<<<kEpilogueBlocks, nullptr, stream>>>(
            packedReal, packedImag, updateReal, updateImag, panelVReal, panelVImag, reflectorReal, reflectorImag,
            matrixReal, matrixImag, panelStart, computeVectors && n != kN512 ? 1 : 0, n - panelStart, n);
    }

    cheevj_fixed_finalize_do(diagonal, offDiagonal, tauReal, tauImag, matrixReal, matrixImag, bounds, info, n, stream);
    if (!computeVectors)
    {
        cheevj_fixed_sturm_do(diagonal, offDiagonal, bounds, eigenvalues, lowWorkspace, highWorkspace, n, stream);
        return;
    }

    cheevj_fixed_sturm_do(diagonal, offDiagonal, bounds, eigenvalues, lowWorkspace, highWorkspace, n, stream);
    cheevj_fixed_vectors_do(diagonal, offDiagonal, eigenvalues, tridiagonalEigenvectorsRowMajor, reflectorReal,
                            reflectorImag, tauReal, tauImag, eigenvectorReal, eigenvectorImag, info, n, stream);
}
