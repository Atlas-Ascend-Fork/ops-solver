/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline int LargeScratchPlanes(bool computeVectors)
    {
        return computeVectors ? CHEEVJ_LARGE_SCRATCH_PLANES_V : CHEEVJ_LARGE_SCRATCH_PLANES_N;
    }

    __aicore__ inline void ProcessLargeInput(GM_ADDR a, GM_ADDR w, GM_ADDR workspace, int n, int inputLda, int jobz,
                                             int uplo, bool knownDiagonal, int maxSweeps)
    {
        if (workspace == nullptr)
        {
            if (GetBlockIdx() == 0)
            {
                infoGlobal.SetValue(0, -8);
            }
            return;
        }
        const int largeInfo =
            PrepareLargeWorkspace(a, w, workspace, n, inputLda, jobz, uplo, knownDiagonal, maxSweeps);
        if (GetBlockIdx() == 0)
        {
            infoGlobal.SetValue(0, largeInfo);
        }
    }

    __aicore__ inline void ProcessLocalInput(int n, int jobz, int uplo)
    {
        InitBuffers();
        LocalTensor<float> inputLocal = inputBuf.Get<float>();
        LocalTensor<float> outLocal = outputBuf.Get<float>();
        LocalTensor<float> aReal = matrixRealBuf.Get<float>();
        LocalTensor<float> aImag = matrixImagBuf.Get<float>();
        LocalTensor<float> vReal = vectorRealBuf.Get<float>();
        LocalTensor<float> vImag = vectorImagBuf.Get<float>();
        LocalTensor<float> wLocal = wBuf.Get<float>();

        CopyInputToLocal(inputLocal, n);
        if (n <= 2)
        {
            SolveTinyHermitian(inputLocal, outLocal, wLocal, n, IsVectorMode(jobz), uplo);
            CopyEigenvaluesToGlobal(wLocal, n);
            if (IsVectorMode(jobz))
            {
                CopyTinyVectorsToGlobal(outLocal, n);
            }
            infoGlobal.SetValue(0, 0);
            return;
        }

        ReconstructHermitian(inputLocal, aReal, aImag, n, uplo);
        if (IsVectorMode(jobz))
        {
            InitIdentity(vReal, vImag, n);
        }
        const bool converged = RunJacobi(aReal, aImag, vReal, vImag, n, IsVectorMode(jobz));
        ExtractAndSort(aReal, vReal, vImag, wLocal, n, IsVectorMode(jobz));
        CopyEigenvaluesToGlobal(wLocal, n);
        if (IsVectorMode(jobz))
        {
            MergeVectorsAndCopyOut(vReal, vImag, outLocal, n);
        }
        infoGlobal.SetValue(0, converged ? 0 : 1);
    }

#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    __aicore__ inline void InitializeLargeWorkspace(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR a,
                                                    GM_ADDR workspace, int n, int inputLda, bool computeVectors,
                                                    int scratchPlanes, bool useStageBlock)
    {
        SplitCompactInterleavedToPlanar(tbufPool, a, workspace, n, inputLda, computeVectors, scratchPlanes);
        if (!useStageBlock)
        {
            MaterializeImagNeg(tbufPool, workspace, n, computeVectors, scratchPlanes);
        }
        if (computeVectors)
        {
            InitPlanarIdentity(tbufPool, workspace, n, scratchPlanes);
        }
    }

    __aicore__ inline int RunPreparedLargeWorkspace(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR a, GM_ADDR w,
                                                    GM_ADDR workspace, int n, bool computeVectors, int scratchPlanes,
                                                    bool knownDiagonal)
    {
        if (!knownDiagonal && n > CHEEVJ_LARGE_NAIVE_MAX_N)
        {
            if (GetBlockIdx() == 0)
            {
                infoGlobal.SetValue(0, 2);
            }
            pipe.Destroy();
            return 2;
        }

        int largeInfo = 2;
        if (knownDiagonal && GetBlockIdx() == 0)
        {
            CheevjPlanarWorkspace planar;
            planar.Bind(workspace, n, computeVectors, scratchPlanes);
            CheevjDiagonalFastPath diagonalFastPath;
            diagonalFastPath.Init(&tbufPool);
            diagonalFastPath.SetMatrix(a, w, planar, computeVectors);
            largeInfo = diagonalFastPath.Process() ? 0 : 2;
            infoGlobal.SetValue(0, largeInfo);
            tbufPool.Reset();
        }
        else if (!knownDiagonal && GetBlockIdx() == 0)
        {
            infoGlobal.SetValue(0, largeInfo);
        }
        CheevjWorkspaceSync();

        largeInfo = infoGlobal.GetValue(0);
        if (largeInfo == 0)
        {
            WriteCheevjPlanarOutput(tbufPool, a, w, workspace, n, computeVectors, scratchPlanes);
            pipe.Destroy();
            return 0;
        }

        CheevjPlanarWorkspace planar;
        planar.Bind(workspace, n, computeVectors, scratchPlanes);
        if (n <= CHEEVJ_LARGE_NAIVE_MAX_N)
        {
            largeInfo = RunLargeNaiveJacobi(tbufPool, a, w, planar, n, computeVectors);
        }
        pipe.Destroy();
        return largeInfo;
    }
#endif

    __aicore__ inline int PrepareLargeWorkspace(GM_ADDR a, GM_ADDR w, GM_ADDR workspace, int n, int inputLda,
                                                int jobz, int uplo, bool knownDiagonal, int maxSweeps)
    {
#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
        (void)uplo;
        const bool computeVectors = IsVectorMode(jobz);
        const int scratchPlanes = LargeScratchPlanes(computeVectors);
        TBufPool<TPosition::VECCALC, 16> tbufPool;
        pipe.InitBufPool(tbufPool, 192 * 1024);

        const bool useStageBlock = !knownDiagonal && n >= CHEEVJ_STAGE_BLOCK_JACOBI16_MIN_N;

        InitializeLargeWorkspace(tbufPool, a, workspace, n, inputLda, computeVectors, scratchPlanes, useStageBlock);

        if (useStageBlock)
        {
            CheevjPlanarWorkspace planar;
            planar.Bind(workspace, n, computeVectors, scratchPlanes);
            const int stageInfo =
                RunStageBlockJacobi16(tbufPool, a, w, workspace, planar, n, computeVectors, maxSweeps);
            pipe.Destroy();
            return stageInfo;
        }

        return RunPreparedLargeWorkspace(tbufPool, a, w, workspace, n, computeVectors, scratchPlanes,
                                         knownDiagonal);
#else
        (void)a;
        (void)w;
        (void)workspace;
        (void)n;
        (void)inputLda;
        (void)jobz;
        (void)uplo;
        (void)knownDiagonal;
        (void)maxSweeps;
        return 2;
#endif
    }

#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)
    __aicore__ inline int PlanarLargeOffset(const CheevjPlanarWorkspace &planar, int row, int col) const
    {
        return col * planar.layout.strideN + row;
    }

    __aicore__ inline void ReconstructLargeHermitianFromUplo(GM_ADDR workspace, int n, bool computeVectors,
                                                             int scratchPlanes, int uplo)
    {
        if (GetBlockIdx() == 0)
        {
            CheevjPlanarWorkspace planar;
            planar.Bind(workspace, n, computeVectors, scratchPlanes);
            const bool lowerMode = IsLowerMode(uplo);
            for (int col = 0; col < n; ++col)
            {
                const int diag = PlanarLargeOffset(planar, col, col);
                planar.aImag.SetValue(diag, 0.0f);
                for (int row = 0; row < n; ++row)
                {
                    if (row == col)
                    {
                        continue;
                    }
                    const bool fillFromConjugate = lowerMode ? (row < col) : (row > col);
                    if (!fillFromConjugate)
                    {
                        continue;
                    }
                    const int dst = PlanarLargeOffset(planar, row, col);
                    const int src = PlanarLargeOffset(planar, col, row);
                    planar.aReal.SetValue(dst, planar.aReal.GetValue(src));
                    planar.aImag.SetValue(dst, -planar.aImag.GetValue(src));
                }
            }
        }
        CheevjWorkspaceSync();
    }

    __aicore__ inline int CompactLargeOffset(int row, int col, int n) const { return (row + col * n) * 2; }

    __aicore__ inline int RunLargeNaiveJacobi(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR compactInterleaved,
                                              GM_ADDR w, CheevjPlanarWorkspace &planar, int n, bool computeVectors)
    {
        if (GetBlockIdx() != 0)
        {
            return 0;
        }

        GlobalTensor<float> compactGlobal;
        compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
        GlobalTensor<float> workW;
        workW.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));

        CheevjPairColumnUpdater updater;
        updater.Init(&tbufPool);
        updater.SetWorkspace(planar);

        int converged = 0;
        for (int sweep = 0; sweep < CHEEVJ_LARGE_NAIVE_MAX_SWEEPS; ++sweep)
        {
            if (RunLargeNaiveSweep(updater, planar, n, computeVectors) == 0)
            {
                converged = 1;
                break;
            }
        }

        if (GetBlockIdx() == 0)
        {
            WriteLargeResult(compactGlobal, workW, planar, n, computeVectors);
            infoGlobal.SetValue(0, converged ? 0 : 1);
        }
        return infoGlobal.GetValue(0);
    }

    __aicore__ inline int RunLargeNaiveSweep(CheevjPairColumnUpdater &updater, CheevjPlanarWorkspace &planar, int n,
                                             bool computeVectors)
    {
        int activeCount = 0;
        for (int p = 0; p < n - 1; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                float app = 0.0f;
                float aqq = 0.0f;
                float absPq = 0.0f;
                float t = 0.0f;
                float c = 1.0f;
                float sReal = 0.0f;
                float sImag = 0.0f;
                if (!PrepareLargePairLocal(planar, p, q, app, aqq, absPq, t, c, sReal, sImag))
                {
                    continue;
                }
                ++activeCount;
                ApplyLargePairVector(updater, planar, p, q, c, sReal, sImag, app, aqq, absPq, t);
                if (computeVectors)
                {
                    updater.RotateVectorColumnsSingleCore(p, q, c, sReal, sImag);
                }
            }
        }
        return activeCount;
    }

    __aicore__ inline int RunLargeBlockJacobi(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR compactInterleaved,
                                              GM_ADDR w, CheevjPlanarWorkspace &planar, int n, bool computeVectors)
    {
        GlobalTensor<float> compactGlobal;
        compactGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(compactInterleaved));
        GlobalTensor<float> workW;
        workW.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(w));

        CheevjBlockJacobi blockJacobi;
        blockJacobi.Init(&tbufPool);
        blockJacobi.SetWorkspace(planar, computeVectors);
        blockJacobi.SetMaxLocalSweeps(CHEEVJ_BLOCK_JACOBI_LOCAL_SWEEPS);
        (void)blockJacobi.ProcessOuterSweeps(CHEEVJ_LARGE_BLOCK_OUTER_SWEEPS);
        CheevjWorkspaceSync();

        int info = 1;
        if (GetBlockIdx() == 0)
        {
            const float offdiagMax = ComputeLargeOffdiagMaxScalar(planar, n);
            WriteLargeResult(compactGlobal, workW, planar, n, computeVectors);
            info = offdiagMax <= CHEEVJ_LARGE_CONV_TOL ? 0 : 1;
            infoGlobal.SetValue(0, info);
        }
        CheevjWorkspaceSync();
        return info;
    }

    __aicore__ inline int RunStageBlockJacobi16(TBufPool<TPosition::VECCALC, 16> &tbufPool, GM_ADDR compactInterleaved,
                                                GM_ADDR w, GM_ADDR workspace, CheevjPlanarWorkspace &planar, int n,
                                                bool computeVectors, int maxSweeps)
    {
        CheevjStageBlockJacobi16 stageJacobi;
        stageJacobi.Init(&tbufPool);
        stageJacobi.SetWorkspace(planar);
        stageJacobi.SetComputeVectors(computeVectors);
        const bool stageSupported = stageJacobi.IsSupportedJobzNPath();
        if (stageSupported)
        {
            const int outerSweeps = maxSweeps > 0 ? maxSweeps : CHEEVJ_STAGE_BLOCK16_OUTER_SWEEPS;
            (void)stageJacobi.ProcessOuterSweeps(outerSweeps);
        }
        CheevjWorkspaceSync();

        if (!stageSupported)
        {
            if (GetBlockIdx() == 0)
            {
                infoGlobal.SetValue(0, 2);
            }
            CheevjWorkspaceSync();
            return 2;
        }

        int info = 1;
        if (GetBlockIdx() == 0)
        {
            tbufPool.Reset();
            const float offdiagMaxSq = ComputeLargeOffdiagMaxSqTiled(tbufPool, planar, n);
            const float convTolSq = CHEEVJ_LARGE_CONV_TOL * CHEEVJ_LARGE_CONV_TOL;
            info = offdiagMaxSq <= convTolSq ? 0 : 1;
            infoGlobal.SetValue(0, info);
        }
        CheevjWorkspaceSync();

        // Every AIV must take the same path through the output writer because it
        // contains global workspace barriers. Reading block0's scalar GM info on
        // every core can observe a stale cache line and split the control flow
        // when the fixed-sweep solve does not converge. Produce the output
        // unconditionally; the host ignores it when block0 reports info != 0.
        tbufPool.Reset();
        WriteCheevjPlanarOutput(tbufPool, compactInterleaved, w, workspace, n, computeVectors,
                                LargeScratchPlanes(computeVectors));
        return GetBlockIdx() == 0 ? info : 0;
    }

    __aicore__ inline float ComputeLargeOffdiagMaxScalar(CheevjPlanarWorkspace &planar, int n)
    {
        float maxOffdiag = 0.0f;
        for (int p = 0; p < n - 1; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const int pq = PlanarLargeOffset(planar, p, q);
                const float real = planar.aReal.GetValue(pq);
                const float imag = planar.aImag.GetValue(pq);
                maxOffdiag = MaxFloat(maxOffdiag, SqrtApprox(real * real + imag * imag));
            }
        }
        return maxOffdiag;
    }

    __aicore__ inline float ComputeLargeOffdiagMaxSqScalar(CheevjPlanarWorkspace &planar, int n)
    {
        float maxOffdiagSq = 0.0f;
        for (int p = 0; p < n - 1; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const int pq = PlanarLargeOffset(planar, p, q);
                const float real = planar.aReal.GetValue(pq);
                const float imag = planar.aImag.GetValue(pq);
                maxOffdiagSq = MaxFloat(maxOffdiagSq, real * real + imag * imag);
            }
        }
        return maxOffdiagSq;
    }
#endif
