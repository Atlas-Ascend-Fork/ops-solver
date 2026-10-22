/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0.
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if defined(__DAV_C310_VEC__) || defined(__DAV_C220_VEC__) || defined(__DAV_310R6_VEC__)

    class LargeOffdiagMaxSqReducer
    {
       public:
        __aicore__ inline LargeOffdiagMaxSqReducer() : n(0) {}

        __aicore__ inline void Init(TBufPool<TPosition::VECCALC, 16> *pipe)
        {
            pipe->InitBuffer(realBuf, CHEEVJ_OFFDIAG_REDUCE_TILE * sizeof(float));
            pipe->InitBuffer(imagBuf, CHEEVJ_OFFDIAG_REDUCE_TILE * sizeof(float));
            pipe->InitBuffer(workBuf, CHEEVJ_OFFDIAG_REDUCE_TILE * sizeof(float));
        }

        __aicore__ inline void SetWorkspace(CheevjPlanarWorkspace &planarWorkspace, int matrixN)
        {
            planar = planarWorkspace;
            n = matrixN;
        }

        __aicore__ inline float Process()
        {
            float maxOffdiagSq = 0.0f;
            for (int col = 0; col < n; ++col)
            {
                for (int rowOffset = 0; rowOffset < n; rowOffset += CHEEVJ_OFFDIAG_REDUCE_TILE)
                {
                    const int elems = MinInt(CHEEVJ_OFFDIAG_REDUCE_TILE, n - rowOffset);
                    maxOffdiagSq = MaxFloat(maxOffdiagSq, ProcessColumnTile(col, rowOffset, elems));
                }
            }
            return maxOffdiagSq;
        }

       private:
        TBuf<TPosition::VECCALC> realBuf;
        TBuf<TPosition::VECCALC> imagBuf;
        TBuf<TPosition::VECCALC> workBuf;
        CheevjPlanarWorkspace planar;
        int n;

        __aicore__ inline float ProcessColumnTile(int col, int rowOffset, int elems)
        {
            LocalTensor<float> realLocal = realBuf.Get<float>();
            LocalTensor<float> imagLocal = imagBuf.Get<float>();
            LocalTensor<float> workLocal = workBuf.Get<float>();

            const uint32_t bytes = static_cast<uint32_t>(elems * sizeof(float));
            DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(realLocal, planar.aReal[ColumnMajorOffset(rowOffset, col)], copyParams, padParams);
            DataCopyPad(imagLocal, planar.aImag[ColumnMajorOffset(rowOffset, col)], copyParams, padParams);
            PipeBarrier<PIPE_ALL>();

            if (col >= rowOffset && col < rowOffset + elems)
            {
                const int diagLocal = col - rowOffset;
                realLocal.SetValue(diagLocal, 0.0f);
                imagLocal.SetValue(diagLocal, 0.0f);
                PipeBarrier<PIPE_ALL>();
            }

            Mul(realLocal, realLocal, realLocal, elems);
            Mul(imagLocal, imagLocal, imagLocal, elems);
            PipeBarrier<PIPE_V>();
            Add(workLocal, realLocal, imagLocal, elems);
            PipeBarrier<PIPE_V>();
            ReduceMax(workLocal, workLocal, workLocal, elems, true);
            PipeBarrier<PIPE_ALL>();
            return workLocal.GetValue(0);
        }

        __aicore__ inline int ColumnMajorOffset(int row, int col) const { return col * planar.layout.strideN + row; }
    };

    __aicore__ inline float ComputeLargeOffdiagMaxSqTiled(TBufPool<TPosition::VECCALC, 16> &tbufPool,
                                                          CheevjPlanarWorkspace &planar, int n)
    {
        LargeOffdiagMaxSqReducer reducer;
        reducer.Init(&tbufPool);
        reducer.SetWorkspace(planar, n);
        return reducer.Process();
    }

    __aicore__ inline bool PrepareLargePairLocal(CheevjPlanarWorkspace &planar, int p, int q, float &app, float &aqq,
                                                 float &absPq, float &t, float &c, float &sReal, float &sImag)
    {
        const int pp = PlanarLargeOffset(planar, p, p);
        const int qq = PlanarLargeOffset(planar, q, q);
        const int pq = PlanarLargeOffset(planar, p, q);
        app = planar.aReal.GetValue(pp);
        aqq = planar.aReal.GetValue(qq);
        const float apqReal = planar.aReal.GetValue(pq);
        const float apqImag = planar.aImag.GetValue(pq);
        absPq = SqrtApprox(apqReal * apqReal + apqImag * apqImag);
        if (absPq <= CHEEVJ_LARGE_CONV_TOL || absPq <= CHEEVJ_MIN_PIVOT)
        {
            return false;
        }

        const float tau = (aqq - app) / (2.0f * absPq);
        const float signTau = tau >= 0.0f ? 1.0f : -1.0f;
        const float absTau = AbsFloat(tau);
        t = signTau / (absTau + SqrtApprox(1.0f + tau * tau));
        c = 1.0f / SqrtApprox(1.0f + t * t);
        const float scale = absPq > 0.0f ? t * c / absPq : 0.0f;
        sReal = apqReal * scale;
        sImag = apqImag * scale;
        return true;
    }

    __aicore__ inline void ApplyLargePairScalar(CheevjPlanarWorkspace &planar, int p, int q, int n, float c,
                                                float sReal, float sImag, float app, float aqq, float absPq, float t)
    {
        for (int k = 0; k < n; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }

            const int kp = PlanarLargeOffset(planar, k, p);
            const int kq = PlanarLargeOffset(planar, k, q);
            const float pReal = planar.aReal.GetValue(kp);
            const float pImag = planar.aImag.GetValue(kp);
            const float qReal = planar.aReal.GetValue(kq);
            const float qImag = planar.aImag.GetValue(kq);

            const float newPReal = c * pReal - sReal * qReal - sImag * qImag;
            const float newPImag = c * pImag - sReal * qImag + sImag * qReal;
            const float newQReal = sReal * pReal - sImag * pImag + c * qReal;
            const float newQImag = sReal * pImag + sImag * pReal + c * qImag;

            planar.aReal.SetValue(kp, newPReal);
            planar.aImag.SetValue(kp, newPImag);
            planar.aImagNeg.SetValue(kp, -newPImag);
            planar.aReal.SetValue(kq, newQReal);
            planar.aImag.SetValue(kq, newQImag);
            planar.aImagNeg.SetValue(kq, -newQImag);

            const int pk = PlanarLargeOffset(planar, p, k);
            const int qk = PlanarLargeOffset(planar, q, k);
            planar.aReal.SetValue(pk, newPReal);
            planar.aImag.SetValue(pk, -newPImag);
            planar.aImagNeg.SetValue(pk, newPImag);
            planar.aReal.SetValue(qk, newQReal);
            planar.aImag.SetValue(qk, -newQImag);
            planar.aImagNeg.SetValue(qk, newQImag);
        }

        FinalizeLargePivot(planar, p, q, app, aqq, absPq, t);
    }

    __aicore__ inline void ApplyLargePairVector(CheevjPairColumnUpdater &updater, CheevjPlanarWorkspace &planar, int p,
                                                int q, float c, float sReal, float sImag, float app, float aqq,
                                                float absPq, float t)
    {
        updater.RotateMatrixColumnsSingleCore(p, q, c, sReal, sImag);
        updater.MirrorMatrixColumnsToRowsSingleCore(p, q);
        FinalizeLargePivot(planar, p, q, app, aqq, absPq, t);
    }

    __aicore__ inline void RepairLargeRowsScalar(CheevjPlanarWorkspace &planar, int p, int q, int n)
    {
        for (int k = 0; k < n; ++k)
        {
            if (k == p || k == q)
            {
                continue;
            }

            const int kp = PlanarLargeOffset(planar, k, p);
            const int pk = PlanarLargeOffset(planar, p, k);
            const float kpReal = planar.aReal.GetValue(kp);
            const float kpImag = planar.aImag.GetValue(kp);
            planar.aReal.SetValue(pk, kpReal);
            planar.aImag.SetValue(pk, -kpImag);
            planar.aImagNeg.SetValue(pk, kpImag);

            const int kq = PlanarLargeOffset(planar, k, q);
            const int qk = PlanarLargeOffset(planar, q, k);
            const float kqReal = planar.aReal.GetValue(kq);
            const float kqImag = planar.aImag.GetValue(kq);
            planar.aReal.SetValue(qk, kqReal);
            planar.aImag.SetValue(qk, -kqImag);
            planar.aImagNeg.SetValue(qk, kqImag);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PublishLargeControl(GlobalTensor<float> control)
    {
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(control[0]);
        DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(control[8]);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline int BuildLargeActivePairs(CheevjPlanarWorkspace &planar, int n)
    {
        int activeCount = 0;
        const int pairCapacity = (planar.layout.planeElems - 1) / 2;
        for (int p = 0; p < n - 1; ++p)
        {
            for (int q = p + 1; q < n; ++q)
            {
                const int pq = PlanarLargeOffset(planar, p, q);
                const float apqReal = planar.aReal.GetValue(pq);
                const float apqImag = planar.aImag.GetValue(pq);
                const float absPq2 = apqReal * apqReal + apqImag * apqImag;
                if (absPq2 > CHEEVJ_LARGE_CONV_TOL * CHEEVJ_LARGE_CONV_TOL && activeCount < pairCapacity)
                {
                    const int pairBase = 1 + activeCount * 2;
                    planar.scratch0.SetValue(pairBase, static_cast<float>(p));
                    planar.scratch0.SetValue(pairBase + 1, static_cast<float>(q));
                    ++activeCount;
                }
            }
        }
        return activeCount;
    }

    __aicore__ inline void PrepareLargePairControl(CheevjPlanarWorkspace &planar, GlobalTensor<float> control, int p,
                                                   int q)
    {
        const int pp = PlanarLargeOffset(planar, p, p);
        const int qq = PlanarLargeOffset(planar, q, q);
        const int pq = PlanarLargeOffset(planar, p, q);
        const float app = planar.aReal.GetValue(pp);
        const float aqq = planar.aReal.GetValue(qq);
        const float apqReal = planar.aReal.GetValue(pq);
        const float apqImag = planar.aImag.GetValue(pq);
        const float absPq = SqrtApprox(apqReal * apqReal + apqImag * apqImag);

        float doRotate = 0.0f;
        float t = 0.0f;
        float c = 1.0f;
        float sReal = 0.0f;
        float sImag = 0.0f;
        if (absPq > CHEEVJ_LARGE_CONV_TOL && absPq > CHEEVJ_MIN_PIVOT)
        {
            const float tau = (aqq - app) / (2.0f * absPq);
            const float signTau = tau >= 0.0f ? 1.0f : -1.0f;
            const float absTau = AbsFloat(tau);
            t = signTau / (absTau + SqrtApprox(1.0f + tau * tau));
            c = 1.0f / SqrtApprox(1.0f + t * t);
            const float scale = absPq > 0.0f ? t * c / absPq : 0.0f;
            sReal = apqReal * scale;
            sImag = apqImag * scale;
            doRotate = 1.0f;
        }

        control.SetValue(0, c);
        control.SetValue(1, sReal);
        control.SetValue(2, sImag);
        control.SetValue(3, app);
        control.SetValue(4, aqq);
        control.SetValue(5, absPq);
        control.SetValue(6, t);
        control.SetValue(7, doRotate);
        control.SetValue(10, static_cast<float>(p));
        control.SetValue(11, static_cast<float>(q));
    }

    __aicore__ inline void FinalizeLargePivot(CheevjPlanarWorkspace &planar, int p, int q, float app, float aqq,
                                              float absPq, float t)
    {
        const int pp = PlanarLargeOffset(planar, p, p);
        const int qq = PlanarLargeOffset(planar, q, q);
        const int pq = PlanarLargeOffset(planar, p, q);
        const int qp = PlanarLargeOffset(planar, q, p);

        planar.aReal.SetValue(pp, app - t * absPq);
        planar.aImag.SetValue(pp, 0.0f);
        planar.aImagNeg.SetValue(pp, 0.0f);
        planar.aReal.SetValue(qq, aqq + t * absPq);
        planar.aImag.SetValue(qq, 0.0f);
        planar.aImagNeg.SetValue(qq, 0.0f);
        planar.aReal.SetValue(pq, 0.0f);
        planar.aImag.SetValue(pq, 0.0f);
        planar.aImagNeg.SetValue(pq, 0.0f);
        planar.aReal.SetValue(qp, 0.0f);
        planar.aImag.SetValue(qp, 0.0f);
        planar.aImagNeg.SetValue(qp, 0.0f);
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void WriteLargeResult(GlobalTensor<float> compactGlobal, GlobalTensor<float> wOut,
                                            CheevjPlanarWorkspace &planar, int n, bool computeVectors)
    {
        for (int idx = 0; idx < n; ++idx)
        {
            wOut.SetValue(idx, planar.aReal.GetValue(PlanarLargeOffset(planar, idx, idx)));
            if (computeVectors)
            {
                planar.scratch0.SetValue(idx, static_cast<float>(idx));
            }
        }

        for (int idx = 1; idx < n; ++idx)
        {
            const float key = wOut.GetValue(idx);
            const float keyCol = computeVectors ? planar.scratch0.GetValue(idx) : 0.0f;
            int insertPos = idx;
            while (insertPos > 0 && wOut.GetValue(insertPos - 1) > key)
            {
                wOut.SetValue(insertPos, wOut.GetValue(insertPos - 1));
                if (computeVectors)
                {
                    planar.scratch0.SetValue(insertPos, planar.scratch0.GetValue(insertPos - 1));
                }
                --insertPos;
            }
            wOut.SetValue(insertPos, key);
            if (computeVectors)
            {
                planar.scratch0.SetValue(insertPos, keyCol);
            }
        }

        if (computeVectors)
        {
            WriteLargeVectorsScalarTail(compactGlobal, planar, n);
        }
    }

    __aicore__ inline void WriteLargeVectorsScalarTail(GlobalTensor<float> compactGlobal, CheevjPlanarWorkspace &planar,
                                                       int n)
    {
        for (int outCol = 0; outCol < n; ++outCol)
        {
            const int srcCol = static_cast<int>(planar.scratch0.GetValue(outCol) + 0.5f);
            for (int row = 0; row < n; ++row)
            {
                const int src = PlanarLargeOffset(planar, row, srcCol);
                const int dst = CompactLargeOffset(row, outCol, n);
                compactGlobal.SetValue(dst, planar.vReal.GetValue(src));
                compactGlobal.SetValue(dst + 1, planar.vImag.GetValue(src));
            }
        }
    }
#endif
