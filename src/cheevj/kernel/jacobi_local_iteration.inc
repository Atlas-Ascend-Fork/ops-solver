/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

    __aicore__ inline bool RunJacobi(LocalTensor<float> aReal, LocalTensor<float> aImag, LocalTensor<float> vReal,
                                     LocalTensor<float> vImag, int n, bool computeVectors)
    {
        const float tolerance = CHEEVJ_REL_TOL * EstimateNorm(aReal, aImag, n);
        for (int sweep = 0; sweep < CHEEVJ_LOCAL_ITERATION_MAX_SWEEPS; ++sweep)
        {
            float maxOffDiagonal = 0.0f;
            for (int p = 0; p < n - 1; ++p)
            {
                for (int q = p + 1; q < n; ++q)
                {
                    const Complex32 apq = GetPlanarComplex(aReal, aImag, p, q, n);
                    const float absPq = ComplexAbs(apq);
                    maxOffDiagonal = MaxFloat(maxOffDiagonal, absPq);
                    if (absPq <= tolerance || absPq <= CHEEVJ_MIN_PIVOT)
                    {
                        continue;
                    }

                    const float app = aReal.GetValue(PlanarOffset(p, p, n));
                    const float aqq = aReal.GetValue(PlanarOffset(q, q, n));
                    const float tau = (aqq - app) / (2.0f * absPq);
                    const float signTau = tau >= 0.0f ? 1.0f : -1.0f;
                    const float absTau = AbsFloat(tau);
                    const float t = signTau / (absTau + SqrtApprox(1.0f + tau * tau));
                    const float c = 1.0f / SqrtApprox(1.0f + t * t);
                    const float scale = absPq > 0.0f ? t * c / absPq : 0.0f;
                    const Complex32 s = MakeComplex(apq.real * scale, apq.imag * scale);

                    ApplyMatrixRotation(aReal, aImag, n, p, q, absPq, t, c, s);
                    if (computeVectors)
                    {
                        ApplyVectorRotation(vReal, vImag, n, p, q, c, s);
                    }
                }
            }
#define CHEEVJ_JACOBI_COMMON_SECTION 1
#include "cheevj_jacobi_common.inc"
#undef CHEEVJ_JACOBI_COMMON_SECTION
        }
        return false;
    }

    __aicore__ inline void SwapVectorColumns(LocalTensor<float> vReal, LocalTensor<float> vImag, int n, int colA,
                                             int colB)
    {
        for (int row = 0; row < n; ++row)
        {
            const int offsetA = PlanarOffset(row, colA, n);
            const int offsetB = PlanarOffset(row, colB, n);
            const float realA = vReal.GetValue(offsetA);
            const float imagA = vImag.GetValue(offsetA);
            vReal.SetValue(offsetA, vReal.GetValue(offsetB));
            vImag.SetValue(offsetA, vImag.GetValue(offsetB));
            vReal.SetValue(offsetB, realA);
            vImag.SetValue(offsetB, imagA);
        }
    }

    __aicore__ inline void ExtractAndSort(LocalTensor<float> aReal, LocalTensor<float> vReal, LocalTensor<float> vImag,
                                          LocalTensor<float> wLocal, int n, bool computeVectors)
    {
        for (int idx = 0; idx < n; ++idx)
        {
            wLocal.SetValue(idx, aReal.GetValue(PlanarOffset(idx, idx, n)));
        }

        for (int idx = 1; idx < n; ++idx)
        {
            const float key = wLocal.GetValue(idx);
            int insertPos = idx;
            while (insertPos > 0 && wLocal.GetValue(insertPos - 1) > key)
            {
                wLocal.SetValue(insertPos, wLocal.GetValue(insertPos - 1));
                if (computeVectors)
                {
                    SwapVectorColumns(vReal, vImag, n, insertPos, insertPos - 1);
                }
                --insertPos;
            }
            wLocal.SetValue(insertPos, key);
        }

        if (computeVectors)
        {
            OrthonormalizeRepeatedClusters(vReal, vImag, wLocal, n);
        }
    }

    __aicore__ inline void OrthonormalizeRepeatedClusters(LocalTensor<float> vReal, LocalTensor<float> vImag,
                                                          LocalTensor<float> wLocal, int n)
    {
        int start = 0;
        while (start < n)
        {
            int end = start + 1;
            while (end < n && AbsFloat(wLocal.GetValue(end) - wLocal.GetValue(start)) <= CHEEVJ_CLUSTER_TOL)
            {
                ++end;
            }
            if (end - start > 1)
            {
                OrthonormalizeCluster(vReal, vImag, n, start, end);
            }
            start = end;
        }
    }

    __aicore__ inline void OrthonormalizeCluster(LocalTensor<float> vReal, LocalTensor<float> vImag, int n, int start,
                                                 int end)
    {
        for (int col = start; col < end; ++col)
        {
            for (int prev = start; prev < col; ++prev)
            {
                Complex32 dot = ClusterDot(vReal, vImag, n, prev, col);
                ClusterAxpy(vReal, vImag, n, col, prev, dot);
            }
            NormalizeClusterColumn(vReal, vImag, n, col);
        }
    }

    __aicore__ inline Complex32 ClusterDot(LocalTensor<float> vReal, LocalTensor<float> vImag, int n, int lhsCol,
                                           int rhsCol)
    {
        float dotReal = 0.0f;
        float dotImag = 0.0f;
        for (int row = 0; row < n; ++row)
        {
            const int lhs = PlanarOffset(row, lhsCol, n);
            const int rhs = PlanarOffset(row, rhsCol, n);
            const float lhsReal = vReal.GetValue(lhs);
            const float lhsImag = vImag.GetValue(lhs);
            const float rhsReal = vReal.GetValue(rhs);
            const float rhsImag = vImag.GetValue(rhs);
            dotReal += lhsReal * rhsReal + lhsImag * rhsImag;
            dotImag += lhsReal * rhsImag - lhsImag * rhsReal;
        }
        return MakeComplex(dotReal, dotImag);
    }

    __aicore__ inline void ClusterAxpy(LocalTensor<float> vReal, LocalTensor<float> vImag, int n, int dstCol,
                                       int srcCol, Complex32 dot)
    {
        for (int row = 0; row < n; ++row)
        {
            const int src = PlanarOffset(row, srcCol, n);
            const int dst = PlanarOffset(row, dstCol, n);
            const float srcReal = vReal.GetValue(src);
            const float srcImag = vImag.GetValue(src);
            const float updateReal = srcReal * dot.real - srcImag * dot.imag;
            const float updateImag = srcReal * dot.imag + srcImag * dot.real;
            vReal.SetValue(dst, vReal.GetValue(dst) - updateReal);
            vImag.SetValue(dst, vImag.GetValue(dst) - updateImag);
        }
    }

    __aicore__ inline void NormalizeClusterColumn(LocalTensor<float> vReal, LocalTensor<float> vImag, int n, int col)
    {
        float norm2 = 0.0f;
        for (int row = 0; row < n; ++row)
        {
            const int offset = PlanarOffset(row, col, n);
            const float real = vReal.GetValue(offset);
            const float imag = vImag.GetValue(offset);
            norm2 += real * real + imag * imag;
        }
        const float invNorm = norm2 > CHEEVJ_MIN_PIVOT ? 1.0f / SqrtApprox(norm2) : 1.0f;
        for (int row = 0; row < n; ++row)
        {
            const int offset = PlanarOffset(row, col, n);
            vReal.SetValue(offset, vReal.GetValue(offset) * invNorm);
            vImag.SetValue(offset, vImag.GetValue(offset) * invNorm);
        }
    }
