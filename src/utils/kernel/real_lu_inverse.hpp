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

namespace RealLuKernel
{
class Inverse
{
   public:
    __aicore__ inline Inverse(TPipe &pipe, int order, bool factor)
    {
        n = order;
        width = (32768 / n / 8) * 8;
        if (width > 128)
        {
            width = 128;
        }
        if (width > Align(n))
        {
            width = Align(n);
        }
        stride = Align(n);
        pipe.InitBuffer(rhsBuf, n * width * sizeof(float));
        pipe.InitBuffer(colBuf, stride * sizeof(float));
        pipe.InitBuffer(tmpBuf, (width > stride ? width : stride) * sizeof(float));
        pipe.InitBuffer(pivBuf, stride * sizeof(int));
        pipe.InitBuffer(luBuf, factor ? n * stride * sizeof(float) : 32);
        pipe.InitBuffer(infoBuf, 32);
        rhs = rhsBuf.Get<float>();
        col = colBuf.Get<float>();
        tmp = tmpBuf.Get<float>();
        piv = pivBuf.Get<int>();
        lu = luBuf.Get<float>();
        infoLocal = infoBuf.Get<int>();
    }
    __aicore__ inline int Tiles() const { return (n + width - 1) / width; }
    __aicore__ inline void Run(GM_ADDR input, GM_ADDR output, GM_ADDR pivot, GM_ADDR info, int lda, int ldc, int first,
                               bool factor, bool directRhs = false)
    {
        GlobalTensor<float> a, c;
        a.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(input));
        c.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(output));
        GlobalTensor<int> result;
        result.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(info));
        bool identityPivots = true;
        int state = PrepareInput(a, pivot, lda, factor, directRhs, identityPivots);
        if (first == 0)
        {
            StoreInt(result, infoLocal, state);
        }
        if (state != 0)
        {
            return;
        }
        if (n == 2)
        {
            InverseTwo(a, c, lda, ldc, factor);
            return;
        }
        int realWidth = n - first < width ? n - first : width;
        InitializeRhs(first, realWidth, directRhs, identityPivots);
        ForwardSubstitute(a, lda, realWidth, factor);
        BackSubstitute(a, lda, realWidth, factor);
        StoreRhs(c, ldc, first, realWidth);
    }

   private:
    __aicore__ inline int PrepareInput(GlobalTensor<float> a, GM_ADDR pivot, int lda, bool factor, bool directRhs,
                                       bool &identityPivots)
    {
        if (factor)
        {
            for (int j = 0; j < n; ++j)
            {
                Load(lu[j * stride], a[static_cast<int64_t>(j) * lda], n);
            }
            return FactorLocal(lu, col, n, n, stride, piv, true);
        }
        int state = PreparePivots(pivot, directRhs, identityPivots);
        for (int k = 0; k < n && state == 0; ++k)
        {
            Load(col, a[static_cast<int64_t>(k) * lda], n);
            if (col.GetValue(k) == 0)
            {
                state = k + 1;
            }
        }
        return state;
    }
    __aicore__ inline int PreparePivots(GM_ADDR pivot, bool directRhs, bool &identityPivots)
    {
        if (pivot == nullptr)
        {
            for (int k = 0; k < n; ++k)
            {
                piv.SetValue(k, k + 1);
            }
            return 0;
        }
        GlobalTensor<int> p;
        p.SetGlobalBuffer(reinterpret_cast<__gm__ int *>(pivot));
        DataCopyExtParams copy{1, static_cast<uint32_t>(n * sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int> pad{true, 0, static_cast<uint8_t>(stride - n), 0};
        DataCopyPad(piv, p, copy, pad);
        Fence<HardEvent::MTE2_S>();
        for (int k = 0; k < n; ++k)
        {
            int value = piv.GetValue(k);
            if (value < k + 1 || value > n)
            {
                return -4;
            }
            if (directRhs && value != k + 1)
            {
                identityPivots = false;
            }
        }
        return 0;
    }
    __aicore__ inline void InverseTwo(GlobalTensor<float> a, GlobalTensor<float> c, int lda, int ldc, bool factor)
    {
        if (!factor)
        {
            Load(col, a, 2);
            lu.SetValue(0, col.GetValue(0));
            lu.SetValue(1, col.GetValue(1));
            Load(col, a[lda], 2);
            lu.SetValue(2, col.GetValue(0));
            lu.SetValue(3, col.GetValue(1));
        }
        float upper = lu.GetValue(factor ? stride : 2);
        float lower = lu.GetValue(1);
        tmp.SetValue(0, lu.GetValue(0));
        tmp.SetValue(1, lu.GetValue(factor ? stride + 1 : 3));
        Fence<HardEvent::S_V>();
        Duplicate(rhs, 1.0f, 8);
        PipeBarrier<PIPE_V>();
        Divide(rhs, rhs, tmp, 2);
        Fence<HardEvent::V_S>();
        float a00 = rhs.GetValue(0), a11 = rhs.GetValue(1);
        float a01 = (upper * a00) * (-a11), a10 = -(a11 * lower);
        a00 = a00 - a01 * lower;
        if (piv.GetValue(0) == 2)
        {
            tmp.SetValue(0, a01);
            tmp.SetValue(1, a11);
            Store(c, tmp, 2);
            tmp.SetValue(0, a00);
            tmp.SetValue(1, a10);
            Store(c[ldc], tmp, 2);
        }
        else
        {
            tmp.SetValue(0, a00);
            tmp.SetValue(1, a10);
            Store(c, tmp, 2);
            tmp.SetValue(0, a01);
            tmp.SetValue(1, a11);
            Store(c[ldc], tmp, 2);
        }
    }
    __aicore__ inline void PermutedRhs(int first, int realWidth)
    {
        // Forward row swaps of I only permute its unit entries. Build
        // their integer labels in the existing division scratch instead
        // of moving every RHS column for each nontrivial pivot.
        auto rowMap = tmp.ReinterpretCast<int>();
        for (int r = 0; r < n; ++r)
        {
            rowMap.SetValue(r, r);
        }
        for (int k = 0; k < n; ++k)
        {
            int p = piv.GetValue(k) - 1;
            if (p != k)
            {
                int value = rowMap.GetValue(k);
                rowMap.SetValue(k, rowMap.GetValue(p));
                rowMap.SetValue(p, value);
            }
        }
        for (int r = 0; r < n; ++r)
        {
            int column = rowMap.GetValue(r) - first;
            if (column >= 0 && column < realWidth)
            {
                rhs.SetValue(r * width + column, 1.0f);
            }
        }
    }
    __aicore__ inline void SwapRhs(int realWidth)
    {
        for (int k = 0; k < n; ++k)
        {
            int p = piv.GetValue(k) - 1;
            if (p != k)
            {
                for (int j = 0; j < realWidth; ++j)
                {
                    float value = rhs.GetValue(k * width + j);
                    rhs.SetValue(k * width + j, rhs.GetValue(p * width + j));
                    rhs.SetValue(p * width + j, value);
                }
            }
        }
    }
    __aicore__ inline void InitializeRhs(int first, int realWidth, bool directRhs, bool identityPivots)
    {
        Duplicate(rhs, 0.0f, n * width);
        Fence<HardEvent::V_S>();
        if (directRhs && !identityPivots)
        {
            PermutedRhs(first, realWidth);
        }
        else
        {
            for (int j = 0; j < realWidth; ++j)
            {
                rhs.SetValue((first + j) * width + j, 1.0f);
            }
            if (!directRhs)
            {
                SwapRhs(realWidth);
            }
        }
        Fence<HardEvent::S_V>();
    }
    __aicore__ inline void ForwardSubstitute(GlobalTensor<float> a, int lda, int realWidth, bool factor)
    {
        for (int k = 0; k < n; ++k)
        {
            if (factor)
            {
                Fence<HardEvent::V_S>();
            }
            else
            {
                Load(col, a[static_cast<int64_t>(k) * lda], n);
            }
            auto coefficients = factor ? lu[k * stride] : col;
            for (int r = k + 1; r < n; ++r)
            {
                float value = coefficients.GetValue(r);
                Axpy(rhs[r * width], rhs[k * width], -value, realWidth);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void BackSubstitute(GlobalTensor<float> a, int lda, int realWidth, bool factor)
    {
        for (int k = n - 1; k >= 0; --k)
        {
            if (!factor)
            {
                Load(col, a[static_cast<int64_t>(k) * lda], n);
            }
            auto coefficients = factor ? lu[k * stride] : col;
            float diagonal = coefficients.GetValue(k);
            Duplicate(tmp, diagonal, width);
            PipeBarrier<PIPE_V>();
            Divide(rhs[k * width], rhs[k * width], tmp, realWidth);
            PipeBarrier<PIPE_V>();
            for (int r = 0; r < k; ++r)
            {
                Axpy(rhs[r * width], rhs[k * width], -coefficients.GetValue(r), realWidth);
            }
            PipeBarrier<PIPE_V>();
        }
    }
    __aicore__ inline void StoreRhs(GlobalTensor<float> c, int ldc, int first, int realWidth)
    {
        Fence<HardEvent::V_S>();
        for (int j = 0; j < realWidth; ++j)
        {
            for (int r = 0; r < n; ++r)
            {
                tmp.SetValue(r, rhs.GetValue(r * width + j));
            }
            Store(c[static_cast<int64_t>(first + j) * ldc], tmp, n);
        }
    }
    int n, width, stride;
    TBuf<TPosition::VECCALC> rhsBuf, colBuf, tmpBuf, pivBuf, luBuf, infoBuf;
    LocalTensor<float> rhs, col, tmp, lu;
    LocalTensor<int> piv, infoLocal;
};
}  // namespace RealLuKernel
