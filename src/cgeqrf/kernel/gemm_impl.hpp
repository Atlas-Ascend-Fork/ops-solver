/*
 * PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
 *
 * Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
 * All rights reserved.
 *
 * This software is dual-licensed:
 *   1) Community license: use, modify, and redistribute freely under the
 *      GNU General Public License v3 or later (GPL-3.0-or-later);
 *   2) Commercial license: for closed-source or commercial use / redistribution,
 *      contact the copyright holder for a separate license.
 * You may choose either. Default terms are GPL-3.0-or-later.
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * This program is free software: you may redistribute it and/or modify it under
 * the terms of the GPL. It is distributed "as is", without warranty.
 * See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
 * Commercial licensing contact: yuantang@fudan.edu.cn
 *
 * -------------------------------------------------------------------------
 * 中文版 (Chinese):
 *
 * PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
 *
 * 本软件采用双重许可（dual licensing）：
 *   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
 *      （GPL-3.0-or-later）自由使用、修改与再分发；
 *   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
 * 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
 *
 * 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
 * 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
 * 商业授权联系：yuantang@fudan.edu.cn
 */
/*!
 * \file gemm_impl.hpp
 * \brief CMatmulCustom 的类外成员定义（L0 搬运与 Mmad 组合）
 */

#pragma once

#include "gemm.hpp"

template <typename T>
    __aicore__ inline void CMatmulCustom<T>::SyncFixFlag()
    {
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
        CrossCoreWaitFlag(0x3);
        syncFlag = false;
    }

template <typename T>
    template <L0CWriteType writeType>
    __aicore__ inline void CMatmulCustom<T>::WriteHead()
    {
        if constexpr (writeType != L0CWriteType::REPLACE)
        {
            SetAtomicAdd<float>();
        }
        if (writeType == L0CWriteType::TAIL && syncFlag)
        {
            SyncFixFlag();
        }
    }

template <typename T>
    template <L0CWriteType writeType>
    __aicore__ inline void CMatmulCustom<T>::WriteTail()
    {
        if (writeType == L0CWriteType::FIRST && syncFlag)
        {
            SyncFixFlag();
        }
        if constexpr (writeType != L0CWriteType::REPLACE)
        {
            SetAtomicNone();
        }
    }

template <typename T>
    template <L0CWriteType writeType>
    __aicore__ inline void CMatmulCustom<T>::WriteTile(int offsetC_1, const LocalTensor<T>& c1RealLocal,
                                                       const LocalTensor<T>& c1ImagLocal)
    {
        WriteHead<writeType>();
        CopyOut<false>(c1RealLocal, offsetC_1);
        CopyOut<true>(c1ImagLocal, offsetC_1);
        WriteTail<writeType>();
    }

template <typename T>
    template <L0CWriteType writeType>
    __aicore__ inline void CMatmulCustom<T>::WriteTileReal(int offsetC_1, const LocalTensor<T>& c1RealLocal)
    {
        WriteHead<writeType>();
        CopyOut<false>(c1RealLocal, offsetC_1);
        WriteTail<writeType>();
    }

template <typename T>
    template <L0CWriteType writeType>
    __aicore__ inline void CMatmulCustom<T>::WriteTileParity(int offsetC_1, const LocalTensor<T>& c1RealLocal,
                                                             const LocalTensor<T>& c1ImagLocal)
    {
        WriteHead<writeType>();
        if (blockIdx % 2 == 0)
        {
            CopyOut<false>(c1RealLocal, offsetC_1);
        }
        else
        {
            CopyOut<true>(c1ImagLocal, offsetC_1);
        }
        WriteTail<writeType>();
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::CopyInComplex(int offsetA_1, int offsetB_1, int cnt)
    {
        CopyInAReal<trans_A, false>(offsetA_1, cnt);
        CopyInAImag<trans_A, false>(offsetA_1, cnt);
        CopyInBReal<trans_B, false>(offsetB_1, cnt);
        CopyInBImag<trans_B, 0>(offsetB_1, cnt);
        CopyInBImag<trans_B, 1>(offsetB_1, cnt);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::CopyInComplexParity(int offsetA_1, int offsetB_1, int cnt)
    {
        if (blockIdx % 2 == 0)
        {
            CopyInAReal<trans_A, false>(offsetA_1, cnt);
            CopyInAImag<trans_A, false>(offsetA_1, cnt);
            CopyInBReal<trans_B, false>(offsetB_1, cnt);
            CopyInBImag<trans_B, 1>(offsetB_1, cnt);
        }
        else
        {
            CopyInBReal<trans_B, false>(offsetB_1, cnt);
            CopyInBImag<trans_B, 0>(offsetB_1, cnt);
            CopyInAReal<trans_A, false>(offsetA_1, cnt);
            CopyInAImag<trans_A, false>(offsetA_1, cnt);
        }
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacStepRealImag(int k, int x, const LocalTensor<T>& a1RealLocal,
                                                             const LocalTensor<T>& b1RealLocal,
                                                             const LocalTensor<T>& b1ImagLocal,
                                                             const LocalTensor<T>& c1RealLocal,
                                                             const LocalTensor<T>& c1ImagLocal)
    {
        SetKInner(k, x);
        SplitA<trans_A>(a1RealLocal, x * baseK);
        SplitB<trans_B>(b1RealLocal, x * baseK);
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        Compute(a2Local, c1RealLocal, !k && !x, 0);
        SplitB<trans_B>(b1ImagLocal, x * baseK);
        Compute(a2Local, c1ImagLocal, !k && !x, 0);
        inQueueA2.FreeTensor(a2Local);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacStepImagReal(int k, int x, const LocalTensor<T>& a1ImagLocal,
                                                             const LocalTensor<T>& b1RealLocal,
                                                             const LocalTensor<T>& b1ImagLocal,
                                                             const LocalTensor<T>& c1RealLocal,
                                                             const LocalTensor<T>& c1ImagLocal)
    {
        SetKInner(k, x);
        SplitA<trans_A>(a1ImagLocal, x * baseK);
        SplitB<trans_B>(b1RealLocal, x * baseK);
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        Compute(a2Local, c1ImagLocal, 0, k + x == kBlocks - 1);
        SplitB<trans_B>(b1ImagLocal, x * baseK);
        Compute(a2Local, c1RealLocal, 0, k + x == kBlocks - 1);
        inQueueA2.FreeTensor(a2Local);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacStepParityReal(int k, int x, const LocalTensor<T>& a1RealLocal,
                                                               const LocalTensor<T>& b1RealLocal,
                                                               const LocalTensor<T>& b1ImagLocal,
                                                               const LocalTensor<T>& c1RealLocal,
                                                               const LocalTensor<T>& c1ImagLocal)
    {
        SetKInner(k, x);
        SplitA<trans_A>(a1RealLocal, x * baseK);
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        if (blockIdx % 2 == 0)
        {
            SplitB<trans_B>(b1RealLocal, x * baseK);
            Compute(a2Local, c1RealLocal, !k && !x, 0);
        }
        else
        {
            SplitB<trans_B>(b1ImagLocal, x * baseK);
            Compute(a2Local, c1ImagLocal, !k && !x, 0);
        }
        inQueueA2.FreeTensor(a2Local);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacStepParityImag(int k, int x, const LocalTensor<T>& a1ImagLocal,
                                                               const LocalTensor<T>& b1RealLocal,
                                                               const LocalTensor<T>& b1ImagLocal,
                                                               const LocalTensor<T>& c1RealLocal,
                                                               const LocalTensor<T>& c1ImagLocal)
    {
        SetKInner(k, x);
        SplitA<trans_A>(a1ImagLocal, x * baseK);
        LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
        if (blockIdx % 2 == 1)
        {
            SplitB<trans_B>(b1RealLocal, x * baseK);
            Compute(a2Local, c1ImagLocal, 0, k + x == kBlocks - 1);
        }
        else
        {
            SplitB<trans_B>(b1ImagLocal, x * baseK);
            Compute(a2Local, c1RealLocal, 0, k + x == kBlocks - 1);
        }
        inQueueA2.FreeTensor(a2Local);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacComplex(int k, int cnt, const LocalTensor<T>& c1RealLocal,
                                                        const LocalTensor<T>& c1ImagLocal)
    {
        LocalTensor<T> a1RealLocal = inQueueA1Real.DeQue<T>();
        LocalTensor<T> b1RealLocal = inQueueB1Real.DeQue<T>();
        LocalTensor<T> b1ImagLocal = inQueueB1Imag.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            MacStepRealImag<trans_A, trans_B>(k, x, a1RealLocal, b1RealLocal, b1ImagLocal, c1RealLocal,
                                              c1ImagLocal);
        }
        inQueueA1Real.FreeTensor(a1RealLocal);
        inQueueB1Imag.FreeTensor(b1ImagLocal);
        LocalTensor<T> a1ImagLocal = inQueueA1Imag.DeQue<T>();
        b1ImagLocal = inQueueB1Imag.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            MacStepImagReal<trans_A, trans_B>(k, x, a1ImagLocal, b1RealLocal, b1ImagLocal, c1RealLocal,
                                              c1ImagLocal);
        }
        inQueueB1Imag.FreeTensor(b1ImagLocal);
        inQueueA1Imag.FreeTensor(a1ImagLocal);
        inQueueB1Real.FreeTensor(b1RealLocal);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacComplexParity(int k, int cnt, const LocalTensor<T>& c1RealLocal,
                                                              const LocalTensor<T>& c1ImagLocal)
    {
        LocalTensor<T> a1RealLocal = inQueueA1Real.DeQue<T>();
        LocalTensor<T> b1RealLocal = inQueueB1Real.DeQue<T>();
        LocalTensor<T> b1ImagLocal = inQueueB1Imag.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            MacStepParityReal<trans_A, trans_B>(k, x, a1RealLocal, b1RealLocal, b1ImagLocal, c1RealLocal,
                                                c1ImagLocal);
        }
        inQueueA1Real.FreeTensor(a1RealLocal);
        LocalTensor<T> a1ImagLocal = inQueueA1Imag.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            MacStepParityImag<trans_A, trans_B>(k, x, a1ImagLocal, b1RealLocal, b1ImagLocal, c1RealLocal,
                                                c1ImagLocal);
        }
        inQueueB1Imag.FreeTensor(b1ImagLocal);
        inQueueA1Imag.FreeTensor(a1ImagLocal);
        inQueueB1Real.FreeTensor(b1RealLocal);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::MacReal(int k, int cnt, const LocalTensor<T>& c1RealLocal)
    {
        LocalTensor<T> a1RealLocal = inQueueA1Real.DeQue<T>();
        LocalTensor<T> b1RealLocal = inQueueB1Real.DeQue<T>();
        for (int x = 0; x < cnt; ++x)
        {
            SetKInner(k, x);
            SplitA<trans_A>(a1RealLocal, x * baseK);
            SplitB<trans_B>(b1RealLocal, x * baseK);
            LocalTensor<T> a2Local = inQueueA2.DeQue<T>();
            Compute(a2Local, c1RealLocal, !k && !x, k + x == kBlocks - 1);
            inQueueA2.FreeTensor(a2Local);
        }
        inQueueA1Real.FreeTensor(a1RealLocal);
        inQueueB1Real.FreeTensor(b1RealLocal);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::BeginKTile(int k, int i, int j, int& offsetA_1, int& offsetB_1)
    {
        static constexpr uint8_t padList[]{0, 0, 0, 0};
        SetKChunk(k);
        if constexpr (trans_A) Load3DSetFMatrixCal(1, totalKALigned, padList);
        Load3DSetFMatrixBCal(1, totalKALigned, padList);
        SetTileOffsets<trans_A, trans_B>(k, i, j, offsetA_1, offsetB_1);
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::RunKBlocks(int i, int j, const LocalTensor<T>& c1RealLocal,
                                                        const LocalTensor<T>& c1ImagLocal)
    {
        for (int k = 0; k < kBlocks; k += stepK)
        {
            int offsetA_1, offsetB_1;
            BeginKTile<trans_A, trans_B>(k, i, j, offsetA_1, offsetB_1);
            CopyInComplex<trans_A, trans_B>(offsetA_1, offsetB_1, cnt);
            MacComplex<trans_A, trans_B>(k, cnt, c1RealLocal, c1ImagLocal);
        }
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::RunKBlocksReal(int i, int j, const LocalTensor<T>& c1RealLocal)
    {
        for (int k = 0; k < kBlocks; k += stepK)
        {
            int offsetA_1, offsetB_1;
            BeginKTile<trans_A, trans_B>(k, i, j, offsetA_1, offsetB_1);
            CopyInAReal<trans_A, false>(offsetA_1, cnt);
            CopyInBReal<trans_B, false>(offsetB_1, cnt);
            MacReal<trans_A, trans_B>(k, cnt, c1RealLocal);
        }
    }

template <typename T>
    template <bool trans_A, bool trans_B>
    __aicore__ inline void CMatmulCustom<T>::RunKBlocksParity(int i, int j, const LocalTensor<T>& c1RealLocal,
                                                              const LocalTensor<T>& c1ImagLocal)
    {
        for (int k = 0; k < kBlocks; k += stepK)
        {
            int offsetA_1, offsetB_1;
            BeginKTile<trans_A, trans_B>(k, i, j, offsetA_1, offsetB_1);
            CopyInComplexParity<trans_A, trans_B>(offsetA_1, offsetB_1, cnt);
            MacComplexParity<trans_A, trans_B>(k, cnt, c1RealLocal, c1ImagLocal);
        }
    }

