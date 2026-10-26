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
 * \file gemm_tile.hpp
 * \brief CMatmulCustom：L1->L0 搬运、Split 与 Mmad 原语（类外定义）
 */

#pragma once

#include "gemm.hpp"

template <typename T>
    template <bool trans, bool neg>
    __aicore__ inline void CMatmulCustom<T>::CopyInATile(int offsetA_1, const GlobalTensor<T>& srcGlobal,
                                                         TQue<QuePosition::A1, 2>& inQueue)
    {
        LocalTensor<T> a1Local = inQueue.AllocTensor<T>();
        if constexpr (trans)
        {
            Nd2NzParams intriParamsMatA{1,
                                        static_cast<uint16_t>(totalKALigned),
                                        static_cast<uint16_t>(tailMAligned),
                                        0,
                                        static_cast<uint16_t>(strideA),
                                        static_cast<uint16_t>(totalKALigned),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(a1Local, srcGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, extGlobal[offsetA_1], intriParamsMatA);
        }
        else
        {
            Nd2NzParams intriParamsMatA{1,
                                        static_cast<uint16_t>(tailM),
                                        static_cast<uint16_t>(totalKALigned),
                                        0,
                                        static_cast<uint16_t>(strideA),
                                        static_cast<uint16_t>(tailMAligned),
                                        1,
                                        1};
            if constexpr (!neg)
                DataCopy(a1Local, srcGlobal[offsetA_1], intriParamsMatA);
            else
                DataCopy(a1Local, extGlobal[offsetA_1], intriParamsMatA);
        }
        inQueue.EnQue(a1Local);
    }

template <typename T>
    template <bool trans, bool neg>
    __aicore__ inline void CMatmulCustom<T>::CopyInAReal(int offsetA_1, int cnt)
    {
        CopyInATile<trans, neg>(offsetA_1, aRealGlobal, inQueueA1Real);
    }

template <typename T>
    template <bool trans, bool neg>
    __aicore__ inline void CMatmulCustom<T>::CopyInAImag(int offsetA_1, int cnt)
    {
        CopyInATile<trans, neg>(offsetA_1, aImagGlobal, inQueueA1Imag);
    }

template <typename T>
    template <bool trans, bool neg>
    __aicore__ inline void CMatmulCustom<T>::CopyInBTile(int offsetB_1, const GlobalTensor<T>& srcGlobal,
                                                         TQue<QuePosition::B1, 2>& inQueue)
    {
        LocalTensor<T> b1Local = inQueue.AllocTensor<T>();
        if constexpr (trans)
        {
            Nd2NzParams intriParamsMatB{
                1,
                static_cast<uint16_t>(tailN),
                static_cast<uint16_t>(totalKALigned),
                0,
                static_cast<uint16_t>(strideB),
                static_cast<uint16_t>(tailNAligned),
                1,
                1,
            };
            if constexpr (!neg)
                DataCopy(b1Local, srcGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, extGlobal[offsetB_1], intriParamsMatB);
        }
        else
        {
            Nd2NzParams intriParamsMatB{
                1,
                static_cast<uint16_t>(totalKALigned),
                static_cast<uint16_t>(tailNAligned),
                0,
                static_cast<uint16_t>(strideB),
                static_cast<uint16_t>(totalKALigned),
                1,
                1,
            };
            if constexpr (!neg)
                DataCopy(b1Local, srcGlobal[offsetB_1], intriParamsMatB);
            else
                DataCopy(b1Local, extGlobal[offsetB_1], intriParamsMatB);
        }
        inQueue.EnQue(b1Local);
    }

template <typename T>
    template <bool trans, bool neg>
    __aicore__ inline void CMatmulCustom<T>::CopyInBReal(int offsetB_1, int cnt)
    {
        CopyInBTile<trans, neg>(offsetB_1, bRealGlobal, inQueueB1Real);
    }

template <typename T>
    template <bool trans, bool neg>
    __aicore__ inline void CMatmulCustom<T>::CopyInBImag(int offsetB_1, int cnt)
    {
        CopyInBTile<trans, neg>(offsetB_1, bImagGlobal, inQueueB1Imag);
    }

template <typename T>
    template <bool trans>
    __aicore__ inline void CMatmulCustom<T>::SplitA(const LocalTensor<T>& a1Local, int prevK)
    {
        LocalTensor<T> a2Local = inQueueA2.AllocTensor<T>();
        LoadData3DParamsV2Pro loadData3DV2;
        if constexpr (trans)
        {
            uint16_t mPos = prevK, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = tailMAligned;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                     ((uint64_t)alignedK << 16) | (uint64_t)tailMAligned;
            loadData3DV2.enTranspose = true;
            LoadData<T>(a2Local, a1Local, loadData3DV2);
        }
        else
        {
            uint16_t mPos = 0, sAL1NOffset_ = 0;
            loadData3DV2.channelSize = alignedK;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sAL1NOffset_ << 32) |
                                     ((uint64_t)tailMAligned << 16) | (uint64_t)alignedK;
            LoadData<T>(a2Local, a1Local[prevK * tailMAligned], loadData3DV2);
        }
        inQueueA2.EnQue(a2Local);
    }

template <typename T>
    template <bool trans>
    __aicore__ inline void CMatmulCustom<T>::SplitB(const LocalTensor<T>& b1Local, int prevK)
    {
        LocalTensor<T> b2Local = inQueueB2.AllocTensor<T>();
        if constexpr (trans)
        {
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = alignedK * tailNAligned / 128;
            loadDataParams.srcStride = 1;
            loadDataParams.dstGap = 0;
            loadDataParams.ifTranspose = 0;
            LoadData<T>(b2Local, b1Local[prevK * tailNAligned], loadDataParams);
        }
        else
        {
            LoadData3DParamsV2Pro loadData3DV2;
            uint16_t mPos = 0, sBL1NOffset_ = 0;
            loadData3DV2.channelSize = tailNAligned;
            loadData3DV2.extConfig = ((uint64_t)mPos << 48) | ((uint64_t)sBL1NOffset_ << 32) |
                                     ((uint64_t)alignedK << 16) | (uint64_t)tailNAligned;
            loadData3DV2.fMatrixCtrl = true;
            LoadData<T>(b2Local, b1Local[prevK * 8], loadData3DV2);
        }
        inQueueB2.EnQue(b2Local);
    }

template <typename T>
    __aicore__ inline void CMatmulCustom<T>::Compute(const LocalTensor<T>& a2Local,
                                                     const LocalTensor<T>& c1Local, int first,
                                                     const int unitFlag)
    {
        LocalTensor<T> b2Local = inQueueB2.DeQue<T>();
        MmadParams mmadParams;
        mmadParams.m = tailMAligned;
        mmadParams.n = tailNAligned;
        mmadParams.k = alignedK;
        mmadParams.unitFlag = unitFlag ? 3 : 2;
        if (first)
        {
            Mmad(c1Local, a2Local, b2Local, mmadParams);
        }
        else
        {
            mmadParams.cmatrixInitVal = false;
            Mmad(c1Local, a2Local, b2Local, c1Local, mmadParams);
        }
        if (tailMAligned * tailNAligned < 2560) PipeBarrier<PIPE_M>();
        inQueueB2.FreeTensor(b2Local);
    }

template <typename T>
    template <bool imag>
    __aicore__ inline void CMatmulCustom<T>::CopyOut(const LocalTensor<T>& c1Local, int offsetC_1)
    {
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = tailN;
        fixpipeParams.mSize = tailM;
        fixpipeParams.srcStride = tailMAligned;
        fixpipeParams.dstStride = strideC;

        fixpipeParams.unitFlag = 3;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 0;
        fixpipeParams.dstNdStride = 0;
        if constexpr (!imag)
            Fixpipe(cRealGlobal[offsetC_1], c1Local, fixpipeParams);
        else
            Fixpipe(cImagGlobal[offsetC_1], c1Local, fixpipeParams);
    }

