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
 * \file gemm.hpp
 * \brief
 */
#include <kernel_operator.h>

__aicore__ inline constexpr int ceil_div(int x, int y)
{
    return (x - 1) / y + 1;
}

__aicore__ inline constexpr int ceil_round(int x, int y)
{
    return ceil_div(x, y) * y;
}

template<typename A_T, typename B_T, typename C_T>
class MatmulCustom
{
    AscendC::GlobalTensor<A_T> a_global_tensor;
    AscendC::GlobalTensor<B_T> b_global_tensor;
    AscendC::GlobalTensor<C_T> c_global_tensor;
    AscendC::TQue<AscendC::QuePosition::A1, 1> a1_t_que;
    AscendC::TQue<AscendC::QuePosition::A2, 1> a2_t_que;
    AscendC::TQue<AscendC::QuePosition::B1, 1> b1_t_que;
    AscendC::TQue<AscendC::QuePosition::B2, 1> b2_t_que;
    AscendC::TQue<AscendC::QuePosition::CO1, 1> co1_t_que;
    static constexpr int L0_M = 128;
    static constexpr int L0_N = 128;
    static constexpr int L0_K = 64;
    static constexpr int L1_M_BLOCKS = 2;
    static constexpr int L1_N_BLOCKS = 2;
    static constexpr int L1_K_BLOCKS = 2;
    static constexpr int L1_M = L0_M * L1_M_BLOCKS;
    static constexpr int L1_N = L0_N * L1_N_BLOCKS;
    static constexpr int L1_K = L0_K * L1_K_BLOCKS;
    int offsetA, offsetB, offsetC;
    int singleCoreM, singleCoreN, singleCoreK;
    int N, K;
    bool isTransposeB, if_atomicadd;
public:
    __aicore__ inline MatmulCustom() {}
    __aicore__ inline void Init(AscendC::TPipe *t_pipe)
    {
        t_pipe->InitBuffer(a1_t_que, 2, L1_M * L1_K * sizeof(A_T));
        t_pipe->InitBuffer(a2_t_que, 2, L0_M * L0_K * sizeof(A_T));
        t_pipe->InitBuffer(b1_t_que, 2, L1_K * L1_N * sizeof(B_T));
        t_pipe->InitBuffer(b2_t_que, 2, L0_K * L0_N * sizeof(B_T));
        t_pipe->InitBuffer(co1_t_que, 2, L0_M * L0_N * sizeof(C_T));
    }
    __aicore__ inline void SetMatrix(AscendC::GlobalTensor<A_T> aGlobal, AscendC::GlobalTensor<B_T> bGlobal, AscendC::GlobalTensor<C_T> cGlobal, int N, int K)
    {
        this->N = N;
        this->K = K;
        this->a_global_tensor = aGlobal;
        this->b_global_tensor = bGlobal;
        this->c_global_tensor = cGlobal;
    }
    // L1_N 循环在外（K 方向内层）的调用入口，cubeidx 用于跨核串行化的原子累加
    __aicore__ inline void Process(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int cubeidx, bool isTransposeB, bool if_atomicadd)
    {
        Run<true>(offsetA, offsetB, offsetC, blockM, blockN, blockK, cubeidx, isTransposeB, if_atomicadd);
    }
    // L1_M 循环在外（K 方向内层）的调用入口，原子累加不做跨核串行化
    __aicore__ inline void Process2(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int cubeidx, bool isTransposeB, bool if_atomicadd)
    {
        Run<false>(offsetA, offsetB, offsetC, blockM, blockN, blockK, cubeidx, isTransposeB, if_atomicadd);
    }

private:
    __aicore__ inline void SetRightFmatrix(int l1_k_aligned)
    {
        static constexpr uint8_t pad_list[4]{};
        AscendC::SetFmatrix(1, l1_k_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_RIGHT);
    }
    __aicore__ inline void SetLeftFmatrix(int l1_m_aligned)
    {
        static constexpr uint8_t pad_list[4]{};
        AscendC::SetFmatrix(1, l1_m_aligned, pad_list, AscendC::FmatrixMode::FMATRIX_LEFT);
    }
    // L1_M x L1_K 的 A 块搬入：GM 上按行主序，L1 上按 NZ 布局
    __aicore__ inline AscendC::LocalTensor<A_T> CopyInA(int i, int k, int l1_k, int l1_m, int l1_m_aligned)
    {
        AscendC::LocalTensor<A_T> a1_local_tensor = a1_t_que.AllocTensor<A_T>();
        AscendC::Nd2NzParams nd2_nz_params;
        nd2_nz_params.ndNum = 1;
        nd2_nz_params.nValue = l1_m;
        nd2_nz_params.dValue = l1_k;
        nd2_nz_params.srcDValue = K;
        nd2_nz_params.dstNzC0Stride = l1_m_aligned;
        nd2_nz_params.dstNzNStride = 1;
        AscendC::DataCopy(a1_local_tensor, a_global_tensor[offsetA + k * L1_M * K + i * L1_K], nd2_nz_params);
        a1_t_que.EnQue(a1_local_tensor);
        return a1_t_que.DeQue<A_T>();
    }
    // L1_K x L1_N 的 B 块搬入，转置与否对应两种 GM 步长/NZ 布局
    __aicore__ inline AscendC::LocalTensor<B_T> CopyInB(int i, int j, int l1_k, int l1_k_aligned, int l1_n, int l1_n_aligned)
    {
        AscendC::LocalTensor<B_T> b1_local_tensor = b1_t_que.AllocTensor<B_T>();
        AscendC::Nd2NzParams nd2_nz_params;
        nd2_nz_params.ndNum = 1;
        nd2_nz_params.dstNzNStride = 1;
        if (isTransposeB)
        {
            nd2_nz_params.nValue = l1_n;
            nd2_nz_params.dValue = l1_k;
            nd2_nz_params.srcDValue = K;
            nd2_nz_params.dstNzC0Stride = l1_n_aligned;
            AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K + j * L1_N * K], nd2_nz_params);
        }
        else
        {
            nd2_nz_params.nValue = l1_k;
            nd2_nz_params.dValue = l1_n;
            nd2_nz_params.srcDValue = N;
            nd2_nz_params.dstNzC0Stride = l1_k_aligned;
            AscendC::DataCopy(b1_local_tensor, b_global_tensor[offsetB + i * L1_K * N + j * L1_N], nd2_nz_params);
        }
        b1_t_que.EnQue(b1_local_tensor);
        return b1_t_que.DeQue<B_T>();
    }
    // 从 L1 的 A 块切出 L0_M x L0_K 的 a2
    __aicore__ inline void SplitA(const AscendC::LocalTensor<A_T> &a1_local_tensor, int l1_k_aligned, int m, int o,
                                  int l0_m_aligned, int l0_k_aligned)
    {
        AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.AllocTensor<A_T>();
        AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
        load_data_3d_params_v2_pro.channelSize = l1_k_aligned;
        short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
        extConfig[0] = l0_k_aligned;
        extConfig[1] = l0_m_aligned;
        extConfig[2] = o * L0_K;
        extConfig[3] = m * L0_M;
        AscendC::LoadData(a2_local_tensor, a1_local_tensor, load_data_3d_params_v2_pro);
        a2_t_que.EnQue(a2_local_tensor);
    }
    // 从 L1 的 B 块切出 L0_K x L0_N 的 b2；转置时走 2D 搬运并按 n 分块累加偏移
    __aicore__ inline void SplitB(const AscendC::LocalTensor<B_T> &b1_local_tensor, int n, int o, int l1_n_aligned,
                                  int l1_n_blocks, int l0_n_aligned, int l0_k_aligned)
    {
        AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.AllocTensor<B_T>();
        if (isTransposeB)
        {
            AscendC::LoadData2dParams loadDataParams;
            int loopCount = l0_k_aligned / 8;
            loadDataParams.startIndex = (l1_n_aligned * o * L0_K + n * L0_N * 8) / 128;
            loadDataParams.repeatTimes = l0_n_aligned / 16;
            loadDataParams.srcStride = 1;
            uint64_t l0bOffset = 0;
            if (l1_n_blocks == 1)
            {
                loadDataParams.repeatTimes *= loopCount;
                AscendC::LoadData(b2_local_tensor, b1_local_tensor, loadDataParams);
            }
            else
            {
                for (int s = 0; s != loopCount; s++)
                {
                    AscendC::LoadData(b2_local_tensor[l0bOffset], b1_local_tensor, loadDataParams);
                    loadDataParams.startIndex += (l1_n_aligned * 8) / 128;
                    l0bOffset += 8 * l0_n_aligned;
                }
            }
        }
        else
        {
            AscendC::LoadData3DParamsV2Pro load_data_3d_params_v2_pro;
            load_data_3d_params_v2_pro.channelSize = l1_n_aligned;
            load_data_3d_params_v2_pro.fMatrixCtrl = true;
            short (&extConfig)[4] = *reinterpret_cast<short (*)[4]>(&load_data_3d_params_v2_pro.extConfig);
            extConfig[0] = l0_n_aligned;
            extConfig[1] = l0_k_aligned;
            extConfig[2] = n * L0_N;
            extConfig[3] = o * L0_K;
            AscendC::LoadData(b2_local_tensor, b1_local_tensor, load_data_3d_params_v2_pro);
        }
        b2_t_que.EnQue(b2_local_tensor);
    }
    // 单个 L0 块的 Mmad，首块（o == 0）清零累加器；小尺寸块后补 PipeBarrier 保证顺序
    __aicore__ inline void ComputeL0Tile(const AscendC::LocalTensor<C_T> &co1_local_tensor, int l0_m, int l0_n,
                                         int l0_k, int o)
    {
        AscendC::LocalTensor<A_T> a2_local_tensor = a2_t_que.DeQue<A_T>();
        AscendC::LocalTensor<B_T> b2_local_tensor = b2_t_que.DeQue<B_T>();
        AscendC::MmadParams mmad_params;
        mmad_params.m = max(l0_m, 2);
        mmad_params.n = l0_n;
        mmad_params.k = l0_k;
        mmad_params.cmatrixInitVal = o == 0;
        AscendC::Mmad(co1_local_tensor, a2_local_tensor, b2_local_tensor, mmad_params);
        if (l0_m * l0_n < 2560)
        {
            AscendC::PipeBarrier<PIPE_M>();
        }
        a2_t_que.FreeTensor(a2_local_tensor);
        b2_t_que.FreeTensor(b2_local_tensor);
    }
    // 单个 L1 tile（L1_M x L1_N）内按 L0 块遍历：切 a2/b2 -> Mmad -> Fixpipe
    template<bool nOuter>
    __aicore__ inline void ComputeTileRow(int k, int j, const AscendC::LocalTensor<A_T> &a1_local_tensor,
                                          const AscendC::LocalTensor<B_T> &b1_local_tensor, int l1_m, int l1_m_aligned,
                                          int l1_k, int l1_k_aligned, int l1_n, int l1_n_aligned, int &cubeidx)
    {
        int l1_m_blocks = ceil_div(l1_m, L0_M);
        int l1_n_blocks = ceil_div(l1_n, L0_N);
        for (int m = 0; m < l1_m_blocks; m++)
        {
            int l0_m = min(l1_m - m * L0_M, L0_M);
            int l0_m_aligned = ceil_round(l0_m, 16);
            for (int n = 0; n < l1_n_blocks; n++)
            {
                int l0_n = min(l1_n - n * L0_N, L0_N);
                int l0_n_aligned = ceil_round(l0_n, 16);
                AscendC::LocalTensor<C_T> co1_local_tensor = co1_t_que.AllocTensor<C_T>();
                int l1_k_blocks = ceil_div(l1_k, L0_K);
                for (int o = 0; o < l1_k_blocks; o++)
                {
                    int l0_k = min(l1_k - o * L0_K, L0_K);
                    int l0_k_aligned = ceil_round(l0_k, 8);
                    SplitA(a1_local_tensor, l1_k_aligned, m, o, l0_m_aligned, l0_k_aligned);
                    SplitB(b1_local_tensor, n, o, l1_n_aligned, l1_n_blocks, l0_n_aligned, l0_k_aligned);
                    ComputeL0Tile(co1_local_tensor, l0_m, l0_n, l0_k, o);
                }
                co1_t_que.EnQue(co1_local_tensor);
                co1_local_tensor = co1_t_que.DeQue<C_T>();
                CopyOutTile<nOuter>(co1_local_tensor, k, m, n, j, l0_m, l0_n, l0_m_aligned, cubeidx);
                co1_t_que.FreeTensor(co1_local_tensor);
            }
        }
    }
    // 指定 L1_N 序号的 B 块与全部 L1_M 块相乘（Process 的 j 外层顺序）
    __aicore__ inline void RunNOuter(int i, int l2_n_blocks, int l2_m_blocks, int l1_k, int l1_k_aligned, int &cubeidx)
    {
        for (int j = 0; j < l2_n_blocks; j++)
        {
            int l1_n = min(singleCoreN - j * L1_N, L1_N);
            int l1_n_aligned = ceil_round(l1_n, 16);
            AscendC::LocalTensor<B_T> b1_local_tensor = CopyInB(i, j, l1_k, l1_k_aligned, l1_n, l1_n_aligned);
            for (int k = 0; k < l2_m_blocks; k++)
            {
                int l1_m = min(singleCoreM - k * L1_M, L1_M);
                int l1_m_aligned = ceil_round(l1_m, 16);
                SetLeftFmatrix(l1_m_aligned);
                AscendC::LocalTensor<A_T> a1_local_tensor = CopyInA(i, k, l1_k, l1_m, l1_m_aligned);
                ComputeTileRow<true>(k, j, a1_local_tensor, b1_local_tensor, l1_m, l1_m_aligned, l1_k, l1_k_aligned,
                                     l1_n, l1_n_aligned, cubeidx);
                a1_t_que.FreeTensor(a1_local_tensor);
            }
            b1_t_que.FreeTensor(b1_local_tensor);
        }
    }
    // 指定 L1_M 序号的 A 块与全部 L1_N 块相乘（Process2 的 k 外层顺序）
    __aicore__ inline void RunMOuter(int i, int l2_n_blocks, int l2_m_blocks, int l1_k, int l1_k_aligned, int &cubeidx)
    {
        for (int k = 0; k < l2_m_blocks; k++)
        {
            int l1_m = min(singleCoreM - k * L1_M, L1_M);
            int l1_m_aligned = ceil_round(l1_m, 16);
            SetLeftFmatrix(l1_m_aligned);
            AscendC::LocalTensor<A_T> a1_local_tensor = CopyInA(i, k, l1_k, l1_m, l1_m_aligned);
            for (int j = 0; j < l2_n_blocks; j++)
            {
                int l1_n = min(singleCoreN - j * L1_N, L1_N);
                int l1_n_aligned = ceil_round(l1_n, 16);
                AscendC::LocalTensor<B_T> b1_local_tensor = CopyInB(i, j, l1_k, l1_k_aligned, l1_n, l1_n_aligned);
                ComputeTileRow<false>(k, j, a1_local_tensor, b1_local_tensor, l1_m, l1_m_aligned, l1_k, l1_k_aligned,
                                      l1_n, l1_n_aligned, cubeidx);
                b1_t_que.FreeTensor(b1_local_tensor);
            }
            a1_t_que.FreeTensor(a1_local_tensor);
        }
    }
    // 统一的分层驱动：nOuter 决定 L1_N/L1_M 谁是外层循环，保持两条路径各自的分块顺序
    template<bool nOuter>
    __aicore__ inline void Run(int offsetA, int offsetB, int offsetC, int blockM, int blockN, int blockK, int cubeidx,
                               bool isTransposeB, bool if_atomicadd)
    {
        this->offsetA = offsetA;
        this->offsetB = offsetB;
        this->offsetC = offsetC;
        this->singleCoreM = blockM;
        this->singleCoreN = blockN;
        this->singleCoreK = blockK;
        this->isTransposeB = isTransposeB;
        this->if_atomicadd = if_atomicadd;

        int l2_m_blocks = ceil_div(singleCoreM, L1_M);
        int l2_n_blocks = ceil_div(singleCoreN, L1_N);
        int l2_k_blocks = ceil_div(singleCoreK, L1_K);
        int cube = cubeidx;

        for (int i = 0; i < l2_k_blocks; i++)
        {
            int l1_k = min(singleCoreK - i * L1_K, L1_K);
            int l1_k_aligned = ceil_round(l1_k, 8);
            SetRightFmatrix(l1_k_aligned);
            if (nOuter)
            {
                RunNOuter(i, l2_n_blocks, l2_m_blocks, l1_k, l1_k_aligned, cube);
            }
            else
            {
                RunMOuter(i, l2_n_blocks, l2_m_blocks, l1_k, l1_k_aligned, cube);
            }
        }
    }
    // nOuter=true 对应 Process 的三态串行化，false 对应 Process2 的纯原子累加
    template<bool nOuter>
    __aicore__ inline void CopyOutTile(const AscendC::LocalTensor<C_T> &co1_local_tensor, int k, int m, int n, int j,
                                       int l0_m, int l0_n, int l0_m_aligned, int &cubeidx)
    {
        AscendC::FixpipeParamsV220 fixpipe_params_v220;
        fixpipe_params_v220.nSize = l0_n;
        fixpipe_params_v220.mSize = l0_m;
        fixpipe_params_v220.srcStride = l0_m_aligned;
        fixpipe_params_v220.dstStride = N;
        int offset = offsetC + (k * L1_M + m * L0_M) * N + n * L0_N + j * L1_N;
        if (!if_atomicadd)
        {
            AscendC::Fixpipe(c_global_tensor[offset], co1_local_tensor, fixpipe_params_v220);
            return;
        }
        if (!nOuter)
        {
            AscendC::SetAtomicAdd<float>();
            AscendC::Fixpipe(c_global_tensor[offset], co1_local_tensor, fixpipe_params_v220);
            AscendC::SetAtomicNone();
            return;
        }
        CopyOutSerialized(co1_local_tensor, offset, fixpipe_params_v220, cubeidx);
    }
    // Process 的原子累加串行化：cubeidx == 0 先写后放旗标，其余核先等旗标再累加
    __aicore__ inline void CopyOutSerialized(const AscendC::LocalTensor<C_T> &co1_local_tensor, int offset,
                                             const AscendC::FixpipeParamsV220 &fixpipe_params_v220, int &cubeidx)
    {
        if (cubeidx == -1)
        {
            AscendC::SetAtomicAdd<float>();
            AscendC::Fixpipe(c_global_tensor[offset], co1_local_tensor, fixpipe_params_v220);
            AscendC::SetAtomicNone();
        }
        else if (cubeidx == 0)
        {
            AscendC::Fixpipe(c_global_tensor[offset], co1_local_tensor, fixpipe_params_v220);
            AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
            AscendC::CrossCoreWaitFlag(0x3);
            cubeidx = -1;
        }
        else if (cubeidx > 0 && cubeidx < AscendC::GetBlockNum())
        {
            AscendC::SetAtomicAdd<float>();
            AscendC::CrossCoreSetFlag<0x0, PIPE_FIX>(0x3);
            AscendC::CrossCoreWaitFlag(0x3);
            AscendC::Fixpipe(c_global_tensor[offset], co1_local_tensor, fixpipe_params_v220);
            AscendC::SetAtomicNone();
            cubeidx = -1;
        }
    }
};
