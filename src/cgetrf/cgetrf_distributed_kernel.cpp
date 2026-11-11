/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Distributed panel structure adapted from sss-hust/ops-solver, commit
// 877434c1ad53b22aea3893c864d3e18c0f8941fe, src/cgetrf/cgetrf_kernel.cpp.
// Keep the caller's interleaved matrix and existing trailing-update path.
#include "kernel_operator.h"

// Device API declarations precede the implementation fragments below.
#include "../utils/complex_solver_kernel.hpp"
#include "../utils/complex_solver_launch.h"

using namespace AscendC;
// DistributedPanel 面板执行器仅由 3510（Ascend950）的 SIMT VF 路径使用，其方法引用
// 上方 3510 守卫内的 VF 声明；非 3510 架构编译时因标识符不存在而失败（issue #389），
// 与文件内其他内核的守卫口径对齐。
#if __NPU_ARCH__ == 3510
namespace DistributedPanel
{
__aicore__ inline void ComplexRecipSafe(float dr, float di, float& invR, float& invI)
{
    const float adr = (dr >= 0.0f) ? dr : -dr;
    const float adi = (di >= 0.0f) ? di : -di;
    const float s = (adr >= adi) ? adr : adi;
    if (!(s > 0.0f))
    {
        invR = 0.0f;
        invI = 0.0f;
        return;
    }
    const float r = dr / s;
    const float i = di / s;
    const float denom = s * (r * r + i * i);
    invR = r / denom;
    invI = -i / denom;
}

#if __NPU_ARCH__ == 3510
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void ReducePanelPivotVf(__ubuf__ const float* input,
                                                                       __ubuf__ float* output, int first, int count,
                                                                       int stride, int rowBase)
{
    float best = -1.0f;
    int row = 0x7fffffff;
    for (int index = first + threadIdx.x; index < count; index += blockDim.x)
    {
        float score = input[index * stride];
        int candidate =
            stride == 1 ? rowBase + index : reinterpret_cast<__ubuf__ const int*>(input)[index * stride + 1];
        if (score > best || (score == best && candidate < row))
        {
            best = score;
            row = candidate;
        }
    }
    float maximum = asc_reduce_max(best);
    row = asc_reduce_min(best == maximum ? row : 0x7fffffff);
    if (threadIdx.x == 0)
    {
        output[0] = maximum;
        reinterpret_cast<__ubuf__ int*>(output)[1] = row;
    }
}
__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void ExchangePanelRowVf(__ubuf__ float* real, __ubuf__ float* imag,
                                                                       __ubuf__ float* pivot,
                                                                       __ubuf__ const float* replacement, int pitch,
                                                                       int row, int width, bool publish)
{
    int column = threadIdx.x;
    if (column >= width) return;
    int index = column * pitch + row;
    if (publish)
    {
        pivot[column] = real[index];
        pivot[64 + column] = imag[index];
    }
    real[index] = replacement[column];
    imag[index] = replacement[64 + column];
}

__simt_vf__ __aicore__ LAUNCH_BOUND(32) inline void SwapLocalPanelRowVf(__ubuf__ float* real, __ubuf__ float* imag,
                                                                        __ubuf__ float* pivot, int pitch, int row,
                                                                        int target, int width)
{
    int column = threadIdx.x;
    if (column >= width) return;
    int first = column * pitch + row, second = column * pitch + target;
    float pr = real[second], pi = imag[second];
    real[second] = real[first];
    imag[second] = imag[first];
    real[first] = pr;
    imag[first] = pi;
    pivot[column] = pr;
    pivot[64 + column] = pi;
}
#endif

class State
{
    static constexpr int32_t TILE_C = 32;
    static constexpr int32_t tileMax = TILE_C * 160;
    int32_t rows, me, numBlocks, k, kb, ld, doPivot, rLocal0, rLocal1, myRows, ldLocal;
    static constexpr uint32_t ARENA_BYTES = (2 * COMPLEX_SOLVER_PANEL * 160 + 7 * tileMax + 4096) * sizeof(float);
    int32_t rowsPerCore, j, ownerJ, jLocal, myBestRow, pivRow, firstZero = 0;
    int64_t ps, syncOff;
    float myBestMag;
    bool useRowPivBuf;
    int32_t pivLocal[64];
    GM_ADDR ws;
    GM_ADDR devIpiv;
    GM_ADDR devInfo;
    LocalTensor<float> arena;
    GlobalTensor<float> base, wr, wi, slotCand, slotRowJ, slotRowPiv;
    GlobalTensor<int32_t> wsPiv0, ipivGm, infoGm;
    LocalTensor<float> localR, localI, tileMR, tileMI, tilePMI, tileUR, tileUI;
    LocalTensor<float> multR, multI, colR, colI, maskVec, invMaskVec, magVec, ubRowJ, ubRowPiv, ubCand;
    LocalTensor<uint32_t> idxTileMult, idxTileU, idxRowExt;
    LocalTensor<int32_t> ubPiv, ubInfo;
    DataCopyPadExtParams<float> pad{false, 0, 0, 0};

    __aicore__ inline void InitGlobals()
    {
        base.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws));
        wr = base[0];
        wi = base[ps];
        wsPiv0.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(ws) + syncOff + 2048);
        if (doPivot) ipivGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(devIpiv));
        infoGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(devInfo));
    }

    __aicore__ inline void BindBuffers()
    {
        LocalTensor<float> pAll = arena;
        localR = pAll[0];
        localI = pAll[COMPLEX_SOLVER_PANEL * 160];

        LocalTensor<float> tAll = arena[2 * COMPLEX_SOLVER_PANEL * 160];
        tileMR = tAll[0];
        tileMI = tAll[tileMax];
        tilePMI = tAll[tileMax * 2];
        tileUR = tAll[tileMax * 3];
        tileUI = tAll[tileMax * 4];

        LocalTensor<uint32_t> iAll = arena[2 * COMPLEX_SOLVER_PANEL * 160 + 5 * tileMax].ReinterpretCast<uint32_t>();
        idxTileMult = iAll[0];
        idxTileU = iAll[tileMax];
    }

    __aicore__ inline void BindScratch()
    {
        LocalTensor<float> sAll = arena[2 * COMPLEX_SOLVER_PANEL * 160 + 7 * tileMax];
        multR = sAll[0];
        multI = sAll[160];
        colR = sAll[320];
        colI = sAll[480];
        maskVec = sAll[640];
        invMaskVec = sAll[800];
        magVec = sAll[960];
        ubRowJ = sAll[1120];
        ubRowPiv = sAll[1248];
        ubCand = sAll[1536];  // 1024 + 128 floats
        idxRowExt = sAll[2816].ReinterpretCast<uint32_t>();
        ubPiv = sAll[3072].ReinterpretCast<int32_t>();
        ubInfo = sAll[3200].ReinterpretCast<int32_t>();
    }

    __aicore__ inline void InitMasks()
    {
        Duplicate(localR, 0.0f, COMPLEX_SOLVER_PANEL * 160);
        Duplicate(localI, 0.0f, COMPLEX_SOLVER_PANEL * 160);
        Duplicate(maskVec, 1.0f, ldLocal);
        Duplicate(invMaskVec, 0.0f, ldLocal);
        PipeBarrier<PIPE_ALL>();

        for (int32_t r = myRows; r < ldLocal; ++r)
        {
            maskVec.SetValue(r, 0.0f);
            invMaskVec.SetValue(r, 1.0f);
        }
        for (int32_t c = 0; c < COMPLEX_SOLVER_PANEL; ++c)
        {
            idxRowExt.SetValue(c, static_cast<uint32_t>(c * ldLocal * sizeof(float)));
        }
        CreateVecIndex(tileMR.ReinterpretCast<int32_t>(), 0, ldLocal);
        PipeBarrier<PIPE_V>();
        ShiftLeft(tileMR.ReinterpretCast<uint32_t>(), tileMR.ReinterpretCast<uint32_t>(), 2U, ldLocal);
        PipeBarrier<PIPE_V>();
        for (int32_t c = 0; c < TILE_C; ++c)
        {
            Adds(idxTileMult[c * ldLocal].ReinterpretCast<int32_t>(), tileMR.ReinterpretCast<int32_t>(), 0, ldLocal);
            Duplicate(idxTileU[c * ldLocal].ReinterpretCast<int32_t>(), static_cast<int32_t>(c * sizeof(float)),
                      ldLocal);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void LoadPanel()
    {
        if (myRows > 0)
        {
            DataCopyExtParams cpIn{static_cast<uint16_t>(kb), static_cast<uint32_t>(myRows * sizeof(float)),
                                   static_cast<uint32_t>((ld - myRows) * sizeof(float)),
                                   static_cast<uint32_t>((ldLocal - ((myRows + 7) / 8) * 8) / 8), 0};
            DataCopyPad(localR, wr[rLocal0], cpIn, pad);
            DataCopyPad(localI, wi[rLocal0], cpIn, pad);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void PrepareColumn()
    {
        ownerJ = j / rowsPerCore;
        jLocal = j % rowsPerCore;

        if (doPivot && myRows > 0)
        {
            Abs(colR, localR[j * ldLocal], ldLocal);
            Abs(colI, localI[j * ldLocal], ldLocal);
            Add(magVec, colR, colI, ldLocal);
        }
        if (me == ownerJ)
        {
            const uint32_t rowBase = static_cast<uint32_t>(jLocal * sizeof(float));
            Gather(ubRowJ, localR, idxRowExt, rowBase, static_cast<uint32_t>(COMPLEX_SOLVER_PANEL));
            Gather(ubRowJ[64], localI, idxRowExt, rowBase, static_cast<uint32_t>(COMPLEX_SOLVER_PANEL));
        }
        PipeBarrier<PIPE_ALL>();

        myBestMag = -1.0f;
        myBestRow = j;
        if (doPivot && myRows > 0)
        {
            const int32_t rStart = (j > rLocal0) ? (j - rLocal0) : 0;
            asc_vf_call<ReducePanelPivotVf>(
                dim3{32, 1, 1}, reinterpret_cast<__ubuf__ const float*>(magVec.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ubInfo.GetPhyAddr()), rStart, myRows, 1, rLocal0);
            PipeBarrier<PIPE_ALL>();
            myBestMag = ubInfo.ReinterpretCast<float>().GetValue(0);
            myBestRow = ubInfo.GetValue(1);
        }
    }

    __aicore__ inline void PublishCandidates()
    {
        const int64_t slotOff = syncOff + 10240 + static_cast<int64_t>(j & 3) * 1536;
        slotCand = base[slotOff];
        slotRowJ = base[slotOff + 1024];
        slotRowPiv = base[slotOff + 1280];

        if (doPivot)
        {
            ubCand.SetValue(0, myBestMag);
            ubCand.ReinterpretCast<int32_t>().SetValue(1, myBestRow);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams cpCandOut{1, 2 * sizeof(float), 0, 0, 0};
            DataCopyPad(slotCand[me * 16], ubCand, cpCandOut);
        }
        if (me == ownerJ)
        {
            DataCopyExtParams cpRowOut{1, 128 * sizeof(float), 0, 0, 0};
            DataCopyPad(slotRowJ, ubRowJ, cpRowOut);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        // Single burst read of slotCand (1024 floats) + slotRowJ (128 floats)
        if (doPivot)
        {
            DataCopyExtParams cpBothIn{1, 1152 * sizeof(float), 0, 0, 0};
            DataCopyPad(ubCand, slotCand, cpBothIn, pad);
        }
        else
        {
            DataCopyExtParams cpRowIn{1, 128 * sizeof(float), 0, 0, 0};
            DataCopyPad(ubCand[1024], slotRowJ, cpRowIn, pad);
        }
        PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SelectPivot()
    {
        if (doPivot)
        {
            asc_vf_call<ReducePanelPivotVf>(
                dim3{32, 1, 1}, reinterpret_cast<__ubuf__ const float*>(ubCand.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(ubInfo.GetPhyAddr()), 0, numBlocks, 16, 0);
            PipeBarrier<PIPE_ALL>();
            pivRow = ubInfo.GetValue(1);
        }
        pivLocal[j] = k + pivRow;
    }

    __aicore__ inline void ExchangePivot()
    {
        useRowPivBuf = true;
        const int32_t ownerP = pivRow / rowsPerCore;
        const int32_t pLocal = pivRow % rowsPerCore;
        if (me == ownerP)
        {
            asc_vf_call<ExchangePanelRowVf>(dim3{32, 1, 1}, reinterpret_cast<__ubuf__ float*>(localR.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ float*>(localI.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ float*>(ubRowPiv.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ const float*>(ubCand[1024].GetPhyAddr()), ldLocal,
                                            pLocal, kb, true);
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams cpRowOut{1, 128 * sizeof(float), 0, 0, 0};
            DataCopyPad(slotRowPiv, ubRowPiv, cpRowOut);
        }
        PipeBarrier<PIPE_ALL>();
        SyncAll();
        PipeBarrier<PIPE_ALL>();

        DataCopyExtParams cpRowIn{1, 128 * sizeof(float), 0, 0, 0};
        DataCopyPad(ubRowPiv, slotRowPiv, cpRowIn, pad);
        PipeBarrier<PIPE_ALL>();
        if (me == ownerJ)
        {
            asc_vf_call<ExchangePanelRowVf>(dim3{32, 1, 1}, reinterpret_cast<__ubuf__ float*>(localR.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ float*>(localI.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ float*>(ubRowPiv.GetPhyAddr()),
                                            reinterpret_cast<__ubuf__ const float*>(ubRowPiv.GetPhyAddr()), ldLocal,
                                            jLocal, kb, false);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void LocalSwap()
    {
        pivRow = doPivot ? myBestRow : j;
        pivLocal[j] = k + pivRow;
        if (pivRow != j)
        {
            useRowPivBuf = true;
            asc_vf_call<SwapLocalPanelRowVf>(dim3{32, 1, 1}, reinterpret_cast<__ubuf__ float*>(localR.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ float*>(localI.GetPhyAddr()),
                                             reinterpret_cast<__ubuf__ float*>(ubRowPiv.GetPhyAddr()), ldLocal, j,
                                             pivRow, kb);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void Normalize(float dr, float di)
    {
        float invR = 0.0f, invI = 0.0f;
        DistributedPanel::ComplexRecipSafe(dr, di, invR, invI);
        PipeBarrier<PIPE_ALL>();

        // Scale multipliers and fuse the remaining panel columns into a rank-1 update.
        Mul(colR, localR[j * ldLocal], maskVec, ldLocal);
        Mul(colI, localI[j * ldLocal], maskVec, ldLocal);
        Muls(multR, colR, invR, ldLocal);
        Axpy<float, float>(multR, colI, -invI, ldLocal);
        Muls(multI, colI, invR, ldLocal);
        Axpy<float, float>(multI, colR, invI, ldLocal);

        Mul(localR[j * ldLocal], localR[j * ldLocal], invMaskVec, ldLocal);
        Add(localR[j * ldLocal], localR[j * ldLocal], multR, ldLocal);
        Mul(localI[j * ldLocal], localI[j * ldLocal], invMaskVec, ldLocal);
        Add(localI[j * ldLocal], localI[j * ldLocal], multI, ldLocal);
    }

    __aicore__ inline void Update(LocalTensor<float> curPiv)
    {
        Muls(colR, multR, -1.0f, ldLocal);
        Muls(colI, multI, -1.0f, ldLocal);
        const int32_t maxRemCols = (kb - 1 - j < TILE_C) ? (kb - 1 - j) : TILE_C;
        const int32_t maxTileLen = maxRemCols * ldLocal;
        Gather(tileMR, colR, idxTileMult, 0U, static_cast<uint32_t>(maxTileLen));
        Gather(tileMI, colI, idxTileMult, 0U, static_cast<uint32_t>(maxTileLen));
        Gather(tilePMI, multI, idxTileMult, 0U, static_cast<uint32_t>(maxTileLen));

        for (int32_t c0 = j + 1; c0 < kb; c0 += TILE_C)
        {
            const int32_t remCols = (kb - c0 < TILE_C) ? (kb - c0) : TILE_C;
            const int32_t remLen = remCols * ldLocal;
            const int32_t dstOff = c0 * ldLocal;
            const uint32_t uBaseR = static_cast<uint32_t>(c0 * sizeof(float));
            const uint32_t uBaseI = static_cast<uint32_t>((64 + c0) * sizeof(float));

            Gather(tileUR, curPiv, idxTileU, uBaseR, static_cast<uint32_t>(remLen));
            Gather(tileUI, curPiv, idxTileU, uBaseI, static_cast<uint32_t>(remLen));

            MulAddDst<float, float>(localR[dstOff], tileMR, tileUR, remLen);
            MulAddDst<float, float>(localI[dstOff], tileMR, tileUI, remLen);
            MulAddDst<float, float>(localR[dstOff], tilePMI, tileUI, remLen);
            MulAddDst<float, float>(localI[dstOff], tileMI, tileUR, remLen);
        }
    }

    __aicore__ inline void Step()
    {
        PrepareColumn();
        pivRow = j;
        useRowPivBuf = false;
        if (numBlocks > 1)
        {
            PublishCandidates();
            SelectPivot();
            if (pivRow != j) ExchangePivot();
        }
        else
            LocalSwap();
        if (j >= rLocal0 && j < rLocal1)
        {
            maskVec.SetValue(j - rLocal0, 0.0f);
            invMaskVec.SetValue(j - rLocal0, 1.0f);
        }
        LocalTensor<float> curPiv = useRowPivBuf ? ubRowPiv : ((numBlocks > 1) ? ubCand[1024] : ubRowJ);
        float dr = curPiv.GetValue(j);
        float di = curPiv.GetValue(64 + j);
        if (dr == 0.0f && di == 0.0f)
        {
            if (firstZero == 0) firstZero = k + j + 1;
            return;
        }
        if (myRows > 0 && rLocal1 > j + 1)
        {
            Normalize(dr, di);
            if (j + 1 < kb) Update(curPiv);
        }
    }

    __aicore__ inline void StorePanel()
    {
        PipeBarrier<PIPE_ALL>();
        if (myRows > 0)
        {
            DataCopyExtParams cpOut{static_cast<uint16_t>(kb), static_cast<uint32_t>(myRows * sizeof(float)),
                                    static_cast<uint32_t>((ldLocal - ((myRows + 7) / 8) * 8) / 8),
                                    static_cast<uint32_t>((ld - myRows) * sizeof(float)), 0};
            DataCopyPad(wr[rLocal0], localR, cpOut);
            DataCopyPad(wi[rLocal0], localI, cpOut);
        }
    }

    __aicore__ inline void StorePivots()
    {
        for (int32_t j = 0; j < kb; ++j)
        {
            ubPiv.SetValue(j, pivLocal[j]);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams cpPiv{1, static_cast<uint32_t>(kb * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(wsPiv0[k], ubPiv, cpPiv);
        PipeBarrier<PIPE_ALL>();

        if (doPivot)
        {
            for (int32_t j = 0; j < kb; ++j)
            {
                ubPiv.SetValue(j, pivLocal[j] + 1);
            }
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(ipivGm[k], ubPiv, cpPiv);
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void StoreInfo()
    {
        DataCopyExtParams cpInfo{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padInfo{false, 0, 0, 0};
        DataCopyPad(ubInfo, infoGm, cpInfo, padInfo);
        PipeBarrier<PIPE_ALL>();
        if (ubInfo.GetValue(0) == 0)
        {
            ubInfo.SetValue(0, firstZero);
            PipeBarrier<PIPE_ALL>();
            DataCopyPad(infoGm, ubInfo, cpInfo);
            PipeBarrier<PIPE_ALL>();
        }
    }

   public:
    __aicore__ inline void Init(GM_ADDR workspace, GM_ADDR pivot, GM_ADDR status, int height, int stride, int begin,
                                int width, int partition, __ubuf__ float* memory)
    {
        ws = workspace;
        devIpiv = pivot;
        devInfo = status;
        k = begin;
        kb = width;
        ld = stride;
        rowsPerCore = partition;
        doPivot = pivot != nullptr;
        rows = height - begin;
        me = GetBlockIdx();
        numBlocks = GetBlockNum();
        ps = static_cast<int64_t>(ld) * COMPLEX_SOLVER_PANEL;
        syncOff = ps * 2;
        rLocal0 = me * rowsPerCore;
        rLocal1 = (rLocal0 + rowsPerCore < rows ? rLocal0 + rowsPerCore : rows);
        myRows = (rLocal1 > rLocal0 ? rLocal1 - rLocal0 : 0);
        ldLocal = rowsPerCore;
        InitGlobals();
        arena = LocalTensor<float>(TPosition::VECCALC, static_cast<uint32_t>(reinterpret_cast<uintptr_t>(memory)),
                                   ARENA_BYTES / sizeof(float));
        BindBuffers();
        BindScratch();
        InitMasks();
        LoadPanel();
    }

    __aicore__ inline void Run()
    {
        for (j = 0; j < kb; ++j) Step();
        StorePanel();
        if (me == 0)
        {
            StorePivots();
            if (firstZero > 0) StoreInfo();
        }
        PipeBarrier<PIPE_ALL>();
    }
};

}  // namespace DistributedPanel
#endif

extern "C" __global__ __aicore__ __schedmode__(1) void complex_factor_distributed_panel(
    GM_ADDR workspace, GM_ADDR pivots, GM_ADDR info, int m, int n, int ld, int columns, int begin, int width,
    int rowsPerCore, int doPivot)
{
    // 任务类型属性必须无条件保留：__schedmode__ 内核在非 3510 架构上若函数体全空，
    // 链接器无法自动推导 kernel 类型（issue #389）。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    __ubuf__ float memory[200704 / sizeof(float)];
    DistributedPanel::State state;
    state.Init(workspace, pivots, info, m, ld, begin, width, rowsPerCore, memory);
    state.Run();
#endif
}

#if __NPU_ARCH__ == 3510
using namespace ComplexSolverKernel;
__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void LoadDistributedPanelVf(ComplexSolverShape shape,
                                                                            __gm__ aclFloatComplex* matrix,
                                                                            __gm__ float* workspace, int begin, int end,
                                                                            int ld)
{
    int height = shape.m - begin;
    int plane = ld * PANEL;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < height * (end - begin);
         index += blockDim.x * gridDim.x)
    {
        int row = index % height;
        int column = index / height;
        auto value = Load(matrix, static_cast<uint64_t>(begin + column) * shape.lda + begin + row);
        workspace[column * ld + row] = value.real;
        workspace[plane + column * ld + row] = value.imag;
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(512) inline void StoreDistributedPanelVf(ComplexSolverShape shape,
                                                                             __gm__ aclFloatComplex* matrix,
                                                                             __gm__ float* workspace, int begin,
                                                                             int end, int ld)
{
    int height = shape.m - begin;
    int plane = ld * PANEL;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < height * (end - begin);
         index += blockDim.x * gridDim.x)
    {
        int row = index % height;
        int column = index / height;
        Store(matrix, static_cast<uint64_t>(begin + column) * shape.lda + begin + row,
              {workspace[column * ld + row], workspace[plane + column * ld + row]});
    }
}
#endif

__global__ __aicore__ void complex_load_distributed_panel(ComplexSolverShape shape, __gm__ uint8_t* matrix,
                                                          __gm__ uint8_t* workspace, int begin, int end, int ld)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<LoadDistributedPanelVf>(dim3{512, 1, 1}, shape, reinterpret_cast<__gm__ aclFloatComplex*>(matrix),
                                        reinterpret_cast<__gm__ float*>(workspace), begin, end, ld);
#endif
}

__global__ __aicore__ void complex_store_distributed_panel(ComplexSolverShape shape, __gm__ uint8_t* matrix,
                                                           __gm__ uint8_t* workspace, int begin, int end, int ld)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
#if __NPU_ARCH__ == 3510
    asc_vf_call<StoreDistributedPanelVf>(dim3{512, 1, 1}, shape, reinterpret_cast<__gm__ aclFloatComplex*>(matrix),
                                         reinterpret_cast<__gm__ float*>(workspace), begin, end, ld);
#endif
}

void ComplexSolverDistributedPanel(ComplexSolverShape shape, void* a, void* work, void* pivot, void* status, int begin,
                                   int end, uint32_t blocks, void* stream)
{
    auto* matrix = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(a));
    auto* workspace = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(work));
    auto* pivots = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(pivot));
    auto* info = reinterpret_cast<__gm__ uint8_t*>(reinterpret_cast<uintptr_t>(status));
    int height = shape.m - begin;
    blocks = ComplexSolverDistributedBlocks(height, blocks);
    if (blocks == 0) return;
    int ld = (height + 7) / 8 * 8;
    int rowsPerCore = (height + blocks - 1) / blocks;
    rowsPerCore = (rowsPerCore + 7) / 8 * 8;
    uint32_t panelBlocks = (height + rowsPerCore - 1) / rowsPerCore;
    complex_load_distributed_panel<<<blocks, nullptr, stream>>>(shape, matrix, workspace, begin, end, ld);
    complex_factor_distributed_panel<<<panelBlocks, nullptr, stream>>>(workspace, pivots, info, shape.m, shape.n, ld,
                                                                       COMPLEX_SOLVER_PANEL, begin, end - begin,
                                                                       rowsPerCore, pivots != nullptr);
    complex_store_distributed_panel<<<blocks, nullptr, stream>>>(shape, matrix, workspace, begin, end, ld);
}
