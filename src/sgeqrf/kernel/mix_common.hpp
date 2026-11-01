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
 * \file mix_common.hpp
 * \brief sgeqrf / sorgqr / sormqr / xlarft / sgels 五个 MIX kernel 的公共件。
 *
 * 五个算子走同一条 LARFB 数据通路，因而共用：cube 侧的类型别名与 BlockMmad 实例、
 * AIC/AIV 之间的六个跨核同步、分 K 的份数与块长决策、按 plan 下发 cube 的两个入口、
 * 以及分 K partial 的定序归约。
 *
 * 这些件以前在五个 mix 头里各有一份副本。公式只在一处的后果是实测踩过的：
 * 份数公式在 cube 侧与 vec 侧各写一份时会漂移，症状是归约漏掉一份 partial
 * （tile 够多时看不出来，只在小形状上错）。要改就改这里。
 */

#ifndef ACLSOLVER_QR_MIX_COMMON_HPP
#define ACLSOLVER_QR_MIX_COMMON_HPP

// CATLASS 的 tiling 宏要在它自己的头之前定好
#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"

#include "householder.hpp"
#include "larfb_plan.hpp"
#include "t_rowparallel.hpp"

// ---- cube 侧类型 ----
// ★ 本文件（以及 *_mix.hpp）只被 src/<op>/arch35/<op>_kernel.cpp 引入，
//   因此可以直接钉死 arch35 的 ArchTag；其他 SOC 不会编到这里。
//   CATLASS 头也只在这条路径上需要 ⇒ 默认 SOC（ascend910b）构建不依赖 CATLASS_HOME。
using namespace Catlass;
using namespace tla;
using Element = float;
using ArchTag = Arch::Ascend950;
using Policy = Gemm::MmadPingpong<ArchTag, true, false, 1, false, 2, 2, 2, 2>;
using L1TileShape = Shape<Int<256>, Int<256>, Int<128>>;
using L0TileShape = Shape<Int<256>, Int<256>, Int<32>>;

template <class TagA, class TagB>
using MmadOf = Gemm::Block::BlockMmadTla<
    Policy, L1TileShape, L0TileShape, Element, Element, Element, void,
    Gemm::Tile::PackedTileCopyTla<ArchTag, Element, TagA, Element, TagB, Element, layout::RowMajor, void>>;
using MmadRR = MmadOf<layout::RowMajor, layout::RowMajor>;
using MmadRC = MmadOf<layout::RowMajor, layout::ColumnMajor>;

// ---- 公共上限与分段 ----
/// kernel 支持的最大行数（host 侧已拦住更大的形状）
constexpr uint32_t MAX_ROWS = 4096;
constexpr uint32_t RED_CAP = 2048;  // partial 归约的 UB 分段长度

constexpr uint32_t kL1M = tla::get<0>(L1TileShape{});
constexpr uint32_t kL1N = tla::get<1>(L1TileShape{});
constexpr uint32_t kL1K = tla::get<2>(L1TileShape{});
constexpr uint32_t KSPLIT_MAX = 32;

// ---- 对齐与上取整（除数为 0 时返回原值，不做除法）----
__aicore__ inline uint32_t AlignUp32(uint32_t x, uint32_t a)
{
    return (a == 0U) ? x : (x + a - 1U) / a * a;
}

__aicore__ inline uint32_t CeilDivU(uint32_t x, uint32_t a) { return (a == 0) ? 0 : (x + a - 1) / a; }

// ---- AIC / AIV 跨核同步 ----
//
// 三个 flag 号的分工（与 CrossCoreSetFlag 的 mode 不是一回事）：
//   mode 0x0 = **同类**核之间会合（AIV↔AIV 或 AIC↔AIC），mode 0x2 = **跨类**通知。
//   flag 0x0：同类会合用（VecBarrier / CubeBarrier）
//   flag 0x1：AIV -> AIC（VecNotifyCube 置位，CubeWaitVec 等待）
//   flag 0x2：AIC -> AIV（CubeNotifyVec 置位，VecWaitCube 等待）
//
// 逐块（面板）的配对形状，五个 kernel 一致：
//   AIV 侧： …算完本块要给 cube 的数据… -> VecBarrier() -> VecNotifyCube() -> VecWaitCube()
//   AIC 侧： CubeWaitVec() -> …cube 的活… -> CubeNotifyVec()
// 即每轮恰好 1 次 0x1 的置位/等待 + 1 次 0x2 的置位/等待。分 K 时中段多一对
// （cube 发布 partial -> vec 归约 -> vec 放行 cube 做 lp[1]），见 LarfbReduceVec：
// 它内部的 VecWaitCube/VecNotifyCube 与 cube 侧 RunPlanOnCubeSplitK 之后那对严格对应，
// 且**两侧都由同一个 wChunks > 1 判定**，所以要么都做要么都不做。
//
// ★ 为什么任一侧都不会"提前退出"把对端吊死（这是重构时最容易破坏的性质）：
//   1. **控制流只由形状决定，不由数值决定**。kernel 从不**读** devInfo（ZeroDevInfo 只写），
//      也没有任何依赖计算结果的分支 ⇒ 不存在"info > 0 就早退"这种路径。
//   2. **循环上界两侧同源**：nBlocks / kmax 由同一个表达式从同一份 params 算出（同一个
//      翻译单元为两种核各编一次），AIV 与 AIC 的迭代次数恒等。
//      （反例见 ascendc-operator-perf「换个 blockDim 就挂死」：两侧各自推导同一个几何量。）
//   3. **空问题根本不下发**：host 对 m/n/k == 0 直接返回，所以进到 kernel 时 nBlocks >= 1。
//   4. 确实存在的提前 return（RunPlanOnCube 的 plan.m/n/k == 0、GeqrfTrailingCube 的
//      trail <= 0、ReducePartials 的 beg >= elems）**只跳过同类 barrier**，而同类 barrier
//      由该类所有核按同一个形状条件一起跳；跨类的 Notify 一律放在这些函数的**调用点**、
//      不在函数体内，因此永远会执行到。
//   ⟹ 改动这一段时请保持"跨类 Notify 留在调用点"这条约定；把它挪进带提前 return 的
//      函数里，就是上面第 4 条的反面（曾经发生过一次，症状是测试跑几十分钟不出结果）。
__aicore__ inline void CubeBarrier() { Arch::CrossCoreBarrier<0x0, PIPE_FIX>(); }
__aicore__ inline void VecBarrier()
{
    AscendC::CrossCoreSetFlag<0x0, PIPE_MTE3>(0x0);
    AscendC::CrossCoreWaitFlag(0x0);
}
__aicore__ inline void VecNotifyCube() { AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(0x1); }
__aicore__ inline void CubeWaitVec() { AscendC::CrossCoreWaitFlag(0x1); }
__aicore__ inline void CubeNotifyVec() { AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x2); }
__aicore__ inline void VecWaitCube() { AscendC::CrossCoreWaitFlag(0x2); }

// ---- 分 K 的份数与块长 ----
/// 分 K 的份数。**cube 侧与 vec 侧都调这一个函数**：两侧各写一份公式会漂移，
/// 症状是归约漏掉一份 partial（大形状 tile 够多、S=1 时看不出来，小形状才错）。
/// 规则：让 tiles*S 填满 AIC；每份至少 L1K 行；再被 host 给的容量 clamp。
__aicore__ inline uint32_t KSplitCount(uint32_t pm, uint32_t pn, uint32_t pk, uint32_t aicNum, uint64_t capElems)
{
    const uint64_t one = static_cast<uint64_t>(pm) * static_cast<uint64_t>(pn);
    if (one == 0 || pk == 0 || aicNum == 0) { return 1; }
    const uint32_t tiles = CeilDivU(pm, kL1M) * CeilDivU(pn, kL1N);
    if (tiles == 0) { return 1; }
    uint32_t smax = KSPLIT_MAX;
    const uint32_t byK = CeilDivU(pk, kL1K);  // 每份至少 L1K 行，否则白付固定开销
    if (smax > byK) { smax = byK; }
    // host 给的容量是硬上限。one 在上面已判过非 0，这里就地再判一次：
    // 静态检查不做跨语句的常量传播，守卫必须和除法在同一个表达式里。
    const uint32_t byCap = (one == 0) ? KSPLIT_MAX : static_cast<uint32_t>(capElems / one);
    if (smax > byCap) { smax = byCap; }
    if (smax <= 1) { return 1; }
    // 选让"最忙核的轮数 / 份数"最小的 S，也就是让 tiles*S 尽量正好铺满 AIC 的整数轮。
    // ★ 不能用 S = ceil(AIC/tiles)：tiles=16、AIC=28 时它给 S=2 ⇒ 32 份要跑 2 轮，
    //   makespan 和 S=1 的 1 轮**完全一样**，白付一次归约（实测 4096^3 ormqr 只快 4%）。
    //   最优是 S=7（112 = 4*28 正好铺满）⇒ 1.75×。
    // 比较用整数交叉相乘，取严格更优才换 ⇒ 平手时保留更小的 S（开销更小）。
    uint32_t bestS = 1;
    uint32_t bestRounds = CeilDivU(tiles, aicNum);
    for (uint32_t sp = 2; sp <= smax; ++sp) {
        const uint32_t r = CeilDivU(tiles * sp, aicNum);
        if (r * bestS < bestRounds * sp) {
            bestS = sp;
            bestRounds = r;
        }
    }
    return bestS;
}

/// 每份的 K 长度（按 L1K 对齐，保证分形不被切碎）
__aicore__ inline uint32_t KSplitChunkLen(uint32_t pk, uint32_t sp)
{
    return CeilDivU(CeilDivU(pk, sp), kL1K) * kL1K;
}

/// 份数与每份长度一起算出来（**调用点不要再推一遍 chunk 长度**：份数是由长度反推的，
/// 两处各算一次有机会给出不一致的一对，表现是 partial 漏算或越界）
__aicore__ inline uint32_t KSplitChunks(uint32_t pm, uint32_t pn, uint32_t pk, uint32_t aicNum, uint64_t capElems,
                                        uint32_t* chunkLen)
{
    const uint32_t sp = KSplitCount(pm, pn, pk, aicNum, capElems);
    if (sp <= 1) {
        *chunkLen = pk;
        return 1;
    }
    const uint32_t c = KSplitChunkLen(pk, sp);
    *chunkLen = c;
    return CeilDivU(pk, c);
}

// ---- 按 plan 下发 cube ----
/// 注意：TagA/TagB 是编译期模板参数，**不读** plan 里的 aColMajorView/bColMajorView，
/// 两者必须由调用点手工对齐，否则改了 plan 却毫无效果。
template <class Mmad, class TagA, class TagB>
__aicore__ inline void RunPlanOnCube(QrCore::GemmPlan const& plan, GM_ADDR pa, GM_ADDR pb, GM_ADDR pd)
{
    if (plan.m == 0 || plan.n == 0 || plan.k == 0) { return; }
    Arch::Resource<ArchTag> resource;
    Mmad mmad(resource);
    TagA tagA(static_cast<uint32_t>(plan.m), static_cast<uint32_t>(plan.k), static_cast<int64_t>(plan.lda));
    TagB tagB(static_cast<uint32_t>(plan.k), static_cast<uint32_t>(plan.n), static_cast<int64_t>(plan.ldb));
    layout::RowMajor tagD(static_cast<uint32_t>(plan.m), static_cast<uint32_t>(plan.n),
                          static_cast<int64_t>(plan.ldd));
    AscendC::GlobalTensor<Element> gmA, gmB, gmD;
    gmA.SetGlobalBuffer((__gm__ Element*)pa + plan.aOffset);
    gmB.SetGlobalBuffer((__gm__ Element*)pb + plan.bOffset);
    gmD.SetGlobalBuffer((__gm__ Element*)pd + plan.dOffset);
    auto tA = tla::MakeTensor(gmA, tla::MakeLayoutFromTag(tagA), Arch::PositionGM{});
    auto tB = tla::MakeTensor(gmB, tla::MakeLayoutFromTag(tagB), Arch::PositionGM{});
    auto tD = tla::MakeTensor(gmD, tla::MakeLayoutFromTag(tagD), Arch::PositionGM{});
    constexpr uint32_t L1M = tla::get<0>(L1TileShape{});
    constexpr uint32_t L1N = tla::get<1>(L1TileShape{});
    constexpr uint32_t L1K = tla::get<2>(L1TileShape{});
    GemmCoord shape{static_cast<uint32_t>(plan.m), static_cast<uint32_t>(plan.n), static_cast<uint32_t>(plan.k)};
    Gemm::Block::GemmIdentityBlockSwizzle<3, 0> sched(shape, MakeCoord(L1M, L1N));
    uint32_t loops = sched.GetCoreLoops();
    if (plan.accumulate) { AscendC::SetAtomicAdd<Element>(); }
    for (uint32_t i = AscendC::GetBlockIdx(); i < loops; i += AscendC::GetBlockNum()) {
        GemmCoord bc = sched.GetBlockCoord(i);
        GemmCoord act = sched.GetActualBlockShape(bc);
        auto ta = GetTile(tA, tla::MakeCoord(bc.m() * L1M, bc.k() * L1K), tla::MakeShape(act.m(), act.k()));
        auto tb = GetTile(tB, tla::MakeCoord(bc.k() * L1K, bc.n() * L1N), tla::MakeShape(act.k(), act.n()));
        auto td = GetTile(tD, tla::MakeCoord(bc.m() * L1M, bc.n() * L1N), tla::MakeShape(act.m(), act.n()));
        mmad(ta, tb, td, act);
    }
    if (plan.accumulate) { AscendC::SetAtomicNone(); }
    AscendC::PipeBarrier<PIPE_ALL>();
}

/// 分 K 版：输出不累加，而是每份 K 块各写一份完整的 (m x n) partial（行主序 ld=n，
/// 第 s 份在 pd + s*m*n），之后由 AIV 定序归约。
/// **不用 SetAtomicAdd**：原子加的到达顺序不定 ⇒ fp32 和每趟都可能不同，而任务书
/// §3.5 要求同输入逐位可复现。
/// 要求 plan.dOffset == 0 且 plan.accumulate == false（调用点保证）。
template <class Mmad, class TagA, class TagB>
__aicore__ inline void RunPlanOnCubeSplitK(QrCore::GemmPlan const& plan, GM_ADDR pa, GM_ADDR pb, GM_ADDR pd,
                                           uint32_t nChunks, uint32_t kc)
{
    if (plan.m == 0 || plan.n == 0 || plan.k == 0) { return; }
    Arch::Resource<ArchTag> resource;
    Mmad mmad(resource);
    TagA tagA(static_cast<uint32_t>(plan.m), static_cast<uint32_t>(plan.k), static_cast<int64_t>(plan.lda));
    TagB tagB(static_cast<uint32_t>(plan.k), static_cast<uint32_t>(plan.n), static_cast<int64_t>(plan.ldb));
    layout::RowMajor tagD(static_cast<uint32_t>(plan.m), static_cast<uint32_t>(plan.n),
                          static_cast<int64_t>(plan.n));
    AscendC::GlobalTensor<Element> gmA, gmB, gmD;
    gmA.SetGlobalBuffer((__gm__ Element*)pa + plan.aOffset);
    gmB.SetGlobalBuffer((__gm__ Element*)pb + plan.bOffset);
    gmD.SetGlobalBuffer((__gm__ Element*)pd);
    auto tA = tla::MakeTensor(gmA, tla::MakeLayoutFromTag(tagA), Arch::PositionGM{});
    auto tB = tla::MakeTensor(gmB, tla::MakeLayoutFromTag(tagB), Arch::PositionGM{});
    auto tD = tla::MakeTensor(gmD, tla::MakeLayoutFromTag(tagD), Arch::PositionGM{});
    GemmCoord shape{static_cast<uint32_t>(plan.m), static_cast<uint32_t>(plan.n), static_cast<uint32_t>(plan.k)};
    Gemm::Block::GemmIdentityBlockSwizzle<3, 0> sched(shape, MakeCoord(kL1M, kL1N));
    const uint32_t tiles = sched.GetCoreLoops();
    if (tiles == 0) { return; }  // 没有 tile 时下面的 i / tiles 不成立
    const uint32_t loops = tiles * nChunks;
    const uint64_t one = static_cast<uint64_t>(plan.m) * static_cast<uint64_t>(plan.n);
    for (uint32_t i = AscendC::GetBlockIdx(); i < loops; i += AscendC::GetBlockNum()) {
        const uint32_t sIdx = i / tiles;
        const uint32_t tIdx = i - sIdx * tiles;
        const uint32_t k0 = sIdx * kc;
        if (k0 >= static_cast<uint32_t>(plan.k)) { continue; }
        uint32_t klen = static_cast<uint32_t>(plan.k) - k0;
        if (klen > kc) { klen = kc; }
        GemmCoord bc = sched.GetBlockCoord(tIdx);
        GemmCoord act = sched.GetActualBlockShape(bc);
        auto ta = GetTile(tA, tla::MakeCoord(bc.m() * kL1M, k0), tla::MakeShape(act.m(), klen));
        auto tb = GetTile(tB, tla::MakeCoord(k0, bc.n() * kL1N), tla::MakeShape(klen, act.n()));
        auto tDs = tla::MakeTensor(gmD[static_cast<int64_t>(sIdx) * static_cast<int64_t>(one)],
                                   tla::MakeLayoutFromTag(tagD), Arch::PositionGM{});
        auto td = GetTile(tDs, tla::MakeCoord(bc.m() * kL1M, bc.n() * kL1N), tla::MakeShape(act.m(), act.n()));
        GemmCoord actK{act.m(), act.n(), klen};
        mmad(ta, tb, td, actK);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ---- 以下只在 AIC 编译（这段会实例化 cube 模板）----
#ifdef __DAV_CUBE__
/// G = V^T V（cube 侧），算完就把结果发布给 vec。
///
/// gp[0] 的行主序视图是 (m=jb, n=jb, k=rows)：M/N 方向只有 1 个 tile ⇒ 不分 K
/// 就只有 1 个核在串着读 2*jb*rows 个元素。rows=4096 时实测约 20 us/块。
/// orgqr 与 ormqr 的这一步逐字相同（只有 params 结构体类型不同），所以收在这里，
/// 两边各留一个取字段的包装。
__aicore__ inline void GemmGBlock(GM_ADDR wsY, GM_ADDR wsG, GM_ADDR wsP, uint32_t rows, uint32_t jb,
                                  uint32_t ldy, uint32_t gChunks, uint32_t gChunkLen)
{
    QrCore::GemmPlan gp[3];
    QrCore::PlanLarfbLeft(rows, jb, jb, ldy, ldy, jb, jb, jb, 0, 0, gp);
    if (gChunks > 1) {
        RunPlanOnCubeSplitK<MmadRC, layout::RowMajor, layout::ColumnMajor>(gp[0], wsY, wsY, wsP, gChunks,
                                                                          gChunkLen);
    } else {
        RunPlanOnCube<MmadRC, layout::RowMajor, layout::ColumnMajor>(gp[0], wsY, wsY, wsG);
    }
    CubeBarrier();
    CubeNotifyVec();
}
#endif  // __DAV_CUBE__

// ---- 以下只在 AIV 编译（cube 侧没有向量流水）----
#ifdef __DAV_VEC__
/// devInfo 置 0 交给 kernel 自己做。host 侧那条 aclrtMemsetAsync(4B) 是**流上一条
/// 独立任务**，实测 min 8.9 us —— 小形状 25 us 的启动地板里 36% 是它。
///
/// 借一块 float 缓冲写：0.0f 的位型就是 0x00000000，即 int32 的 0。
/// 只有 0 号核做；devInfo 允许为空（Xlarft 这类没有 devInfo 的接口传 nullptr）。
__aicore__ inline void ZeroDevInfo(GM_ADDR devInfo, const AscendC::LocalTensor<float>& scratch, uint32_t coreId)
{
    if (devInfo == nullptr || coreId != 0) {
        return;
    }
    AscendC::GlobalTensor<float> infoT;
    infoT.SetGlobalBuffer((__gm__ float*)devInfo);
    AscendC::Duplicate(scratch, 0.0f, 8);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::DataCopyExtParams cpI{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(infoT, scratch, cpI);
    AscendC::PipeBarrier<PIPE_ALL>();
}

/// 物化一个 V 块到 wsY：对角隐含的 1 写上，对角以上置 0。
///
/// UB 张量偏移一律为 0；非对齐的一侧交给 DataCopyPad 在 GM 上处理。
/// src 是列主序的反射子来源（orgqr 用 A、ormqr 用 V），ldSrc 是它的行距。
__aicore__ inline void MaterializeVBlock(const AscendC::GlobalTensor<float>& src,
                                         const AscendC::GlobalTensor<float>& Y,
                                         AscendC::TQue<AscendC::QuePosition::VECIN, 1>& qIn,
                                         const AscendC::LocalTensor<float>& work, uint32_t j, uint32_t jb,
                                         uint32_t rows, uint32_t ldSrc, uint32_t ldy, uint32_t coreId,
                                         uint32_t coreNum)
{
    for (uint32_t c = coreId; c < jb; c += coreNum) {
        const uint32_t len = rows - c;
        if (c > 0) {
            AscendC::Duplicate(work, 0.0f, AlignUp32(c, 8));
            AscendC::PipeBarrier<PIPE_ALL>();
            AscendC::DataCopyExtParams cpz{1, static_cast<uint32_t>(c * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(Y[static_cast<uint64_t>(c) * ldy], work, cpz);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
        AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
        AscendC::LocalTensor<float> cc = qIn.AllocTensor<float>();
        AscendC::DataCopyPad(cc, src[static_cast<uint64_t>(j + c) * ldSrc + (j + c)], cp, pd);
        qIn.EnQue(cc);
        cc = qIn.DeQue<float>();
        cc.SetValue(0, 1.0f);  // 对角隐含为 1
        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(Y[static_cast<uint64_t>(c) * ldy + c], cc, cp);
        qIn.FreeTensor(cc);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
}

/// 一个块的 T 三角递推，写出 **-T**（LARFB 要的是 -T，共用实现给的是 LAPACK 约定的 +T）。
///
/// 推导：T(r,i) = sum_{c=r..i-1} T(r,c) * w(c)，w(c) = -tau_i * G(c,i)
/// 只用到**第 r 行自己的历史值** + 全核共享的只读 G/tau ⇒ 行与行之间无依赖，
/// 按行分核后核间零通信、零屏障。原来那版是单核 O(jb^3) 标量三重循环，
/// 实测在 2048^3 上占 81%（nb=128 时 91%）。
///
/// 关键路径 = 第 0 行的 jb^2/2 次 FMA（jb=64 时 2016 次）。行用轮转分配以均衡负载
/// （第 r 行的工作量 = (jb-r)(jb-r-1)/2，差异极大）。
///
/// UB 行距取 AlignUp32(jb,8)：用 jb 本身时，jb 非 8 倍数会让后面 DataCopyPad 的
/// UB 源偏移不对齐 ⇒ **静默什么都不搬**（小 jb 才暴露）。
/// T 是 jb x jb 列主序（ldt = jb），本核的"行 r"在列主序里是跨步的，所以逐行写。
__aicore__ inline void BuildMinusTBlock(const AscendC::GlobalTensor<float>& T,
                                        const AscendC::GlobalTensor<float>& G,
                                        const AscendC::GlobalTensor<float>& tau,
                                        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& qG,
                                        AscendC::TQue<AscendC::QuePosition::VECIN, 1>& qTau,
                                        const AscendC::LocalTensor<float>& tl, uint32_t j, uint32_t jb,
                                        uint32_t coreId, uint32_t coreNum)
{
    AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
    AscendC::LocalTensor<float> gl = qG.AllocTensor<float>();
    AscendC::DataCopyExtParams cpG{1, static_cast<uint32_t>(jb * jb * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(gl, G, cpG, pd);
    qG.EnQue(gl);
    gl = qG.DeQue<float>();
    AscendC::LocalTensor<float> taul = qTau.AllocTensor<float>();
    AscendC::DataCopyExtParams cpTau{1, static_cast<uint32_t>(jb * sizeof(float)), 0, 0, 0};
    AscendC::DataCopyPad(taul, tau[j], cpTau, pd);
    qTau.EnQue(taul);
    taul = qTau.DeQue<float>();

    const uint32_t ldtUb = AlignUp32(jb, 8);
    const uint32_t myRows =
        QrCore::BuildTRowParallel<false, false>(tl, ldtUb, gl, jb, G, taul, jb, coreId, coreNum);
    if (myRows > 0) {
        AscendC::Muls(tl, tl, -1.0f, myRows * ldtUb);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    qG.FreeTensor(gl);
    qTau.FreeTensor(taul);
    uint32_t q = 0;
    for (uint32_t r = coreId; r < jb; r += coreNum, ++q) {
        AscendC::DataCopyExtParams cpR{1, static_cast<uint32_t>(jb * sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPad(T[static_cast<uint64_t>(r) * jb], tl[q * ldtUb], cpR);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ---- 分 K partial 的定序归约 ----
/// 把 nChunks 份 partial 定序累加进 dst（elems 个元素）。按核切连续片，片内分段过 UB。
/// 顺序固定（s 从 0 到 nChunks-1、片由 coreId 决定）⇒ 结果逐位可复现。
__aicore__ inline void ReducePartials(AscendC::GlobalTensor<float> dst, AscendC::GlobalTensor<float> src,
                                      uint64_t elems, uint32_t nChunks, AscendC::LocalTensor<float> acc,
                                      AscendC::LocalTensor<float> tmp, uint32_t cap, uint32_t coreId,
                                      uint32_t coreNum)
{
    const uint64_t per = AlignUp32(CeilDivU(static_cast<uint32_t>(elems), static_cast<uint32_t>(coreNum)), 8U);
    const uint64_t beg = static_cast<uint64_t>(coreId) * per;
    if (beg >= elems) { return; }
    uint64_t len = elems - beg;
    if (len > per) { len = per; }
    for (uint64_t off = 0; off < len; off += cap) {
        uint32_t cnt = static_cast<uint32_t>(len - off);
        if (cnt > cap) { cnt = cap; }
        AscendC::DataCopyExtParams cp{1, cnt * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
        AscendC::DataCopyPad(acc, src[beg + off], cp, pd);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        for (uint32_t sIdx = 1; sIdx < nChunks; ++sIdx) {
            AscendC::DataCopyPad(tmp, src[static_cast<uint64_t>(sIdx) * elems + beg + off], cp, pd);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::Add(acc, acc, tmp, cnt);
            // V -> MTE2 的 WAR：下一趟 DataCopyPad 会覆写 tmp，必须等这次 Add 读完。
            // PipeBarrier<PIPE_V> 只排 V 与 V，挡不住 MTE2（nChunks>=3 才暴露：
            // 512^3 上 max|Q_dev-Q_cpu| 直接 0.41）
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        }
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopyPad(dst[beg + off], acc, cp);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }
}
/// LARFB 第一次 GEMM 分 K 时，vec 侧把 partial 定序归约掉再放 cube 继续。
/// 不分 K 时整段是空操作（一次 flag 都不动）。
///
/// 这里的 Wait/Notify 必须与 cube 侧 lp[0] 之后那对 Notify/Wait 一一对应。
/// orgqr 与 ormqr 的这一步逐字相同；orgqr 还要在之后再等一次"本块做完"，
/// 那次等待留在它自己的包装里，跨核配对不变。
__aicore__ inline void LarfbReduceVec(const AscendC::GlobalTensor<float>& W,
                                      const AscendC::GlobalTensor<float>& P, const QrCore::GemmPlan* lp,
                                      const AscendC::LocalTensor<float>& work,
                                      const AscendC::LocalTensor<float>& red, uint32_t wChunks, uint32_t coreId,
                                      uint32_t coreNum)
{
    if (wChunks > 1) {
        VecWaitCube();  // partial 齐了
        ReducePartials(W, P, static_cast<uint64_t>(lp[0].m) * lp[0].n, wChunks, work, red, RED_CAP, coreId,
                       coreNum);
        VecBarrier();
        VecNotifyCube();  // 归约完，cube 可以做 lp[1]
    }
}

#endif  // __DAV_VEC__

#endif  // ACLSOLVER_QR_MIX_COMMON_HPP
