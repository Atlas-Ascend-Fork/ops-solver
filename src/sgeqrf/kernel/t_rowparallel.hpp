/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * 块反射子三角因子 T 的递推：**按行分核、核间零通信**（设备侧专用）
 *
 * 推导（FORWARD）：H_1...H_k = I - V T V^T，LAPACK slarft 的递推是
 *     T(1:i-1, i) = T(1:i-1, 1:i-1) * ( -tau_i * V(:, 1:i-1)^T V(:, i) )
 *     T(i, i)     = tau_i
 * 记 G = V^T V，逐元素展开就是
 *     T(r, i) = sum_{c=r..i-1} T(r, c) * ( -tau_i * G(c, i) )
 * 右边只用到**第 r 行自己的历史值** + 全核共享的只读 G / tau
 * ⇒ 行与行之间无依赖，按行分核后核间零通信、零屏障。
 * 关键路径 = 第 0 行的 k^2/2 次 FMA。
 *
 * 绝不要退回"单核 O(k^3) 标量三重循环"：实测那样在 ormqr 2048^3 上占 81%（nb=128 时 91%）。
 *
 * BACKWARD（H_k...H_1，T 下三角）是镜像：i 递减，
 *     T(r, i) = sum_{c=i+1..r} T(r, c) * ( -tau_i * G(c, i) )，T(i,i) = tau_i
 * 同样只读第 r 行自己的历史值（那些列的 i 更大，先算过）。
 *
 * 本文件只在 kernel 里用（依赖 AscendC），host 侧的参考实现在 householder.h 的 Larft。
 */
#ifndef QR_CORE_T_ROWPARALLEL_H
#define QR_CORE_T_ROWPARALLEL_H

#ifdef __CCE_AICORE__

namespace QrCore {

/// 本核拥有的行数（行按 coreId 轮转分配：第 r 行的工作量是 (k-r)(k-r-1)/2，差异极大，
/// 轮转比连续切块均衡得多）
__aicore__ inline uint32_t TRowsOfCore(uint32_t k, uint32_t coreId, uint32_t coreNum)
{
    if (coreId >= k || coreNum == 0) {
        return 0;
    }
    return (k - coreId + coreNum - 1u) / coreNum;
}

/// T 的一个元素：T(r,i) = sum_c T(r,c) * (-tau_i * G(c,i))，c 的范围由 direct 决定。
/// StreamG 时 gl 里是 G 的**第 i 行**（靠 G 对称取行代替跨步取列），下标因此退化成 c。
template <bool Backward, bool StreamG>
__aicore__ inline float TRecurrenceSum(const AscendC::LocalTensor<float>& tl, uint32_t ldtUb,
                                       const AscendC::LocalTensor<float>& gl, uint32_t ldg, uint32_t q, uint32_t r,
                                       uint32_t i, float taui)
{
    float sum = 0.0f;
    if (Backward) {
        for (uint32_t c = i + 1u; c <= r; ++c) {
            const float g = StreamG ? gl.GetValue(c) : gl.GetValue(c * ldg + i);
            sum += tl.GetValue(q * ldtUb + c) * (-taui * g);
        }
    } else {
        for (uint32_t c = r; c < i; ++c) {
            const float g = StreamG ? gl.GetValue(c) : gl.GetValue(c * ldg + i);
            sum += tl.GetValue(q * ldtUb + c) * (-taui * g);
        }
    }
    return sum;
}

/**
 * 算出本核负责的那几行 T，紧凑放在 tl 里（第 q 行起点 q*ldtUb，行内按列连续）。
 *
 * 结果是 **+T**（LAPACK 约定）。LARFB 要的是 -T，由调用方自己 Muls(-1)。
 * tau_i == 0 时整列留 0（含对角），与 LAPACK slarft 的 H(i)=I 分支等价。
 *
 * @tparam Backward false = FORWARD（上三角，i 递增）；true = BACKWARD（下三角，i 递减）
 * @tparam StreamG  false：gl 里已经是整块 G（k x k 行主序，行距 ldg），一次性读进 UB —— k 小时
 *                        指令最省。
 *                  true ：gl 只是一个长度 >= k 的暂存；每个 i 从 gGm 现取**第 i 行**。
 *                        这里用的是 G = V^T V 的**对称性**（G(c,i) == G(i,c)）：递推要的是
 *                        "第 i 列"，而列在行主序里是跨步的 —— DataCopyPad 的每个块独占 32B，
 *                        4 字节的块会退化成逐元素搬，所以必须换成等价的"第 i 行"连续读。
 *                        k=256 时整块 G 要 256 KB，放不进单核 UB，只能走这条路。
 * @param ldg   StreamG=false 时是 G 在 UB 里的行距；true 时是 G 在 GM 里的行距
 * @return 本核拥有的行数
 */
template <bool Backward, bool StreamG>
__aicore__ inline uint32_t BuildTRowParallel(const AscendC::LocalTensor<float>& tl, uint32_t ldtUb,
                                             const AscendC::LocalTensor<float>& gl, uint32_t ldg,
                                             const AscendC::GlobalTensor<float>& gGm,
                                             const AscendC::LocalTensor<float>& taul, uint32_t k, uint32_t coreId,
                                             uint32_t coreNum)
{
    const uint32_t myRows = TRowsOfCore(k, coreId, coreNum);
    if (myRows > 0) {
        AscendC::Duplicate(tl, 0.0f, myRows * ldtUb);
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    for (uint32_t t = 0; t < k; ++t) {
        const uint32_t i = Backward ? (k - 1u - t) : t;
        const float taui = taul.GetValue(i);
        if (StreamG) {
            AscendC::DataCopyExtParams cp{1, static_cast<uint32_t>(k * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<float> pd{false, 0, 0, 0};
            AscendC::DataCopyPad(gl, gGm[static_cast<uint64_t>(i) * ldg], cp, pd);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(EVENT_ID0);
        }
        uint32_t q = 0;
        for (uint32_t r = coreId; r < k; r += coreNum, ++q) {
            if (Backward) {
                if (r < i) {
                    continue;
                }
            } else {
                if (r > i) {
                    break;
                }
            }
            if (r == i) {
                tl.SetValue(q * ldtUb + i, taui);
                continue;
            }
            if (taui == 0.0f) {
                continue;  // 整列留 0
            }
            tl.SetValue(q * ldtUb + i, TRecurrenceSum<Backward, StreamG>(tl, ldtUb, gl, ldg, q, r, i, taui));
        }
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    return myRows;
}

}  // namespace QrCore

#endif  // __CCE_AICORE__
#endif  // QR_CORE_T_ROWPARALLEL_H
