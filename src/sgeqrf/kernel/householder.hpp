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
 * Householder 标量层与面板分解（纯算术，不依赖 ACL / AscendC，可本地 g++ 自验）
 *
 * 约定逐字对齐 LAPACK（这是精度判据成立的前提：§3.2.1-1 要求把 NPU 输出的 (v, tau)
 * 升到 float64 后喂 CPU 的 dorgqr 构造 Q）：
 *
 *   给定 x = [alpha; x2]，取
 *       beta = -sign(alpha) * ||x||          （sign(0) 取 +1）
 *       tau  = (beta - alpha) / beta
 *       v2   = x2 / (alpha - beta)，v0 恒为 1（不显式存储）
 *   则  H = I - tau * v * v^T  满足  H * x = [beta; 0]
 *
 * 退化情形：||x2|| == 0 时 tau = 0、H = I（LAPACK 同款，不做符号翻转）。
 *
 * T 的递推（forward / columnwise，对齐 LAPACK slarft）：
 *       T[i,i]     = tau_i
 *       T[0:i, i]  = -tau_i * T[0:i, 0:i] * (V[:, 0:i]^T * V[:, i])
 */
#ifndef QR_CORE_HOUSEHOLDER_H
#define QR_CORE_HOUSEHOLDER_H

#include <cmath>
#include <limits>

namespace QrCore {

/// 列主序取元素
template <class T>
inline T& ColMajorAt(T* base, int ld, int r, int c)
{
    return base[static_cast<long long>(c) * ld + r];
}
template <class T>
inline const T& ColMajorAt(const T* base, int ld, int r, int c)
{
    return base[static_cast<long long>(c) * ld + r];
}

/// Larfg 的定标兜底（只在范数溢出 / 下溢时调用，判据见 Larfg）。
/// 做法同 LAPACK 的 snrm2：先取 |alpha| 与 |x2| 的最大值 cmax，按它缩放后再求范数，
/// 最后把 beta 乘回 cmax。anyBelow = false 表示对角以下确实全零（H = I）。
template <class T>
inline void LarfgScaled(int n, T& alpha, T* x2, T& tau, bool anyBelow)
{
    if (!anyBelow) {
        tau = T(0);
        return;
    }
    T cmax = (alpha < T(0)) ? -alpha : alpha;
    for (int i = 0; i < n - 1; ++i) {
        const T v = (x2[i] < T(0)) ? -x2[i] : x2[i];
        if (v > cmax) {
            cmax = v;
        }
    }
    const T invMax = T(1) / cmax;
    const T d = alpha * invMax;
    T s2 = d * d;
    for (int i = 0; i < n - 1; ++i) {
        const T e = x2[i] * invMax;
        s2 += e * e;
    }
    const T sq = std::sqrt(s2);
    const T betaS = cmax * ((d >= T(0)) ? -sq : sq);
    tau = (betaS - alpha) / betaS;
    const T denS = alpha - betaS;
    const T scaleS = (denS != T(0)) ? (T(1) / denS) : T(0);
    for (int i = 0; i < n - 1; ++i) {
        x2[i] *= scaleS;
    }
    alpha = betaS;
}

/**
 * 生成一个 Householder 反射子。
 *
 * @param n      向量总长（含对角元）
 * @param alpha  入：x[0]；出：beta
 * @param x2     x[1..n-1] 的首地址（连续存储，步长 1）；出：归一化后的 v2
 * @param tau    出
 */
template <class T>
inline void Larfg(int n, T& alpha, T* x2, T& tau)
{
    if (n <= 1) {
        tau = T(0);
        return;
    }
    // ||x2||^2
    T sumSq = T(0);
    for (int i = 0; i < n - 1; ++i) {
        sumSq += x2[i] * x2[i];
    }
    // ★ 定标兜底（LAPACK 的 snrm2 做法）。本函数是测试的 CPU 标杆且会实例化成 float，
    //   不定标时与 kernel 同时犯错、判据失效，所以两边都要有：
    //     · sumSq 或 a*a + sumSq 溢出成 Inf/NaN（|元素| 约 >= 1.8e19）；
    //     · sumSq 下溢成 0 而 x2 并非全零（|元素| 约 <= 1e-19）。
    //   **只在这两种情形重算**，正常数据仍走下面原来的式子，逐位不变。
    bool anyBelow = false;
    for (int i = 0; i < n - 1; ++i) {
        if (x2[i] != T(0)) {
            anyBelow = true;
            break;
        }
    }
    const T lim = std::numeric_limits<T>::max();
    const bool sumOk = (sumSq >= T(0)) && (sumSq <= lim);
    const bool totOk = sumOk && ((alpha * alpha + sumSq) <= lim);
    if (!totOk || (sumSq == T(0) && anyBelow)) {
        LarfgScaled(n, alpha, x2, tau, anyBelow);
        return;
    }
    if (sumSq == T(0)) {
        tau = T(0);  // 已是 [alpha; 0]，H = I
        return;
    }
    const T a = alpha;
    const T norm = std::sqrt(a * a + sumSq);
    // sign(0) = +1，与 LAPACK 的 SIGN(norm, a) 一致
    const T beta = (a >= T(0)) ? -norm : norm;
    // sumSq > 0 保证 norm > 0 ⇒ beta = ±norm 非 0，且 a - beta 与 a 同号、同样非 0。
    // 静态检查不做这层推导，所以就地再判一次。
    tau = (beta != T(0)) ? ((beta - a) / beta) : T(0);
    const T denScale = a - beta;
    const T scale = (denScale != T(0)) ? (T(1) / denScale) : T(0);
    for (int i = 0; i < n - 1; ++i) {
        x2[i] *= scale;
    }
    alpha = beta;
}

/**
 * 面板的非分块 QR（对齐 LAPACK sgeqr2）。
 * A 为 m x nb 列主序（ld = lda）；返回时上三角为 R、严格下三角为 v（v0 隐含为 1），tau 长度 min(m, nb)。
 */
template <class T>
inline void Geqr2(int m, int nb, T* A, int lda, T* tau)
{
    const int kmax = (m < nb) ? m : nb;
    for (int j = 0; j < kmax; ++j) {
        const int len = m - j;
        T& alpha = ColMajorAt(A, lda, j, j);
        T* x2 = &ColMajorAt(A, lda, j + 1, j);
        Larfg(len, alpha, x2, tau[j]);

        if (tau[j] != T(0) && j + 1 < nb) {
            const T beta = alpha;
            alpha = T(1);  // 临时把 v0 置 1，便于统一做内积
            for (int c = j + 1; c < nb; ++c) {
                T dot = T(0);
                for (int r = j; r < m; ++r) {
                    dot += ColMajorAt(A, lda, r, j) * ColMajorAt(A, lda, r, c);
                }
                const T w = tau[j] * dot;
                for (int r = j; r < m; ++r) {
                    ColMajorAt(A, lda, r, c) -= w * ColMajorAt(A, lda, r, j);
                }
            }
            alpha = beta;  // 恢复 R 的对角元
        }
    }
}

/**
 * 由 (V, tau) 构造块三角因子 T（forward / columnwise，对齐 LAPACK slarft）。
 * V 为 n x k 列主序（ldv），下梯形，对角隐含为 1；T 为 k x k 列主序（ldt），上三角。
 */
template <class T>
inline void Larft(int n, int k, const T* V, int ldv, const T* tau, T* Tm, int ldt)
{
    for (int i = 0; i < k; ++i) {
        if (tau[i] == T(0)) {
            for (int r = 0; r <= i; ++r) {
                ColMajorAt(Tm, ldt, r, i) = T(0);
            }
            continue;
        }
        // t = -tau_i * V[:, 0:i]^T * V[:, i]，注意对角隐含为 1、上方为 0
        for (int r = 0; r < i; ++r) {
            T dot = T(0);
            for (int row = i; row < n; ++row) {
                const T vr = (row == r) ? T(1) : ((row > r) ? ColMajorAt(V, ldv, row, r) : T(0));
                const T vi = (row == i) ? T(1) : ColMajorAt(V, ldv, row, i);
                dot += vr * vi;
            }
            ColMajorAt(Tm, ldt, r, i) = -tau[i] * dot;
        }
        // T[0:i, i] <- T[0:i, 0:i] * T[0:i, i]（上三角矩阵乘，就地）
        for (int r = 0; r < i; ++r) {
            T acc = T(0);
            for (int c = r; c < i; ++c) {
                acc += ColMajorAt(Tm, ldt, r, c) * ColMajorAt(Tm, ldt, c, i);
            }
            ColMajorAt(const_cast<T*>(Tm), ldt, r, i) = acc;
        }
        ColMajorAt(Tm, ldt, i, i) = tau[i];
    }
}

/// slarft 的 BACKWARD 分支：H_k...H_1 = I - V T V^T。
/// V 的第 i 列在第 (n-k+i) 行隐含 1、**下方**为 0，T 是**下三角**。
///
/// ★ 第二重循环必须 r 从大到小：T(r,i) = sum_{c=i+1..r} T(r,c)*T(c,i) 读的是 c <= r 的那些
///   T(c,i)，而它们正是本列刚写的值。r 递增会读到已被覆盖的项（FORWARD 恰好相反：
///   那边读 c >= r，所以 r 递增才对）。
template <class T>
inline void LarftBackward(int n, int k, const T* V, int ldv, const T* tau, T* Tm, int ldt)
{
    for (int i = k - 1; i >= 0; --i) {
        const int di = n - k + i;  // 第 i 列隐含 1 的行
        if (tau[i] == T(0)) {
            for (int r = i; r < k; ++r) {
                ColMajorAt(Tm, ldt, r, i) = T(0);
            }
            continue;
        }
        for (int r = i + 1; r < k; ++r) {
            const int dr = n - k + r;
            T dot = T(0);
            for (int row = 0; row <= di; ++row) {
                const T vr = (row == dr) ? T(1) : ((row < dr) ? ColMajorAt(V, ldv, row, r) : T(0));
                const T vi = (row == di) ? T(1) : ColMajorAt(V, ldv, row, i);
                dot += vr * vi;
            }
            ColMajorAt(Tm, ldt, r, i) = -tau[i] * dot;
        }
        for (int r = k - 1; r > i; --r) {
            T acc = T(0);
            for (int c = i + 1; c <= r; ++c) {
                acc += ColMajorAt(Tm, ldt, r, c) * ColMajorAt(Tm, ldt, c, i);
            }
            ColMajorAt(Tm, ldt, r, i) = acc;
        }
        ColMajorAt(Tm, ldt, i, i) = tau[i];
    }
}


/// 由 Geqr2 的输出 (A 下梯形里的 v, tau) 生成显式 Q 的前 n 列，原地覆盖 A。
/// 对齐 LAPACK dorg2r：倒着处理每个反射子，H_i 只作用到第 i..n-1 列、第 i.. 行。
/// 仅作 CPU 参考用（O(m*n*k)，不追求效率）。
inline void Org2r(int m, int n, int k, float* A, int lda, const float* tau)
{
    // 第 k..n-1 列先置成单位阵的对应列
    for (int c = k; c < n; ++c) {
        for (int r = 0; r < m; ++r) {
            A[static_cast<long long>(c) * lda + r] = (r == c) ? 1.0f : 0.0f;
        }
    }
    for (int i = k - 1; i >= 0; --i) {
        const float ti = tau[i];
        // H_i 作用到第 i+1..n-1 列（v 的第 i 个分量隐含为 1）
        if (i + 1 < n && ti != 0.0f) {
            for (int c = i + 1; c < n; ++c) {
                float dot = A[static_cast<long long>(c) * lda + i];  // v[i] = 1
                for (int r = i + 1; r < m; ++r) {
                    dot += A[static_cast<long long>(i) * lda + r] * A[static_cast<long long>(c) * lda + r];
                }
                const float w = ti * dot;
                A[static_cast<long long>(c) * lda + i] -= w;
                for (int r = i + 1; r < m; ++r) {
                    A[static_cast<long long>(c) * lda + r] -= w * A[static_cast<long long>(i) * lda + r];
                }
            }
        }
        // 本列就地变成 Q 的第 i 列
        for (int r = i + 1; r < m; ++r) {
            A[static_cast<long long>(i) * lda + r] *= -ti;
        }
        A[static_cast<long long>(i) * lda + i] = 1.0f - ti;
        for (int r = 0; r < i; ++r) {
            A[static_cast<long long>(i) * lda + r] = 0.0f;
        }
    }
}

}  // namespace QrCore

#endif  // QR_CORE_HOUSEHOLDER_H
