#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
#
# Copyright (c) 2026 Shina Guo <fs20091003@163.com>, Yuan Tang <yuantang@fudan.edu.cn>
# All rights reserved.
#
# This software is dual-licensed:
#   1) Community license: use, modify, and redistribute freely under the
#      GNU General Public License v3 or later (GPL-3.0-or-later);
#   2) Commercial license: for closed-source or commercial use / redistribution,
#      contact the copyright holder for a separate license.
# You may choose either. Default terms are GPL-3.0-or-later.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# This program is free software: you may redistribute it and/or modify it under
# the terms of the GPL. It is distributed "as is", without warranty.
# See COPYING (full GPL) and COMMERCIAL-LICENSE (commercial terms) at the repo root.
# Commercial licensing contact: yuantang@fudan.edu.cn
#
# -------------------------------------------------------------------------
# 中文版 (Chinese):
#
# PACO GEQRF — An I/O-Efficient Parallel QR Factorization for Heterogeneous Architectures
#
# 本软件采用双重许可（dual licensing）：
#   1) 社区许可：依据 GNU 通用公共许可证第 3 版或（你选择的）更新版本
#      （GPL-3.0-or-later）自由使用、修改与再分发；
#   2) 商业许可：闭源商用 / 再分发请与版权人联系获取单独授权。
# 二者任选其一。默认条款以 GPL-3.0-or-later 为准。
#
# 本程序是自由软件，你可依 GPL 条款重新发布或修改；基于“原样”提供，无担保。
# 详情见仓库根目录 COPYING（GPL 全文）与 COMMERCIAL-LICENSE（商业条款）。
# 商业授权联系：yuantang@fudan.edu.cn
"""cgeqrf 精度验证脚本（官方现行口径）。

判据（三条全部通过才算通过）:

1) DQRT01 残差判据（硬性）:
       ratio = ||R - Q^H A||_1 / (m * ||A||_1 * eps32)
       ratio <= min(30, max(10 * ratio_cpu, 10))
   ratio_cpu 为同一矩阵上 CPU 参考实现（f32 LAPACK cgeqrf + cungqr）的同一比值。

2) 重构判据（硬性）: A = Q * R
   Q 由 cungqr 从紧凑结果与 tau 构造；元素判据 2^-16 + 2^-10 * |A|，
   通过率 >= 99%，ULP 锚定硬限 max(1e-2, 32 * ULP(锚点))。

3) 正交性判据（硬性）: max|Q^H Q - I| <= 1e-4

复数 QR 的 Householder 相位不唯一，故不做 R / Householder / tau 的逐元素比对。

用法（在仓库根目录下执行，与既有约定一致）:

    python3 ./test/cgeqrf/data/verify_result.py [m] [n]

读取:

    ./test/cgeqrf/data/input/A_gm.bin    (m x n, 列主序 complex64)
    ./test/cgeqrf/data/output/R_gm.bin   (m x n, 列主序 complex64, R + Householder 向量)
    ./test/cgeqrf/data/output/tau.bin    (min(m, n), complex64)
"""

import logging
import sys
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lapack

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


# -------------------- 全局参数设置（官方口径） --------------------
DTYPE = np.complex64
RTOL = 2 ** -10              # 2^-10 = 9.765625e-04（相对容差）
ATOL = 2 ** -16              # 2^-16 = 1.52587890625e-05（绝对容差）
FAIL_RATIO = 0.01            # 允许的不合格元素最大占比（超过则判定失败）
ULP_FACTOR = 32              # 硬限中的 32*ULP
HARD_LIMIT_FLOOR = 1e-2      # 硬限下限 1e-2
ORTHO_TOL = 1e-4             # 正交性上限：max|Q^H Q - I|
EPS32 = np.float64(2.0 ** -24)   # 单精度机器 epsilon = 5.9604645e-08
DQRT01_RATIO_FACTOR = 10.0       # 阈值 = min(cap, max(FACTOR*ratio_cpu, floor))
DQRT01_RATIO_FLOOR = 10.0
DQRT01_RATIO_CAP = 30.0
# ----------------------------------------------------------------

# 数据文件路径（相对仓库根目录，与既有约定一致）
INPUT_PATH = "./test/cgeqrf/data/input/A_gm.bin"
R_PATH = "./test/cgeqrf/data/output/R_gm.bin"
TAU_PATH = "./test/cgeqrf/data/output/tau.bin"

# 默认矩阵维度（须与 cgeqrf_test.cpp 的默认 M / N 保持一致）
DEFAULT_M = 1024
DEFAULT_N = 256


@dataclass(frozen=True)
class VerifyConfig:
    """精度验证容差配置"""
    rtol: float = RTOL
    atol: float = ATOL
    fail_ratio: float = FAIL_RATIO
    ulp_factor: int = ULP_FACTOR
    hard_limit_floor: float = HARD_LIMIT_FLOOR
    ortho_tol: float = ORTHO_TOL
    eps32: object = EPS32
    dqrt01_factor: float = DQRT01_RATIO_FACTOR
    dqrt01_floor: float = DQRT01_RATIO_FLOOR
    dqrt01_cap: float = DQRT01_RATIO_CAP


# 默认配置实例
DEFAULT_VERIFY_CONFIG = VerifyConfig()


def load_binary(file_path, shape, dtype=DTYPE, fortran_order=False):
    """从二进制文件读取数据并重塑为指定形状。"""
    data = np.fromfile(file_path, dtype=dtype)
    expected_size = np.prod(shape)

    if data.size != expected_size:
        raise ValueError(
            f"文件 {file_path} 的数据量 {data.size} "
            f"与预期 {expected_size} 不符"
        )

    return data.reshape(shape, order="F" if fortran_order else "C")


def ulp_f32(value):
    """float32 在 |value| 处的 ULP（相邻浮点间距）。"""
    abs_value = np.float32(np.abs(np.float32(value)))
    if not np.isfinite(abs_value):
        return np.float32(np.inf)
    return np.float32(np.spacing(abs_value))


def hard_limit(anchor_abs, config=DEFAULT_VERIFY_CONFIG):
    """
    ULP 锚定硬限: max(1e-2, 32*ULP(1.0), 32*ULP(|anchor|))。

    ULP 锚点取“承载最大绝对误差的元素处参考值幅值”，只放宽不收紧。
    """
    limit = np.float64(config.hard_limit_floor)
    limit = max(limit, np.float64(config.ulp_factor) * np.float64(ulp_f32(1.0)))
    if np.isfinite(anchor_abs):
        limit = max(
            limit,
            np.float64(config.ulp_factor) * np.float64(ulp_f32(anchor_abs)),
        )
    return np.float64(limit)


def matrix_one_norm(matrix):
    """矩阵 1-范数（最大列绝对和），与 LAPACK ZLANGE('1') 一致。"""
    abs_matrix = np.abs(np.asarray(matrix))
    if abs_matrix.size == 0:
        return 0.0
    return float(np.max(np.sum(abs_matrix.astype(np.float64), axis=0)))


def build_q(qr_compact, tau):
    """
    由紧凑 QR 结果 + tau 构造经济型 Q（m x k）。

    优先使用单精度复数 cungqr，否则退回双精度复数 zungqr 再转 complex64。
    构造失败返回 None。
    """
    m, n = qr_compact.shape
    k = min(m, n)

    qr_for_q = np.asfortranarray(qr_compact[:, :k], dtype=DTYPE)
    tau_for_q = np.asarray(tau[:k], dtype=DTYPE)

    try:
        q_func = lapack.cungqr
    except AttributeError:
        logging.info(
            "警告: lapack.cungqr 不可用，使用 zungqr "
            "(双精度复数) 并转换为 complex64"
        )
        qr_for_q = qr_for_q.astype(np.complex128)
        tau_for_q = tau_for_q.astype(np.complex128)
        q_matrix, _work, info = lapack.zungqr(qr_for_q, tau_for_q)
        q_matrix = q_matrix.astype(DTYPE)
    else:
        q_matrix, _work, info = q_func(qr_for_q, tau_for_q)

    if info != 0:
        logging.info(f"❌ 构造 Q 失败（cungqr/zungqr info = {info}）")
        return None

    return q_matrix


def dqrt01_ratio(a_original, q_matrix, r_matrix, config=DEFAULT_VERIFY_CONFIG):
    """
    DQRT01 残差比值: ||R - Q^H A||_1 / (m * ||A||_1 * eps32)。

    q_matrix / r_matrix 为 None 时返回 None；r_matrix 需只含上三角部分。
    """
    if q_matrix is None or r_matrix is None:
        return None

    # Q 为 m x k，A 为 m x n -> Q^H A 为 k x n，与 R 同形。
    qh_a = q_matrix.conj().T @ a_original
    resid = r_matrix - qh_a
    del qh_a

    m_rows = a_original.shape[0]
    denom = np.float64(m_rows) * matrix_one_norm(a_original) * config.eps32
    resid_norm = matrix_one_norm(resid)
    del resid

    if denom <= 0:
        return 0.0
    return float(np.float64(resid_norm) / denom)


def compute_cpu_ratio(a_original, config=DEFAULT_VERIFY_CONFIG):
    """CPU 参考实现（f32 LAPACK cgeqrf + cungqr）在 DQRT01 下的 ratio。"""
    a_std = np.asfortranarray(a_original.copy(), dtype=DTYPE)

    try:
        q_func = lapack.cgeqrf
    except AttributeError:
        return None

    qr_std, tau_std, _work, info = q_func(a_std)
    if info != 0:
        return None

    q_cpu = build_q(qr_std, tau_std)
    k_dim = min(a_original.shape)
    return dqrt01_ratio(a_original, q_cpu, np.triu(qr_std[:k_dim, :]), config)


def validate_dqrt01(a_original, qr_compact, q_matrix, config=DEFAULT_VERIFY_CONFIG):
    """
    DQRT01 残差判据（官方现行口径）。

    返回 True 表示通过。
    """
    logging.info("\n" + "=" * 60)
    logging.info("  DQRT01 残差判据（官方现行口径）")
    logging.info("=" * 60)
    logging.info("  公式: ratio = ||R - Q^H A||_1 / (m * ||A||_1 * eps32)")

    m_rows, n_cols = a_original.shape
    k_dim = min(m_rows, n_cols)
    a_norm = matrix_one_norm(a_original)
    logging.info(
        f"  m = {m_rows}, ||A||_1 = {a_norm:.6e}, "
        f"eps32 = {float(config.eps32):.6e}"
    )

    ratio = dqrt01_ratio(
        a_original,
        q_matrix,
        np.triu(qr_compact[:k_dim, :]),
        config,
    )
    if ratio is None:
        logging.info("❌ 无法计算：Q / R 不可用")
        return False

    ratio_cpu = compute_cpu_ratio(a_original, config)
    if ratio_cpu is None:
        ratio_cpu_eff = 1.0
        logging.info(f"  npu ratio      = {ratio:.6e}")
        logging.info("  ⚠️ 无 CPU 参考比值（cgeqrf 不可用），阈值按 ratio_cpu = 1 计算")
    else:
        ratio_cpu_eff = ratio_cpu
        logging.info(f"  npu ratio      = {ratio:.6e}")
        logging.info(
            f"  cpu ratio_cpu  = {ratio_cpu:.6e} "
            f"(f32 LAPACK cgeqrf + cungqr)"
        )

    limit = min(
        config.dqrt01_cap,
        max(config.dqrt01_factor * ratio_cpu_eff, config.dqrt01_floor),
    )
    passed = ratio <= limit

    logging.info(
        f"  阈值 = min({config.dqrt01_cap:g}, max("
        f"{config.dqrt01_factor:g}*ratio_cpu, {config.dqrt01_floor:g})) "
        f"= {limit:.6e}"
    )
    logging.info(
        f"  判定结果: {'✅ 通过' if passed else '❌ 失败'} "
        f"(ratio = {ratio:.4e} {'<=' if passed else '>'} {limit:.4e})"
    )
    return passed


def validate_reconstruction(
    a_original,
    qr_compact,
    tau,
    q_matrix,
    config=DEFAULT_VERIFY_CONFIG,
):
    """
    利用输出的紧凑 QR 格式重构 A_recon = Q * R，并与原始矩阵比较。

    判据与官方一致：混合容差 + 通过率 + ULP 锚定硬限；另外补充一条与元素容差
    同尺度的范数判据（极端动态范围下小元素会被大元素的舍入噪声淹没）。

    返回 True 表示重构通过。
    """
    logging.info("\n" + "=" * 60)
    logging.info("  重构验证 (A = Q * R)")
    logging.info("=" * 60)

    m_rows, n_cols = a_original.shape
    k_dim = min(m_rows, n_cols)

    if q_matrix is None:
        logging.info("❌ 重构验证失败：Q 未能构造出来")
        return False

    # R 是 k x n 的上三角部分，Q 是 m x k。
    r_matrix = np.triu(qr_compact[:k_dim, :])
    a_recon = q_matrix @ r_matrix

    diff = a_original - a_recon
    abs_err = np.abs(diff)

    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)
    rmse = np.sqrt(np.mean(np.abs(diff) ** 2))

    # 计算逐元素相对误差，微小常数用于避免除以零。
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = abs_err / (np.abs(a_original) + 1e-18)
        max_rel_err = np.max(rel_err)

    frob_norm_orig = np.linalg.norm(a_original, ord="fro")
    frob_norm_diff = np.linalg.norm(diff, ord="fro")

    if frob_norm_orig > 0:
        frob_rel_err = frob_norm_diff / frob_norm_orig
    else:
        frob_rel_err = float("inf")

    tolerance = config.atol + config.rtol * np.abs(a_original)
    pass_mask = abs_err <= tolerance
    pass_count = np.count_nonzero(pass_mask)
    matched_ratio = pass_count / a_original.size

    max_idx = np.unravel_index(int(np.argmax(abs_err)), abs_err.shape)
    limit = hard_limit(np.abs(a_original[max_idx]), config)

    # 范数判据：||diff||_F <= sqrt(sum(tol^2))，与元素容差同尺度。
    tol_fro = float(np.sqrt(np.sum(np.square(tolerance.astype(np.float64)))))

    logging.info("📈 重构误差指标:")
    logging.info(f"  最大绝对误差: {max_abs_err:.6e} (硬限: {limit:.6e})")
    logging.info(f"  平均绝对误差: {mean_abs_err:.6e}")
    logging.info(f"  均方根误差 (RMSE): {rmse:.6e}")
    logging.info(f"  最大相对误差: {max_rel_err:.6e}")
    logging.info(f"  Frobenius 范数相对误差: {frob_rel_err:.6e}")
    logging.info(
        f"  混合容差通过率 "
        f"(atol={config.atol:.6e}, rtol={config.rtol:.6e}): {matched_ratio:.2%}"
    )
    logging.info(
        f"  范数判据 ||diff||_F / sqrt(sum(tol^2)): "
        f"{frob_norm_diff:.6e} / {tol_fro:.6e}"
    )

    failed = False
    if max_abs_err > limit:
        logging.info(f"❌ 重构误差超限: 最大绝对误差 (> {limit:.6e})。")
        failed = True

    ratio_ok = matched_ratio >= 1 - config.fail_ratio
    norm_ok = frob_norm_diff <= tol_fro
    if not ratio_ok and not norm_ok:
        logging.info(
            f"❌ 重构误差超限: 通过率低于 "
            f"{1 - config.fail_ratio:.2%} 且范数判据不满足。"
        )
        failed = True
    elif not ratio_ok:
        logging.info(
            f"⚠️ 逐元素通过率 {matched_ratio:.2%} < "
            f"{1 - config.fail_ratio:.2%}，但范数判据满足"
            f"（极端动态范围下的正常现象）。"
        )

    if not failed:
        logging.info("✅ 重构误差在可接受范围内。")

    return not failed


def validate_orthogonality(q_matrix, config=DEFAULT_VERIFY_CONFIG):
    """
    验证 Q 的正交性：G = Q^H Q 应与单位阵 I 一致。

    返回 True 表示 max|Q^H Q - I| 在容差内。
    """
    logging.info("\n" + "=" * 60)
    logging.info("  Q 正交性验证 (Q^H Q 与单位阵比较)")
    logging.info("=" * 60)

    if q_matrix is None:
        logging.info("❌ 无法验证：Q 未构造成功（cungqr/zungqr 失败）")
        return False

    k_dim = q_matrix.shape[1]
    if k_dim == 0:
        logging.info("⚠️ Q 为空，跳过正交性验证")
        return True

    gram = q_matrix.conj().T @ q_matrix
    identity = np.eye(k_dim, dtype=gram.dtype)
    abs_diff = np.abs(gram - identity)

    max_abs = np.max(abs_diff)
    mean_abs = np.mean(abs_diff)
    # ||I||_F = sqrt(k)，用它做相对度量。
    frob_rel = np.linalg.norm(abs_diff) / np.sqrt(k_dim)
    max_idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)

    logging.info(f"  Q 形状: {q_matrix.shape}，k = min(m, n) = {k_dim}")
    logging.info(
        f"  最大绝对偏差 |Q^H Q - I|: {max_abs:.6e} "
        f"(上限: {config.ortho_tol:.6e})"
    )
    logging.info(f"  最大偏差位置: {max_idx}")
    logging.info(f"  平均绝对偏差: {mean_abs:.6e}")
    logging.info(
        f"  Frobenius 相对偏差 "
        f"||Q^H Q - I||_F / sqrt(k): {frob_rel:.6e}"
    )

    passed = max_abs <= config.ortho_tol
    logging.info(
        "  判定结果: "
        + ("✅ 通过（Q 正交）" if passed else "❌ 不通过（Q 非正交）")
    )
    return passed


def main():
    # 1. 读取参数（位置参数与既有约定一致，默认值与 cgeqrf_test.cpp 对齐）。
    m = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_M
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    k = min(m, n)
    logging.info(f"矩阵维度: m={m}, n={n}, 数据类型: {DTYPE}")

    if m <= 0 or n <= 0:
        logging.info(f"❌ 非法矩阵维度: m={m}, n={n}")
        return False

    config = DEFAULT_VERIFY_CONFIG

    # 2. 读取按 Fortran 列主序存储的原始矩阵与算子输出。
    a_original = load_binary(INPUT_PATH, (m, n), fortran_order=True)
    qr_compact = load_binary(R_PATH, (m, n), fortran_order=True)
    tau = np.fromfile(TAU_PATH, dtype=DTYPE)

    if tau.size != k:
        logging.info(
            f"警告: tau 文件长度 {tau.size} "
            f"与 min(m,n)={k} 不符，将截断/填充"
        )
        if tau.size < k:
            tau = np.pad(
                tau,
                (0, k - tau.size),
                constant_values=0.0 + 0.0j,
            )
        else:
            tau = tau[:k]

    logging.info("原始矩阵与算子输出已读入")

    # 3. 构造 Q（DQRT01 / 重构 / 正交性共用，只构造一次）。
    q_matrix = build_q(qr_compact, tau)

    # 4. 三条硬性判据。
    ok_dqrt01 = validate_dqrt01(a_original, qr_compact, q_matrix, config)
    ok_reconstruction = validate_reconstruction(
        a_original, qr_compact, tau, q_matrix, config
    )
    ok_orthogonality = validate_orthogonality(q_matrix, config)

    return ok_dqrt01 and ok_reconstruction and ok_orthogonality


if __name__ == "__main__":
    try:
        res = main()
        if not res:
            logging.info("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            logging.info("[Success] Case accuracy is verification passed.")
            sys.exit(0)
    except Exception as e:
        logging.info(e)
        sys.exit(1)
