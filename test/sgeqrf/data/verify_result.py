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
import logging
import sys
from dataclasses import dataclass
import numpy as np
from scipy.linalg import lapack

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


# -------------------- 全局参数默认值 --------------------
default_dtype = np.float32
default_rtol = 9.77e-4          # 相对容差
default_atol = 1.53e-5          # 绝对容差
default_fail_ratio = 0.01       # 允许的不合格元素最大占比（超过则判定失败）
default_max_abs_limit = 1e-3    # 最大绝对误差上限
# ----------------------------------------------------


@dataclass
class CompareConfig:
    """误差比较与判定配置"""
    rtol: float = default_rtol
    atol: float = default_atol
    fail_ratio: float = default_fail_ratio
    max_abs_limit: float = default_max_abs_limit


def load_binary(file_path, shape, dtype=default_dtype, fortran_order=False):
    """从二进制文件读取数据并重塑为指定形状"""
    data = np.fromfile(file_path, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"文件 {file_path} 的数据量 {data.size} 与预期 {np.prod(shape)} 不符")
    mat = data.reshape(shape, order='F' if fortran_order else 'C')
    return mat



def compare_matrices(name, ref, test, config: CompareConfig):
    """
    使用混合容差标准逐元素比对：
       通过条件：|ref - test| <= atol + rtol * |ref|
    整体通过条件：
       1) 通过率 >= 1 - fail_ratio
       2) 最大绝对误差 <= max_abs_limit
    """
    if ref.shape != test.shape:
        logging.info(f"❌ {name}: 形状不匹配 {ref.shape} vs {test.shape}")
        return False

    total = ref.size
    if total == 0:
        logging.info(f"⚠️ {name}: 矩阵为空，跳过比较")
        return True

    diff = ref - test
    abs_err = np.abs(diff)
    tolerance = config.atol + config.rtol * np.abs(ref)
    pass_mask = (abs_err <= tolerance)
    pass_count = np.sum(pass_mask)
    matched_ratio = pass_count / total

    max_abs_err = np.max(abs_err)
    max_idx_flat = np.argmax(abs_err)
    max_idx = np.unravel_index(max_idx_flat, abs_err.shape)
    max_ref_val = ref[max_idx]
    max_test_val = test[max_idx]

    passed = (matched_ratio >= (1 - config.fail_ratio)) and (max_abs_err <= config.max_abs_limit)

    logging.info(f"\n📊 {name} 比较结果:")
    logging.info(f"  形状: {ref.shape}")
    logging.info(f"  最大绝对误差: {max_abs_err:.6e} (上限: {config.max_abs_limit:.6e})")
    logging.info(f"  最大误差位置: {max_idx}, 参考值={max_ref_val:.6e}, 测试值={max_test_val:.6e}")
    logging.info(f"  通过元素数: {pass_count} / {total} ({matched_ratio:.2%})")
    logging.info(f"  要求通过率: ≥ {1 - config.fail_ratio:.2%}")
    logging.info(f"  判定结果: {'✅ 通过' if passed else '❌ 失败'}")

    fail_count = total - pass_count
    if fail_count > 0:
        fail_mask = ~pass_mask
        indices = np.argwhere(fail_mask)
        err_vals = abs_err[tuple(indices.T)]
        sorted_order = np.argsort(err_vals)[::-1]
        num_show = min(10, fail_count)
        sorted_indices = indices[sorted_order[:num_show]]
        logging.info(f"  不合格元素示例 (按绝对误差降序，最多显示 {num_show} 个):")
        for idx in sorted_indices:
            coord = tuple(idx)
            ref_val = ref[coord]
            test_val = test[coord]
            abs_err_val = abs_err[coord]
            tol_val = tolerance[coord]
            if ref.ndim == 1:
                coord_str = f"[{idx[0]}]"
            else:
                coord_str = f"[{idx[0]}, {idx[1]}]"
            logging.info(f"    {coord_str}: 参考={ref_val:.6e}, 测试={test_val:.6e}, "
                  f"绝对误差={abs_err_val:.6e}, 允许容差={tol_val:.6e}")

    return passed



def validate_reconstruction(a_original, qr_compact, tau, config: CompareConfig):
    """
    利用 NPU 输出的紧凑 QR 格式（qr_compact, tau）重构矩阵 A_recon = Q * R，
    并与原始 A_original 比较，输出各项误差指标。
    返回 True 如果重构误差在可接受范围内（仅作提示，不强制判定）。
    """
    logging.info("\n" + "=" * 60)
    logging.info("  NPU QR 分解重构验证 (独立于 LAPACK 标准结果)")
    logging.info("=" * 60)

    m, n = a_original.shape
    k = min(m, n)

    # SORGQR 的输入必须是 M x K，且 K <= M
    qr_for_q = np.asfortranarray(
        qr_compact[:, :k],
        dtype=default_dtype
    )

    tau_for_q = np.asarray(tau[:k], dtype=default_dtype)

    try:
        q_func = lapack.sorgqr
    except AttributeError:
        logging.info("警告: lapack.sorgqr 不可用，使用 dorgqr 并转换为双精度")
        qr_for_q = qr_for_q.astype(np.float64)
        tau_for_q = tau_for_q.astype(np.float64)
        q, work, info = lapack.dorgqr(qr_for_q, tau_for_q)
        q = q.astype(default_dtype)
    else:
        q, work, info = q_func(qr_for_q, tau_for_q)

    if info != 0:
        logging.info(f"❌ 重构 Q 失败，info = {info}")
        return False

    # R 是 K x N
    r = np.triu(qr_compact[:k, :])

    # Q: M x K, R: K x N
    a_recon = q @ r

    # 计算误差指标
    diff = a_original - a_recon
    abs_err = np.abs(diff)
    max_abs_err = np.max(abs_err)
    mean_abs_err = np.mean(abs_err)
    rmse = np.sqrt(np.mean(diff**2))
    # 相对误差（避免除以零）
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_err = abs_err / (np.abs(a_original) + 1e-18)
        max_rel_err = np.max(rel_err)
    # Frobenius 范数相对误差
    frob_norm_orig = np.linalg.norm(a_original, ord='fro')
    frob_norm_diff = np.linalg.norm(diff, ord='fro')
    if frob_norm_orig > 0:
        frob_rel_err = frob_norm_diff / frob_norm_orig
    else:
        frob_rel_err = float('inf')

    # 混合容差通过率（使用 config 中的值）
    tolerance = config.atol + config.rtol * np.abs(a_original)
    pass_mask = (abs_err <= tolerance)
    pass_count = np.sum(pass_mask)
    matched_ratio = pass_count / a_original.size

    logging.info("📈 重构误差指标:")
    logging.info(f"  最大绝对误差: {max_abs_err:.6e}")
    logging.info(f"  平均绝对误差: {mean_abs_err:.6e}")
    logging.info(f"  均方根误差 (RMSE): {rmse:.6e}")
    logging.info(f"  最大相对误差: {max_rel_err:.6e}")
    logging.info(f"  Frobenius 范数相对误差: {frob_rel_err:.6e}")
    logging.info(f"  混合容差通过率 (atol={config.atol}, rtol={config.rtol}): {matched_ratio:.2%}")

    # 如果最大绝对误差或通过率异常，给出警告（阈值固定，也可考虑放入 config）
    warn = False
    if max_abs_err > 1e-3:
        logging.info("⚠️ 警告: 最大绝对误差较大 (>1e-3)，可能存在数值问题或实现错误。")
        warn = True
    if matched_ratio < 0.99:
        logging.info("⚠️ 警告: 通过率低于 99%，重构精度可能不足。")
        warn = True
    if not warn:
        logging.info("✅ 重构误差在可接受范围内。")

    return True



def main():
    # 1. 读取参数
    rows = 1024 if len(sys.argv) < 2 else int(sys.argv[1])
    cols = 256 if len(sys.argv) < 3 else int(sys.argv[2])
    logging.info(f"矩阵维度: M={rows}, N={cols}, 数据类型: {default_dtype}")

    # 创建全局配置实例（使用默认值，也可按需覆盖）
    config = CompareConfig()

    # 2. 读取原始矩阵（假设按 Fortran 列优先存储）
    input_mat = load_binary('./test/sgeqrf/data/input/matrix.bin', (rows, cols), fortran_order=True)
    logging.info("原始矩阵已读入")

    # 3. 调用 LAPACK sgeqrf 计算标准 QR 分解（单精度）
    a_std = np.asfortranarray(input_mat.copy(), dtype=default_dtype)
    try:
        qr_func = lapack.sgeqrf
    except AttributeError:
        logging.info("警告: lapack.sgeqrf 不可用，将使用双精度 dgeqrf，结果会转为 float32")
        qr_std_d, tau_std_d, work, info = lapack.dgeqrf(a_std.astype(np.float64))
        qr_std = qr_std_d.astype(default_dtype)
        tau_std = tau_std_d.astype(default_dtype)
    else:
        qr_std, tau_std, work, info = qr_func(a_std)
    if info != 0:
        logging.info(f"LAPACK sgeqrf 失败，info = {info}")
        return
    logging.info("LAPACK 标准结果计算完成")

    # 4. 读取 NPU 输出（紧凑格式，单精度）
    npu_qr = load_binary('./test/sgeqrf/data/output/result.bin', (rows, cols), fortran_order=True)
    npu_tau = np.fromfile('./test/sgeqrf/data/output/tau.bin', dtype=default_dtype)
    if npu_tau.size != min(rows, cols):
        logging.info(f"警告: tau 文件长度 {npu_tau.size} 与 min(M,N)={min(rows, cols)} 不符，将截断/填充")
        if npu_tau.size < min(rows, cols):
            npu_tau = np.pad(npu_tau, (0, min(rows, cols) - npu_tau.size), constant_values=0.0)
        else:
            npu_tau = npu_tau[:min(rows, cols)]
    logging.info("NPU 输出文件已读入")

    # 5. 重构验证（传入 config）
    ok = validate_reconstruction(input_mat, npu_qr, npu_tau, config)

    # 6. 提取上三角 R（包括对角线）
    r_std = np.triu(qr_std)
    r_npu = np.triu(npu_qr)

    # 7. 提取下三角的 Householder 向量（严格下三角，不包括对角线）
    h_std = np.tril(qr_std, -1)
    h_npu = np.tril(npu_qr, -1)

    # 8. 对比各组件（传入 config）
    logging.info("\n========== 开始对比（与 LAPACK 标准结果） ==========")
    ok &= compare_matrices("上三角 R", r_std, r_npu, config)
    ok &= compare_matrices("下三角 Householder 向量", h_std, h_npu, config)
    ok &= compare_matrices("TAU 向量", tau_std, npu_tau, config)

    if ok:
        logging.info("\n✅ 所有对比通过，NPU 结果与 LAPACK 标准结果一致（单精度）。")
        sys.exit(0)
    else:
        logging.info("\n❌ 存在差异，请检查 NPU 实现或数值精度。")
        sys.exit(1)


if __name__ == "__main__":
    main()