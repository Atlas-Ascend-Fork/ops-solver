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
# See COPYING-PACO (full GPL) and LICENSE-PACO (dual licensing notice) at the repo root.
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
# 详情见仓库根目录 COPYING-PACO（GPL 全文）与 LICENSE-PACO（双重许可说明）。
# 商业授权联系：yuantang@fudan.edu.cn
import logging
import sys
import os

import numpy as np

# 注意（issue #573 检视整改）：本脚本产物按脚本所在目录落盘（__file__ 相对解析），
# 而测试二进制 cgeqrf_test.cpp 与 verify_result.py 此前按 cwd 相对读取 ./test/...。
# verify_result.py 已由配套 PR 同步改为 __file__ 解析；测试二进制仍需从仓根运行
# （build.sh --run 形态从仓根执行，README 单步引导也是仓根 cwd——当前不影响主流程）。
_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def print_matrix_corner(matrix, name, rows=5, cols=5):
    """打印矩阵左上角"""
    logging.info(f"\n{name} matrix corner:")
    r = min(rows, matrix.shape[0])
    c = min(cols, matrix.shape[1])
    logging.info(matrix[:r, :c])


def print_first_column_norm(matrix):
    """打印矩阵第一列的二范数"""
    first_col = matrix[:, 0]
    norm = np.linalg.norm(first_col)
    logging.info(f"\nFirst column norm: {norm:10.6f}")
    logging.info(f"First column elements: {first_col[:5]}")


def generate_test_matrix(m, n, zero_ratio=0.0):
    """生成测试矩阵（复数complex64），零列比例为 zero_ratio（0~1）"""
    np.random.seed(42)
    # 生成复数的实部和虚部
    real_part = np.random.randn(m, n).astype(np.float32)
    imag_part = np.random.randn(m, n).astype(np.float32)
    input_matrix = (real_part + 1j * imag_part).astype(np.complex64)

    # 随机选择需要置零的列
    if zero_ratio > 0:
        num_zero_cols = int(round(n * zero_ratio))
        if num_zero_cols == 0 and zero_ratio > 0:
            num_zero_cols = 1
        zero_col_indices = np.random.choice(n, size=num_zero_cols, replace=False)
        input_matrix[:, zero_col_indices] = 0.0 + 0.0j
        logging.info(f"Zero columns set at indices: {zero_col_indices}")

    # 计算QR分解（complex64）
    q_matrix, r_matrix = np.linalg.qr(input_matrix.astype(np.complex64))
    return (
        input_matrix.astype(np.complex64),
        q_matrix.astype(np.complex64),
        r_matrix.astype(np.complex64),
    )


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    zero_ratio = float(sys.argv[3]) if len(sys.argv) > 3 else 0.001
    # 参数域校验（issue #485，对齐 12a03c5 LU 族 gen_data 口径）：m/n 非正数此前
    # 在 numpy 深处报 'negative dimensions are not allowed'，错误定位差；
    # zero_ratio 越界 [0, 1] 会静默生成失真的稀疏度夹具，污染比对基准
    if m <= 0:
        raise ValueError(f"m 必须是正整数, got {m}")
    if n <= 0:
        raise ValueError(f"n 必须是正整数, got {n}")
    if not 0 <= zero_ratio <= 1:
        raise ValueError(f"zero_ratio 必须在 [0, 1] 区间, got {zero_ratio}")

    # 确保输出目录存在
    output_dir = f"{_D}input"
    os.makedirs(output_dir, exist_ok=True)

    # 生成测试数据
    input_matrix, q_matrix, r_matrix = generate_test_matrix(m, n, zero_ratio)

    # 打印矩阵信息（仅作调试用）
    print_matrix_corner(input_matrix, "Input")
    print_first_column_norm(input_matrix)
    print_matrix_corner(q_matrix, "Q")
    print_matrix_corner(r_matrix, "R")

    # 保存数据到文件（强制列主序）
    input_matrix.astype(np.complex64).ravel(order="F").tofile(
        f"{_D}input/A_gm.bin"
    )

    logging.info(
        f"\nGenerated test matrix with shape ({m}, {n}), zero_ratio = {zero_ratio:.2f}"
    )
    logging.info(
        "Saved input matrix (column-major complex64) to ./test/cgeqrf/data/input/A_gm.bin"
    )


if __name__ == "__main__":
    main()
