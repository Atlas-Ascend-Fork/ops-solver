#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def write_complex_interleaved(filename: Path, matrix: np.ndarray) -> None:
    """
    将复数矩阵按 C 行优先展开，并按照：
        real0, imag0, real1, imag1, ...
    的形式写入二进制文件。

    文件中不包含维数、数据类型等额外信息。
    """
    filename.parent.mkdir(parents=True, exist_ok=True)

    matrix = np.ascontiguousarray(matrix)
    flat = matrix.ravel(order="C")

    if matrix.dtype == np.complex64:
        real_dtype = np.float32
    elif matrix.dtype == np.complex128:
        real_dtype = np.float64
    else:
        raise TypeError(f"不支持的数据类型：{matrix.dtype}")

    output = np.empty(flat.size * 2, dtype=real_dtype)
    output[0::2] = flat.real
    output[1::2] = flat.imag
    output.tofile(filename)


def generate_positive_definite_matrix(
    n: int,
    seed: int,
    diagonal_shift: float,
    dtype: np.dtype,
) -> np.ndarray:
    """
    生成 n*n Hermitian 正定复数矩阵。

    构造方式：
        A = X @ X^H + diagonal_shift * I

    当 diagonal_shift > 0 时，A 为严格正定矩阵。
    """
    if n <= 0:
        raise ValueError("n 必须是正整数")

    if diagonal_shift <= 0:
        raise ValueError("diagonal_shift 必须大于 0，以保证矩阵严格正定")

    rng = np.random.default_rng(seed)

    # 除以 sqrt(2n)，避免矩阵元素随 n 增大得过快。
    scale = np.sqrt(2.0 * n)
    x_real = rng.standard_normal((n, n)) / scale
    x_imag = rng.standard_normal((n, n)) / scale
    x = x_real + 1j * x_imag

    a = x @ x.conj().T
    a += diagonal_shift * np.eye(n, dtype=np.complex128)

    # 消除浮点运算可能产生的极小非 Hermitian 误差。
    a = (a + a.conj().T) * 0.5

    return np.asarray(a, dtype=dtype)


def parse_arguments():
    """解析命令行参数并返回解析结果"""
    parser = argparse.ArgumentParser(description="生成 n*n Hermitian 正定复数矩阵，并输出矩阵和 Cholesky 标准结果。")
    parser.add_argument("n", nargs="?", type=int, default=128, help="矩阵阶数，默认值为 128")
    parser.add_argument("--seed", type=int, default=2026, help="随机数种子，默认值为 2026")
    parser.add_argument("--shift", type=float, default=1.0, help="添加到矩阵对角线上的正数，默认值为 1.0")
    parser.add_argument("--dtype", choices=["complex64", "complex128"], default="complex64", help="输出复数类型，默认值为 complex64")
    parser.add_argument("--input-file", type=Path, default=Path("./test/cpotrf/data/input/A_gm.bin"), help="原始矩阵输出路径，默认 ./test/cpotrf/data/input/A_gm.bin")
    parser.add_argument("--u-file", type=Path, default=Path("./test/cpotrf/data/output/U_std.bin"), help="上三角矩阵输出路径，默认 ./test/cpotrf/data/output/U_std.bin")
    parser.add_argument("--l-file", type=Path, default=Path("./test/cpotrf/data/output/L_std.bin"), help="下三角矩阵输出路径，默认 ./test/cpotrf/data/output/L_std.bin")

    args = parser.parse_args()
    if args.n <= 0:
        parser.error("n 必须是正整数")
    return args


def main() -> None:
    args = parse_arguments()

    dtype = np.dtype(args.dtype)

    # 生成正定复数矩阵。
    a = generate_positive_definite_matrix(
        n=args.n,
        seed=args.seed,
        diagonal_shift=args.shift,
        dtype=dtype,
    )

    # NumPy 的 cholesky 返回下三角矩阵 L：
    l_matrix = np.linalg.cholesky(a).astype(dtype, copy=False)

    # U = L^H，因此 A = U^H @ U
    u_matrix = np.ascontiguousarray(l_matrix.conj().T)

    write_complex_interleaved(args.input_file, a)
    write_complex_interleaved(args.u_file, u_matrix)
    write_complex_interleaved(args.l_file, l_matrix)

    # 简单验证分解结果。
    reconstruction_l = l_matrix @ l_matrix.conj().T
    reconstruction_u = u_matrix.conj().T @ u_matrix

    l_error = np.max(np.abs(a - reconstruction_l))
    u_error = np.max(np.abs(a - reconstruction_u))
    hermitian_error = np.max(np.abs(a - a.conj().T))

    eigenvalues = np.linalg.eigvalsh(a.astype(np.complex128))

    logging.info("矩阵和 Cholesky 标准结果生成完成")
    logging.info(f"矩阵维数             : {args.n} x {args.n}")
    logging.info(f"复数数据类型         : {dtype}")
    logging.info(f"随机数种子           : {args.seed}")
    logging.info(f"对角线偏移           : {args.shift}")
    logging.info(f"A 输出路径           : {args.input_file}")
    logging.info(f"U 输出路径           : {args.u_file}")
    logging.info(f"L 输出路径           : {args.l_file}")
    logging.info(f"A 文件大小           : {args.input_file.stat().st_size} bytes")
    logging.info(f"U 文件大小           : {args.u_file.stat().st_size} bytes")
    logging.info(f"L 文件大小           : {args.l_file.stat().st_size} bytes")
    logging.info(f"Hermitian 最大误差   : {hermitian_error:.6e}")
    logging.info(f"L 重构最大绝对误差   : {l_error:.6e}")
    logging.info(f"U 重构最大绝对误差   : {u_error:.6e}")
    logging.info(f"最小特征值           : {eigenvalues[0]:.6e}")
    logging.info(f"最大特征值           : {eigenvalues[-1]:.6e}")
    logging.info(f"估算条件数           : {eigenvalues[-1] / eigenvalues[0]:.6e}")


if __name__ == "__main__":
    main()
