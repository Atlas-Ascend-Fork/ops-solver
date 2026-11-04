#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import sys

import numpy as np

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

eps = 1e-3
error_tol = 1e-4


def forward_substitution(LU, b, piv):
    """Ly = Pb (前向替换)"""
    n = LU.shape[0]
    y = b.copy()

    # 应用 pivot
    for i in range(len(piv)):
        y[[i, piv[i]]] = y[[piv[i], i]]

    # 解 Ly = Pb
    for i in range(n):
        for j in range(i):
            y[i] -= LU[i, j] * y[j]

    return y


def backward_substitution(LU, y):
    """Ux = y (后向替换)"""
    n = LU.shape[0]
    x = y.copy()

    for i in reversed(range(n)):
        for j in range(i + 1, n):
            x[i] -= LU[i, j] * x[j]
        x[i] /= LU[i, i]

    return x


def solve_lu(LU, b, piv):
    """通过 LU 分解求解 Ax=b"""
    y = forward_substitution(LU, b, piv)
    x = backward_substitution(LU, y)
    return x


def l1norm(a):
    """计算矩阵 L1 范数"""
    return np.max(np.sum(np.abs(a), axis=0))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    nrhs = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    # 读取数据
    lu = np.fromfile("./test/cgetrs/data/input/A_gm.bin", dtype=np.complex64).reshape(
        n, n
    )

    rhs = np.fromfile("./test/cgetrs/data/input/B_gm.bin", dtype=np.complex64).reshape(
        n, nrhs
    )

    # W_gm.bin 为 1-based 主元契约（issue #344 对齐后），还原 0-based 供本脚本交换使用
    piv = np.fromfile("./test/cgetrs/data/input/W_gm.bin", dtype=np.int32).reshape(n) - 1

    solution = np.fromfile(
        "./test/cgetrs/data/output/X_gm.bin", dtype=np.complex64
    ).reshape(n, nrhs)

    if not all(np.isfinite(array).all() for array in (lu, rhs, solution)):
        logging.info("Non-finite input or solution cannot pass accuracy verification.")
        return False

    # 计算 expected 解
    reference = np.zeros_like(solution)
    for k in range(nrhs):
        reference[:, k] = solve_lu(lu, rhs[:, k], piv)

    if not np.isfinite(reference).all():
        logging.info("Non-finite reference solution cannot pass accuracy verification.")
        return False

    logging.info("-------------------------------------------------------")
    cnt = 0

    for j in range(nrhs):
        for i in range(n):
            # 计算绝对误差
            abs_err = np.abs(reference[i, j] - solution[i, j])

            # 计算相对误差（分母用参考值的绝对值，避免除零）
            denom = np.abs(reference[i, j]) if np.abs(reference[i, j]) > 1e-12 else 1.0
            rel_err = abs_err / denom

            # Arrays were checked for finite values before substitution.
            if abs_err > eps and rel_err > eps:
                cnt += 1
                if cnt <= 32:
                    logging.info(
                        f"index: {i}, {j}, golden: {reference[i, j]:.4f}, actual: {solution[i, j]:.4f}, "
                        f"abs_err: {abs_err:.4f}, rel_err: {rel_err:.4f}"
                    )

    logging.info(f"Wrong Indices: {cnt}, ratio: {cnt / (n * nrhs):.4f}")
    logging.info(f"l1norm diff: {l1norm(reference - solution) / l1norm(reference):.8f}")

    return cnt / (n * nrhs) <= error_tol


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
