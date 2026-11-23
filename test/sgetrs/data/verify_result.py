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

import os
import sys
import logging
import numpy as np

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

eps = 1e-3
error_tol = 1e-4


def forward_substitution(lu, b, piv):
    """Ly = Pb (前向替换)"""
    n = lu.shape[0]
    y = b.copy()

    # 应用 pivot
    for i in range(len(piv)):
        y[[i, piv[i]]] = y[[piv[i], i]]

    # 解 Ly = Pb
    for i in range(n):
        for j in range(i):
            y[i] -= lu[i, j] * y[j]

    return y


def backward_substitution(lu, y):
    """Ux = y (后向替换)"""
    n = lu.shape[0]
    x = y.copy()

    for i in reversed(range(n)):
        for j in range(i + 1, n):
            x[i] -= lu[i, j] * x[j]
        x[i] /= lu[i, i]

    return x


def solve_lu(lu, b, piv):
    """通过 lu 分解求解 Ax=b"""
    y = forward_substitution(lu, b, piv)
    x = backward_substitution(lu, y)
    return x


def l1norm(a):
    """计算矩阵 L1 范数"""
    return np.max(np.sum(np.abs(a), axis=0))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    nrhs = int(sys.argv[2]) if len(sys.argv) > 2 else 32

    # 读取数据
    lu = np.fromfile(f"{_D}input/A_gm.bin", dtype=np.float32).reshape(n, n)
    b = np.fromfile(f"{_D}input/B_gm.bin", dtype=np.float32).reshape(n, nrhs)
    # W_gm.bin 为 1-based 主元契约（issue #344 对齐后），还原 0-based 供本脚本交换使用；
    # 内容校验对齐同族 cgetrf verify 形态（issue #460）：越界值此前抛裸 IndexError，
    # 0 值经 -1 后为合法负索引静默回绕交换末行、以错误基准误判。
    raw_piv = np.fromfile(f"{_D}input/W_gm.bin", dtype=np.int32).reshape(n)
    for i, pivot in enumerate(raw_piv):
        if not (i + 1 <= pivot <= n):
            raise ValueError(
                f"invalid pivot at index {i}: {pivot}, expected [{i + 1}, {n}]"
            )
    piv = raw_piv - 1
    x = np.fromfile(f"{_D}output/X_gm.bin", dtype=np.float32).reshape(n, nrhs)

    if not all(np.isfinite(array).all() for array in (lu, b, x)):
        logging.info("Non-finite input or solution cannot pass accuracy verification.")
        return False

    # 计算 expected 解
    x_ref = np.zeros_like(x)
    for k in range(nrhs):
        x_ref[:, k] = solve_lu(lu, b[:, k], piv)

    if not np.isfinite(x_ref).all():
        logging.info("Non-finite reference solution cannot pass accuracy verification.")
        return False

    logging.info("-------------------------------------------------------")
    cnt = 0

    for j in range(nrhs):
        for i in range(n):
            # 计算绝对误差
            abs_err = np.abs(x_ref[i, j] - x[i, j])
            
            # 计算相对误差（分母用参考值的绝对值，避免除零）
            denom = np.abs(x_ref[i, j]) if np.abs(x_ref[i, j]) > 1e-12 else 1.0
            rel_err = abs_err / denom
            
            # 判断条件：绝对误差 > eps 且 相对误差 > eps
            # Arrays were checked for finite values before substitution.
            if abs_err > eps and rel_err > eps:
                cnt += 1
                if cnt <= 32:
                    logging.info(
                        f"index: {i}, {j}, golden: {x_ref[i, j]:.4f}, actual: {x[i, j]:.4f}, "
                        f"abs_err: {abs_err:.4f}, rel_err: {rel_err:.4f}"
                    )
                    
    logging.info(f"Wrong Indices: {cnt}, ratio: {cnt / (n * nrhs):.4f}")
    logging.info(f"l1norm diff: {l1norm(x_ref - x) / l1norm(x_ref):.8f}")

    return cnt / (n * nrhs) <= error_tol


if __name__ == "__main__":
    try:
        res = main()
        if not res:
            logging.info("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            logging.info("[Success] Case accuracy verification passed.")
            sys.exit(0)
    except Exception as e:
        logging.info(e)
        sys.exit(1)
