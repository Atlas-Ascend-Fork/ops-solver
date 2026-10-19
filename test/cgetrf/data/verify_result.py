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

import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
)
from verify_utils import read_binary_strict  # noqa: E402

eps = 1e-3
error_tol = 1e-4


def gemm(a):
    """矩阵乘法,用于验证 LU 分解结果"""
    m, n = a.shape
    b = np.zeros((m, n), dtype=np.complex64)
    for i in range(m):
        for j in range(n):
            for k in range(min(i, j) + 1):
                if k == i:
                    b[i, j] += 1.0 * a[k, j]
                else:
                    b[i, j] += a[i, k] * a[k, j]
    return b


def l1norm(a):
    """计算矩阵的 L1 范数"""
    return np.max(np.sum(np.abs(a), axis=0))


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    t = min(m, n)

    # 读取输入文件（尾部不完整元素直接报错，issue #202）
    a = read_binary_strict("./test/cgetrf/data/input/A_gm.bin", np.complex64).reshape(
        m, n
    )
    x = read_binary_strict("./test/cgetrf/data/output/A_gm.bin", np.complex64).reshape(
        m, n
    )
    w = read_binary_strict("./test/cgetrf/data/output/W_gm.bin", np.uint32).reshape(t)

    if not np.isfinite(a).all() or not np.isfinite(x).all():
        print("Non-finite input or LU output cannot pass accuracy verification.")
        return False

    # 根据 pivot 数组进行行交换
    for j in range(n):
        for i in range(t):
            pivot = int(w[i])
            if not (i + 1 <= pivot <= m):
                raise ValueError(
                    f"invalid pivot at index {i}: {pivot}, expected [{i + 1}, {m}]"
                )
            p = pivot - 1  # convert 1-based to 0-based
            a[[i, p], j] = a[[p, i], j]

    # 计算 expected 结果
    b = gemm(x)
    if not np.isfinite(b).all():
        print("Non-finite LU product cannot pass accuracy verification.")
        return False

    print("-------------------------------------------------------")
    cnt = 0
    for j in range(n):
        for i in range(m):
            ratio = np.abs(a[i, j] - b[i, j]) / np.abs(a[i, j])
            if (
                ratio > eps
                or np.isnan(b[i, j].real)
                or np.isnan(b[i, j].imag)
                or np.isinf(b[i, j].real)
                or np.isinf(b[i, j].imag)
            ):
                cnt += 1
                if cnt <= 32:
                    print(
                        f"index: {i}, {j}, golden: {a[i, j].real:.4f}, {a[i, j].imag:.4f}, "
                        f"actual: {b[i, j].real:.4f}, {b[i, j].imag:.4f}, ratio: {ratio:.4f}"
                    )

    print(f"Wrong Indices: {cnt}, ratio: {cnt / (m * n):.4f}")
    print(f"l1norm diff: {l1norm(a - b) / l1norm(a):.8f}")

    return cnt / (m * n) <= error_tol


if __name__ == "__main__":
    try:
        res = main()
        if not res:
            print("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            print("[Success] Case accuracy verification passed.")
            sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
