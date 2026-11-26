# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：lu_verify_utils 的 LU 参考解计算（issue #635）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from lu_verify_utils import backward_substitution, forward_substitution, solve_lu


def _lu_from_matrix(a):
    """构造 packed LU（L 严格下三角 + U 上三角）与 0-based 单位主元。"""
    n = a.shape[0]
    lu = np.tril(a, -1) + np.triu(a)
    piv = np.arange(n, dtype=np.int32)  # 0-based identity（load_pivots 输出格式）
    return lu, piv


def test_forward_substitution_diagonal():
    n = 4
    lu = np.triu(np.eye(n, dtype=np.float32))  # 上三角=I → forward 什么都不做
    piv = np.arange(n, dtype=np.int32)  # 0-based identity（load_pivots 输出格式）
    b = np.arange(n, dtype=np.float32)
    y = forward_substitution(lu, b.reshape(-1, 1), piv)
    # L=I（下三角无值），forward 保持 b 不变
    np.testing.assert_allclose(y.ravel(), b, atol=1e-6)


def test_backward_substitution_identity():
    n = 4
    lu = np.eye(n, dtype=np.float32)  # U=I → x=y
    y = np.arange(n, dtype=np.float32)
    x = backward_substitution(lu, y.reshape(-1, 1))
    np.testing.assert_allclose(x.ravel(), y, atol=1e-6)


def test_solve_lu_diagonal_system():
    n = 8
    rng = np.random.RandomState(42)
    a = np.diag(rng.rand(n).astype(np.float32) + 1.0)  # 对角占优
    lu, piv = _lu_from_matrix(a)
    b = rng.rand(n).astype(np.float32)
    x = solve_lu(lu, b.reshape(-1, 1), piv)
    expected = np.linalg.solve(a.astype(np.float64), b.reshape(-1, 1).astype(np.float64))
    np.testing.assert_allclose(x.ravel(), expected.ravel(), rtol=1e-4, atol=1e-5)


def test_solve_lu_identity():
    n = 4
    lu = np.eye(n, dtype=np.float32)
    piv = np.arange(n, dtype=np.int32)  # 0-based identity（load_pivots 输出格式）
    b = np.arange(n, dtype=np.float32)
    x = solve_lu(lu, b.reshape(-1, 1), piv)
    np.testing.assert_allclose(x.ravel(), b, atol=1e-6)


def test_forward_substitution_real_lower():
    """L 有严格下三角非零值：Ly=b 有非平凡解。"""
    n = 3
    lu = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, 4.0, 1.0],
    ], dtype=np.float32)
    piv = np.arange(n, dtype=np.int32)
    b = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    y = forward_substitution(lu, b, piv)
    # 手工解 Ly=b：y0=1, y1=2-2*1=0, y2=3-3*1-4*0=0
    np.testing.assert_allclose(y.ravel(), [1.0, 0.0, 0.0], atol=1e-6)


def test_backward_substitution_real_upper():
    """U 有非对角非零值：Ux=y 有非平凡解。"""
    n = 3
    lu = np.array([
        [2.0, 1.0, 3.0],
        [0.0, 2.0, 1.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    y = np.array([[2.0], [2.0], [1.0]], dtype=np.float32)
    x = backward_substitution(lu, y)
    # 手工解 Ux=y：x2=1, x1=(2-1)/2=0.5, x0=(2-0.5-3)/2=-0.75
    np.testing.assert_allclose(x.ravel(), [-0.75, 0.5, 1.0], atol=1e-5)


def test_solve_lu_full_system():
    """完整非退化 LU：solve_lu 结果与 numpy.linalg.solve 一致。"""
    n = 4
    rng = np.random.RandomState(7)
    a = rng.rand(n, n).astype(np.float32) + np.eye(n, dtype=np.float32) * n
    b = rng.rand(n, 1).astype(np.float32)
    # 用 numpy 的 LU 分解提取 L（严格下三角）和 U（上三角）
    import scipy.linalg as sla
    lu_packed, piv_scipy = sla.lu_factor(a.astype(np.float64))
    # scipy lu_factor 的 piv 是 0-based（部分主元时值域 [i, n-1]，与 load_pivots 输出同格式）
    piv_0based = piv_scipy.astype(np.int32)
    # scipy 的 lu_packed: 上三角为 U，严格下三角为 L 的系数
    lu = np.tril(lu_packed, -1) + np.triu(lu_packed)
    # 已在上面直接构造 piv_0based
    x = solve_lu(lu, b, piv_0based)
    expected = np.linalg.solve(a.astype(np.float64), b.astype(np.float64))
    np.testing.assert_allclose(x.ravel(), expected.ravel(), rtol=1e-3, atol=1e-4)
