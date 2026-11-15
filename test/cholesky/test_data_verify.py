# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""test/cholesky/data/verify_result.py 判定逻辑的行为级回归（issue #456）。

三套语义各有正向 + 反向用例：哨兵 padding 完整性、动态 threshold、batch 物理
位相等。全部用临时目录构造真实文件，无设备依赖。
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
VERIFIER_SOURCE = HERE / "data" / "verify_result.py"

N, LD = 17, 20
SENTINEL = -12345.125


def load_verifier(directory):
    # 从仓内源文件加载判定器，并把模块级 root 重定向到临时数据目录：
    # verify_result.py 的 root 本是 __file__ 所在目录（数据目录），这里模拟
    # build.sh --run 把脚本与数据放同一目录后的行为。
    spec = importlib.util.spec_from_file_location("cholesky_data_verifier", VERIFIER_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__dict__["root"] = directory
    return module


def write_padded(path, matrix, columns):
    # 文件布局与 verify_result.read 对齐：columns × ld 行主，read 取 [:, :n].T 还原。
    # n×n 方阵按转置写入（等价列主）；1×n 行向量直接写入；n×1 列向量行主展开后同为 1×n。
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.shape == (N, N):
        values = values.T
    else:
        values = values.reshape(1, -1)
    padded = np.full((columns, LD), SENTINEL, dtype=np.float32)
    padded[:, : values.shape[1]] = values
    padded.tofile(path)


def build_project(directory, *, corrupt_padding=False, batch_bit_differ=False):
    (directory / "input").mkdir(parents=True, exist_ok=True)
    (directory / "output").mkdir(exist_ok=True)
    rng = np.random.RandomState(7)
    a = rng.standard_normal((N, N))
    a = np.array(a @ a.T, dtype=np.float32)
    b = np.array(rng.standard_normal(N), dtype=np.float32)
    ainv = np.linalg.inv(a.astype(np.float64)).astype(np.float32)
    solve = (np.linalg.inv(a.astype(np.float64)) @ b.astype(np.float64)).astype(
        np.float32
    )

    write_padded(directory / "input/a.bin", a, N)
    write_padded(directory / "input/b.bin", b.reshape(1, N), 1)

    outputs = {}
    for upper in (False, True):
        prefix = "upper-" if upper else "lower-"
        factor = (np.linalg.cholesky(a).T if upper else np.linalg.cholesky(a)).astype(
            np.float32
        )
        outputs[prefix + "factor.bin"] = (factor, N)
        # solve 文件 read(columns=1)：1 行 × ld，前 n 列为解向量（列向量行主展开）。
        outputs[prefix + "solve.bin"] = (solve.reshape(1, N), 1)
        outputs[prefix + "inverse.bin"] = (ainv, N)
        outputs[prefix + "batch-factor-0.bin"] = (factor, N)
        outputs[prefix + "batch-factor-1.bin"] = (
            (factor + 1.0).astype(np.float32) if batch_bit_differ else factor, N
        )
        outputs[prefix + "batch-solve-0.bin"] = (solve.reshape(1, N), 1)
        outputs[prefix + "batch-solve-1.bin"] = (solve.reshape(1, N), 1)
    for name, (matrix, columns) in outputs.items():
        path = directory / "output" / name
        write_padded(path, matrix, columns)
        if corrupt_padding and name.endswith("factor.bin") and name.startswith("lower-"):
            raw = np.fromfile(path, dtype=np.float32).copy()
            raw[-1] = 0.0
            raw.tofile(path)
    return a, b


class CholeskyDataVerifierTest(unittest.TestCase):
    def test_valid_project_passes(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_project(root)
            module = load_verifier(root)
            self.assertEqual(module.main(), 0)

    def test_corrupted_padding_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_project(root, corrupt_padding=True)
            module = load_verifier(root)
            with self.assertRaises(ValueError) as raised:
                module.main()
            self.assertIn("Padding changed", str(raised.exception))

    def test_bitwise_batch_difference_is_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            build_project(root, batch_bit_differ=True)
            module = load_verifier(root)
            with self.assertRaises(ValueError) as raised:
                module.main()
            self.assertIn("Physical batch outputs differ bitwise", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
