# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：verify_utils.run_verification 的 CLI 契约（issue #654）。

覆盖 build.sh 三段式 run 流程消费的退出码语义：阶数缺省 32、显式阶数
参数、非整数参数与阶数不匹配异常均返回 1，校验通过返回 0。
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from verify_utils import run_verification


def _mismatch(index, expected, actual):
    return f"index {index}: expected {expected}, actual {actual}"


def _write_pair(tmp, order, inv_scale):
    a = np.eye(order, dtype=np.float32) * 2
    inv = np.eye(order, dtype=np.float32) * inv_scale
    input_path = Path(tmp) / "input.bin"
    inv_path = Path(tmp) / "inv.bin"
    a.tofile(input_path)
    inv.tofile(inv_path)
    return str(input_path), str(inv_path)


def test_default_order_passes(monkeypatch):
    order = 32
    with tempfile.TemporaryDirectory() as tmp:
        input_path, inv_path = _write_pair(tmp, order, 0.5)
        monkeypatch.setattr(sys, "argv", ["verify_result.py"])
        assert run_verification(input_path, inv_path, np.float32, _mismatch) == 0


def test_explicit_order_passes(monkeypatch):
    order = 8
    with tempfile.TemporaryDirectory() as tmp:
        input_path, inv_path = _write_pair(tmp, order, 0.5)
        monkeypatch.setattr(sys, "argv", ["verify_result.py", str(order)])
        assert run_verification(input_path, inv_path, np.float32, _mismatch) == 0


def test_wrong_inverse_fails(monkeypatch):
    order = 8
    with tempfile.TemporaryDirectory() as tmp:
        input_path, inv_path = _write_pair(tmp, order, 3.0)
        monkeypatch.setattr(sys, "argv", ["verify_result.py", str(order)])
        assert run_verification(input_path, inv_path, np.float32, _mismatch) == 1


def test_non_integer_order_argument_fails(monkeypatch):
    order = 8
    with tempfile.TemporaryDirectory() as tmp:
        input_path, inv_path = _write_pair(tmp, order, 0.5)
        monkeypatch.setattr(sys, "argv", ["verify_result.py", "abc"])
        assert run_verification(input_path, inv_path, np.float32, _mismatch) == 1


def test_order_mismatch_fails(monkeypatch):
    order = 8
    with tempfile.TemporaryDirectory() as tmp:
        input_path, inv_path = _write_pair(tmp, order, 0.5)
        monkeypatch.setattr(sys, "argv", ["verify_result.py", "16"])
        assert run_verification(input_path, inv_path, np.float32, _mismatch) == 1


def test_nonfinite_output_fails(monkeypatch):
    order = 8
    with tempfile.TemporaryDirectory() as tmp:
        input_path, inv_path = _write_pair(tmp, order, 0.5)
        broken = np.fromfile(inv_path, dtype=np.float32).copy()
        broken[0] = np.nan
        broken.tofile(inv_path)
        monkeypatch.setattr(sys, "argv", ["verify_result.py", str(order)])
        assert run_verification(input_path, inv_path, np.float32, _mismatch) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
