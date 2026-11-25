# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：verify_utils.verify_inverse 的精度判定（issue #634）。"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from verify_utils import verify_inverse


def _mismatch(index, expected, actual):
    return f"index {index}: expected {expected}, actual {actual}"


def test_identity_inverse_passes():
    n = 8
    a = np.eye(n, dtype=np.float32)
    inv = np.eye(n, dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        input_path = str(Path(tmp) / "input.bin")
        inv_path = str(Path(tmp) / "inv.bin")
        a.tofile(input_path)
        inv.tofile(inv_path)
        result = verify_inverse(input_path, inv_path, n, np.float32, _mismatch)
        assert result is True, "identity inverse should pass"


def test_wrong_inverse_fails():
    n = 8
    rng = np.random.RandomState(42)
    a = rng.rand(n, n).astype(np.float32) + np.eye(n, dtype=np.float32) * n
    wrong_inv = rng.rand(n, n).astype(np.float32)  # not the inverse
    with tempfile.TemporaryDirectory() as tmp:
        input_path = str(Path(tmp) / "input.bin")
        inv_path = str(Path(tmp) / "wrong.bin")
        a.tofile(input_path)
        wrong_inv.tofile(inv_path)
        result = verify_inverse(input_path, inv_path, n, np.float32, _mismatch)
        assert result is False, "wrong inverse should fail"
