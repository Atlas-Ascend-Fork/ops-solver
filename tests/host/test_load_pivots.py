# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：lu_verify_utils.load_pivots 的主元校验（issue #636）。"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from lu_verify_utils import load_pivots


def _write_pivots(values, path):
    np.asarray(values, dtype=np.int32).tofile(path)


def test_valid_pivots():
    n = 4
    raw = [1, 2, 3, 4]  # 1-based identity
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "W.bin")
        _write_pivots(raw, path)
        piv = load_pivots(path, n)
        np.testing.assert_array_equal(piv, [0, 1, 2, 3])  # 0-based


def test_swapped_pivots():
    n = 4
    raw = [2, 3, 3, 4]  # 1-based LAPACK ipiv：部分主元合法序列
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "W.bin")
        _write_pivots(raw, path)
        piv = load_pivots(path, n)
        np.testing.assert_array_equal(piv, [1, 2, 2, 3])


def test_zero_pivot_rejected():
    n = 4
    raw = [0, 2, 3, 4]  # 0 is invalid (1-based contract)
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "W.bin")
        _write_pivots(raw, path)
        with pytest.raises(ValueError, match="invalid pivot at index 0: 0"):
            load_pivots(path, n)


def test_out_of_range_pivot_rejected():
    n = 4
    raw = [1, 2, 3, 5]  # 5 > n=4
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "W.bin")
        _write_pivots(raw, path)
        with pytest.raises(ValueError, match="invalid pivot at index 3: 5"):
            load_pivots(path, n)
