# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：verify_utils.read_binary_strict 的四场景（issue #633）。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from verify_utils import read_binary_strict


def test_valid_file(tmp_path):
    data = np.arange(8, dtype=np.float32)
    path = str(tmp_path / "valid.bin")
    data.tofile(path)
    result = read_binary_strict(path, np.float32, 8)
    np.testing.assert_array_equal(result, data)


def test_short_bytes_rejected(tmp_path):
    data = np.arange(8, dtype=np.float32)
    path = str(tmp_path / "short.bin")
    data[:6].tofile(path)  # 24 bytes（非 4 的整数倍？24/4=6 是整数——用 7 元素即 28B）
    # 实际测非整数倍字节
    with open(path, "wb") as f:
        f.write(data[:6].tobytes()[:22])  # 22 bytes / 4B = 5.5
    with pytest.raises(ValueError, match="not a multiple of"):
        read_binary_strict(path, np.float32)


def test_extra_bytes_rejected(tmp_path):
    data = np.arange(8, dtype=np.float32)
    path = str(tmp_path / "extra.bin")
    data.tofile(path)
    with open(path, "ab") as f:
        f.write(b"\x00\x00")  # 2 extra bytes
    with pytest.raises(ValueError, match="not a multiple of"):
        read_binary_strict(path, np.float32)


def test_wrong_count_rejected(tmp_path):
    data = np.arange(8, dtype=np.float32)
    path = str(tmp_path / "count.bin")
    data.tofile(path)
    with pytest.raises(ValueError, match="expected 10 elements, got 8"):
        read_binary_strict(path, np.float32, 10)


def test_expected_count_none_skips_check(tmp_path):
    data = np.arange(8, dtype=np.float32)
    path = str(tmp_path / "none.bin")
    data.tofile(path)
    result = read_binary_strict(path, np.float32)  # no expected_count
    assert result.size == 8
