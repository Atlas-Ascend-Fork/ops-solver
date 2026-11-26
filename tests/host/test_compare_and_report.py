# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：lu_verify_utils.compare_and_report 的 eps 双条件比对（issue #639）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from lu_verify_utils import compare_and_report


def test_identical_passes():
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    result = compare_and_report(x.copy(), x, 4, 3)
    assert result is True


def test_within_eps_passes():
    x_ref = np.ones((4, 3), dtype=np.float32)
    x = x_ref + np.float32(1e-6)  # 远小于 eps=1e-3
    result = compare_and_report(x, x_ref, 4, 3)
    assert result is True


def test_large_diff_fails():
    x_ref = np.ones((4, 3), dtype=np.float32)
    x = x_ref * 10.0  # 远超 eps
    result = compare_and_report(x, x_ref, 4, 3)
    assert result is False


def test_relative_only_within_eps():
    """abs_err > eps 但 rel_err <= eps（大数小差异）——双条件要求同时超才计入。"""
    x_ref = np.full((4, 3), 1e6, dtype=np.float32)
    x = x_ref + 100.0  # abs_err=100 > eps；rel_err=100/1e6=1e-4 < eps → 不计入
    result = compare_and_report(x, x_ref, 4, 3)
    assert result is True


def test_absolute_only_within_eps():
    """rel_err > eps 但 abs_err <= eps（小数大比例）——双条件要求同时超才计入。"""
    x_ref = np.full((4, 3), 1e-8, dtype=np.float32)
    x = x_ref * 10.0  # rel_err=9e-1 > eps；abs_err=9e-8 < eps → 不计入
    result = compare_and_report(x, x_ref, 4, 3)
    assert result is True


def test_mixed_partial_elements():
    """部分元素超差、部分通过——任一元素双条件超即 FAIL。"""
    x_ref = np.ones((4, 3), dtype=np.float32)
    x = x_ref.copy()
    x[0, 0] = 10.0  # 该元素 abs=9>eps 且 rel=9>eps → 超差
    x[1, 1] = 1.001  # 该元素 abs=0.001<eps → 通过
    x[2, 2] = 1e6    # abs 大但 rel = (1e6-1)/1=999999>eps → 超差
    result = compare_and_report(x, x_ref, 4, 3)
    assert result is False


def test_mixed_within_ratio():
    """部分超差但占比在容差内（error_ratio <= error_tol）——此时因全量比对应 FAIL。

    compare_and_report 无 error_tol 参数（返回 bool），全量比对任一超差即 False。
    """
    x_ref = np.ones((4, 3), dtype=np.float32)
    x = x_ref.copy()
    x[0, 0] = 2.0  # 唯一超差元素
    result = compare_and_report(x, x_ref, 4, 3)
    assert result is False
