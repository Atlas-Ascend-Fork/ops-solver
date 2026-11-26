# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：lu_verify_utils.l1norm 的矩阵范数（issue #643）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from lu_verify_utils import l1norm


def test_l1norm_identity():
    assert l1norm(np.eye(3, dtype=np.float32)) == 1.0


def test_l1norm_known():
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # 列绝对值和的最大值：max(1+3, 2+4) = 6
    assert l1norm(a) == 6.0


def test_l1norm_zero():
    assert l1norm(np.zeros((3, 3), dtype=np.float32)) == 0.0


def test_l1norm_negative():
    a = np.array([[-1, -2], [-3, -4]], dtype=np.float32)
    assert l1norm(a) == 6.0  # 绝对值
