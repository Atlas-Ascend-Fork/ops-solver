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

import logging
import os
import sys

# 日志流显式声明（issue #662）：verify_utils 不再在导入期 basicConfig，
# 本脚本的输出流由自身决定（默认 stderr，与既有行为一致）。
logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
)
from verify_utils import DTYPE_COMPLEX64, run_verification  # noqa: E402


def format_mismatch(index, expected, actual):
    rdiff = abs(actual - expected) / abs(expected)
    return (
        f"data index: {index:08d}, expected: {expected.real:.9f}{expected.imag:+.9f}j, "
        f"actual: {actual.real:.9f}{actual.imag:+.9f}j, rdiff: {rdiff:.6f}"
    )


if __name__ == "__main__":
    # 数据定位保持 cwd 相对路径（issue #448 口径）：host 测试
    # （tests/host/test_nonfinite_accuracy_verdict.py）以 tmp_path 为 cwd 构造数据
    # 并运行本脚本——若按 __file__ 解析会去脚本所在目录找 tmp_path 写的 input/（不存在）。
    # 本算子与其余 OPERATORS 算子、cheevj basis-verdict 均因 host 测试
    # cwd 数据流而保持 cwd 相对（host 测试以 tmp_path 为 cwd 构造数据）。
    # sys.path 的 utils 推导保留 __file__（模块导入，非数据定位）。
    sys.exit(
        run_verification(
            "./test/cgetri/data/input/A_gm.bin",
            "./test/cgetri/data/output/A_gm.bin",
            DTYPE_COMPLEX64,
            format_mismatch,
        )
    )
