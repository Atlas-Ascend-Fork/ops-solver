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
from verify_utils import DTYPE_FLOAT32, run_verification  # noqa: E402


def format_mismatch(index, expected, actual):
    rdiff = abs(actual - expected) / abs(expected)
    return f"data index: {index:08d}, expected: {expected:.9f}, actual: {actual:.9f}, rdiff: {rdiff:.6f}"


# 数据定位保持 cwd 相对路径（host 测试以 tmp_path 为 cwd 构造数据并运行，issue #448 口径）。
if __name__ == "__main__":
    sys.exit(
        run_verification(
            "./test/sgetri/data/input/A_gm.bin",
            "./test/sgetri/data/output/A_gm.bin",
            DTYPE_FLOAT32,
            format_mismatch,
        )
    )
