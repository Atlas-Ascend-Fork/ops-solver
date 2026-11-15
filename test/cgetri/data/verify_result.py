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

import os
import sys

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
    # 数据定位对齐全仓契约：按 cwd 相对路径（其余 17 个 verify_result.py 同款），
    # 此前按 __file__ 定位是全仓唯一特例，host 测试被迫为 cgetri 单拷整套模块
    # （issue #448）。sys.path 的 utils 推导保留 __file__（模块导入，非数据定位）。
    sys.exit(
        run_verification(
            "./test/cgetri/data/input/A_gm.bin",
            "./test/cgetri/data/output/A_gm.bin",
            DTYPE_COMPLEX64,
            format_mismatch,
        )
    )
