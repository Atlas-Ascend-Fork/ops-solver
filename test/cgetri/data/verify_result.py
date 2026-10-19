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
    data_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(data_dir, "input", "A_gm.bin")
    output_path = os.path.join(data_dir, "output", "A_gm.bin")

    sys.exit(
        run_verification(
            input_path,
            output_path,
            DTYPE_COMPLEX64,
            format_mismatch,
        )
    )
