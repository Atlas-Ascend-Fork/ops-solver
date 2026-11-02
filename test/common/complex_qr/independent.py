# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import argparse
import numpy as np
from runtime import Runtime
from test_support import report


def matrix(rng, rows, columns):
    return np.asfortranarray(
        (
            rng.normal(size=(rows, columns)) + 1j * rng.normal(size=(rows, columns))
        ).astype(np.complex64)
    )


def run(test_operator):
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    rt = Runtime(args.library, args.device)
    try:
        test_operator(rt)
    finally:
        rt.close()
    report("Independent operator tests passed")
