# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate the default 255 x 255 positive-definite sample matrix."""

import argparse
import logging
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate a positive-definite spotrf input"
    )
    parser.add_argument(
        "--n", type=int, default=255, help="matrix dimension (default: 255)"
    )
    args = parser.parse_args()
    if not 1 <= args.n <= 8192:
        parser.error("--n must be between 1 and 8192")
    root = Path(__file__).resolve().parent
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "output").mkdir(parents=True, exist_ok=True)
    n = args.n  # Default matches spotrf_test.cpp.
    rng = np.random.RandomState(42)
    values = rng.standard_normal((n, n)).astype(np.float32)
    matrix = values @ values.T / n + np.eye(n, dtype=np.float32)
    matrix.astype(np.float32).tofile(root / "input" / "A_gm.bin")
    logging.info("Generated a %s x %s positive-definite matrix", n, n)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
