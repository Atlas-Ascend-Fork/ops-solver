# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Verify the default legacy spotrf sample by reconstructing its input."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np


def read_matrix(path, n):
    expected_bytes = n * n * np.dtype(np.float32).itemsize
    if path.stat().st_size != expected_bytes:
        raise ValueError("Unexpected matrix file size: {}".format(path))
    return np.fromfile(path, dtype=np.float32).reshape(n, n)


def main():
    parser = argparse.ArgumentParser(description="Verify a spotrf result")
    parser.add_argument(
        "--n", type=int, default=255, help="matrix dimension (default: 255)"
    )
    parser.add_argument(
        "--uplo", choices=("L", "U"), default="L", help="factor mode (default: L)"
    )
    args = parser.parse_args()
    if not 1 <= args.n <= 8192:
        parser.error("--n must be between 1 and 8192")
    root = Path(__file__).resolve().parent
    n = args.n
    matrix = read_matrix(root / "input" / "A_gm.bin", n)
    result = read_matrix(root / "output" / "A_gm.bin", n)
    # The legacy A2/A3 interface stores L^T in the upper triangle for uplo=L.
    factor = np.triu(result) if args.uplo == "L" else np.tril(result)
    reconstructed = factor.T @ factor if args.uplo == "L" else factor @ factor.T
    if (
        not np.isfinite(result).all()
        or np.any(np.diag(factor) <= 0)
        or not np.allclose(reconstructed, matrix, rtol=5e-3, atol=5e-3)
    ):
        logging.error("[Failed] Cholesky reconstruction does not match the input")
        return 1
    logging.info("[Success] Cholesky reconstruction matches the input")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        logging.error("[Failed] %s", error)
        sys.exit(1)
