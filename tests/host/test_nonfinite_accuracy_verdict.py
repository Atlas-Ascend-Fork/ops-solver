# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run actual Solver CLIs on Host fixtures, never pretend to produce NPU output."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
OPERATORS = (
    "sgetrf",
    "cgetrf",
    "sgetri",
    "cgetri",
    "cgetri_batched",
    "cmatinv_batched",
)
CASES = (
    "identity",
    "valid_diagonal",
    "finite_outlier",
    "finite_large_error",
    "nan_output",
    "inf_output",
    "nan_input",
    "inf_input",
    "finite_overflow_product",
)


def make_inputs(operator):
    # Explicit finite Host fixtures, never claimed device kernel outputs.
    dtype = np.float32 if operator.startswith("s") else np.complex64
    batched = operator.endswith("batched")
    n, batch = (32, 512) if batched else (128, 1)
    a = np.repeat(np.eye(n, dtype=dtype)[np.newaxis, :, :], batch, axis=0)
    return a, a.copy(), n, batch


def apply_case(case, a, output, lu):
    if case == "valid_diagonal":
        a *= 2
        output *= 2 if lu else 0.5
    elif case == "finite_outlier":
        output[0, -1, -1] = 1.1
    elif case == "finite_large_error":
        output.fill(0)
    elif case in ("nan_output", "inf_output"):
        output[0, -1, -1] = float("nan" if case == "nan_output" else "inf")
    elif case in ("nan_input", "inf_input"):
        a[0, -1, -1] = float("nan" if case == "nan_input" else "inf")
    elif case == "finite_overflow_product":
        if lu:
            # Nonsingular A corner; deliberately invalid finite factors overflow.
            for matrix in (a, output):
                matrix[0, -1, 0] = 1e30
                matrix[0, 0, -1] = 1e30
        else:
            a[0, -1, -1] = 1e30
            output[0, -1, -1] = 1e30


def write_inputs(operator, a, output, directory):
    target = directory / "test" / operator / "data"
    (target / "input").mkdir(parents=True)
    (target / "output").mkdir()
    a.tofile(target / "input/A_gm.bin")
    output_file = "Ainv_gm.bin" if operator.endswith("batched") else "A_gm.bin"
    output.tofile(target / "output" / output_file)
    if operator.endswith("getrf"):
        np.arange(1, a.shape[-1] + 1, dtype=np.uint32).tofile(
            target / "output/W_gm.bin"
        )


def run_comparison(operator, n, batch, directory):
    arguments = (
        [str(n), str(n)]
        if operator.endswith("getrf")
        else [str(batch), str(n)]
        if operator.endswith("batched")
        else [str(n)]
    )
    command = [
        sys.executable,
        str(ROOT / "test" / operator / "data/verify_result.py"),
        *arguments,
    ]
    return subprocess.run(
        command,
        cwd=directory,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("operator", OPERATORS)
@pytest.mark.parametrize("case", CASES)
def test_nonfinite_accuracy_is_not_a_tolerated_error(operator, case, tmp_path):
    a, output, n, batch = make_inputs(operator)
    apply_case(case, a, output, operator.endswith("getrf"))
    write_inputs(operator, a, output, tmp_path)
    run = run_comparison(operator, n, batch, tmp_path)
    # Preserve the existing 1e-4 finite mismatch allowance and rejection.
    valid = case in ("identity", "valid_diagonal", "finite_outlier")
    expected = 0 if valid else 1
    assert run.returncode == expected, run.stdout + run.stderr
    assert ("[Success]" if valid else "[Failed]") in run.stdout + run.stderr
