# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run actual Solver CLIs on Host fixtures, never pretend to produce NPU output."""

import os
import runpy
import shutil
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
    script = ROOT / "test" / operator / "data/verify_result.py"
    if operator == "cgetri":
        # The current CLI locates its data beside __file__; copy full production modules.
        script = directory / "test" / operator / "data/verify_result.py"
        shutil.copyfile(ROOT / "test" / operator / "data/verify_result.py", script)
        utilities = directory / "test/utils"
        utilities.mkdir(exist_ok=True)
        shutil.copyfile(
            ROOT / "test/utils/verify_utils.py", utilities / "verify_utils.py"
        )
    command = [sys.executable, str(script), *arguments]
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


@pytest.mark.parametrize("sign", (1, -1))
@pytest.mark.parametrize("steps", (0, 1, 32, 64))
def test_qr_ulp_cli_keeps_a_finite_hard_limit(sign, steps, tmp_path):
    """Use controlled finite QR files; no device output is produced by this test."""
    n = 16
    maximum = np.float32(np.finfo(np.float32).max * sign)
    a = np.eye(n, dtype=np.complex64) * maximum
    result = a.copy()
    changed = maximum
    for _ in range(steps):
        changed = np.nextafter(changed, np.float32(0))
    result[0, 0] = changed
    data = tmp_path / "test/cgeqrf/data"
    (data / "input").mkdir(parents=True)
    (data / "output").mkdir()
    a.T.tofile(data / "input/A_gm.bin")
    result.T.tofile(data / "output/R_gm.bin")
    np.zeros(n, dtype=np.complex64).tofile(data / "output/tau.bin")
    # Q is exactly the identity (zero tau); A has condition number one.
    run = subprocess.run(
        [
            sys.executable,
            str(ROOT / "test/cgeqrf/data/verify_result.py"),
            str(n),
            str(n),
        ],
        cwd=tmp_path,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    valid = steps <= 32
    assert run.returncode == (0 if valid else 1), run.stdout + run.stderr
    assert ("[Success]" if valid else "[Failed]") in run.stdout + run.stderr


@pytest.mark.parametrize(
    ("value", "expected"),
    (
        (np.finfo(np.float32).max, np.ldexp(1.0, 104)),
        (-np.finfo(np.float32).max, np.ldexp(1.0, 104)),
        (1.0, np.ldexp(1.0, -23)),
        (-1.0, np.ldexp(1.0, -23)),
        (0.0, np.ldexp(1.0, -149)),
    ),
)
def test_qr_ulp_spacing_matches_float32_grid(value, expected):
    module = runpy.run_path(str(ROOT / "test/cgeqrf/data/verify_result.py"))
    assert module["ulp_f32"](value) == expected
