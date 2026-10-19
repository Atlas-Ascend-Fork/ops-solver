# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the actual verifier on known valid/invalid Host fixture files."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "test/cheevj/data/verify_result.py"
SPEC = importlib.util.spec_from_file_location("real_cheevj_verifier", VERIFIER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def write_fixture(directory, n, basis):
    """A=I has W=1 and any orthonormal complex basis; no NPU was invoked."""
    data = directory / "test/cheevj/data"
    for name in ["input", "output", "golden"]:
        (data / name).mkdir(parents=True, exist_ok=True)
    np.eye(n, dtype=np.complex64).ravel(order="F").tofile(data / "input/A_gm.bin")
    np.ones(n, dtype=np.float32).tofile(data / "golden/W_golden.bin")
    np.ones(n, dtype=np.float32).tofile(data / "output/W_gm.bin")
    basis.astype(np.complex64).ravel(order="F").tofile(data / "output/V_gm.bin")


def actual_cli(directory, n):
    return subprocess.run(
        [sys.executable, str(VERIFIER), str(n), "V", "L", "identity"],
        cwd=directory,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "n,damage", [(128, "zero"), (256, "scale"), (256, "duplicate")]
)
def test_cli_rejects_invalid_basis_even_with_zero_eigen_residual(n, damage, tmp_path):
    basis = np.eye(n, dtype=np.complex64)
    if damage == "zero":
        basis[:, 0] = 0.0
    elif damage == "scale":
        basis[:, 0] *= 0.25
    else:
        basis[:, 1] = basis[:, 0]
    # A=I and W=1 make the eigen residual exactly zero even for these invalid
    # columns. A unitary eigenspace basis must still be rejected when damaged.
    write_fixture(tmp_path, n, basis)
    result = actual_cli(tmp_path, n)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "[Failed]" in result.stdout + result.stderr


@pytest.mark.parametrize("n", [8, 128])
def test_cli_accepts_orthonormal_complex_basis(n, tmp_path):
    basis = np.eye(n, dtype=np.complex64)
    basis[:2, :2] = np.asarray([[0.6j, 0.8j], [-0.8, 0.6]], dtype=np.complex64)
    write_fixture(tmp_path, n, basis)
    result = actual_cli(tmp_path, n)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[Success]" in result.stdout + result.stderr


def test_large_probe_branch_accepts_identity_basis():
    basis = np.eye(1025, dtype=np.complex64)
    assert MODULE.orthogonality_error(basis) <= MODULE.ORTHOGONAL_TOL


def test_large_probe_branch_rejects_zero_column():
    basis = np.eye(1025, dtype=np.complex64)
    basis[:, 0] = 0.0
    assert MODULE.orthogonality_error(basis) > MODULE.ORTHOGONAL_TOL


def test_nonfinite_basis_is_still_rejected(tmp_path):
    basis = np.eye(128, dtype=np.complex64)
    basis[0, 0] = np.nan
    write_fixture(tmp_path, 128, basis)
    result = actual_cli(tmp_path, 128)
    assert result.returncode == 1, result.stdout + result.stderr
