# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the entire SSgels test TU; transport and outputs are CPU fixtures."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).with_name("sgels_accuracy")


@pytest.mark.parametrize("optimization", ("-O0", "-O2"))
def test_nonfinite_solutions_and_residuals_fail(optimization, tmp_path):
    compiler = shutil.which("g++") or shutil.which("c++")
    assert compiler is not None, "Host C++17 compiler required"
    binary = tmp_path / "verdict"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            optimization,
            "-fno-fast-math",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-ffunction-sections",
            "-fdata-sections",
            "-I" + str(FIXTURE),
            "-I" + str(ROOT / "include"),
            "-I" + str(ROOT / "test/sgels"),
            str(FIXTURE / "verdict_test.cpp"),
            "-Wl,--gc-sections",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run = subprocess.run([str(binary)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "cases=40 failures=0" in run.stdout
