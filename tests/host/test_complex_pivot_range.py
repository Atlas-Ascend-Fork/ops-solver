# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Production pivot helpers with CPU thread exchanges, not device simulation."""

import shutil
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def pivot_source():
    kernel = (ROOT / "src/utils/complex_solver_kernel.hpp").read_text()
    end = kernel.index(
        "template <typename MatrixPointer, typename PivotPointer, typename InfoPointer>\n"
        "__simt_callee__ __aicore__ inline void FactorPanel"
    )
    start = kernel.index("constexpr uint32_t THREADS =")
    body = kernel[start:end]
    register = (ROOT / "src/utils/complex_solver_register.hpp").read_text()
    start = register.index("template <int Order>")
    end = register.index(
        "template <int Width>\n__simt_callee__ __aicore__ inline void RegisterFactor"
    )
    body += register[start:end]
    row = (ROOT / "src/utils/complex_solver_row_register.hpp").read_text()
    start = row.index(
        "template <int Width>\n__simt_callee__ __aicore__ __attribute__((always_inline)) inline int DirectRowPivot"
    )
    end = row.index(
        "template <int Width>\n__simt_callee__ __aicore__ __attribute__((always_inline)) inline int InvertDirectRows"
    )
    body += row[start:end]
    cached = (ROOT / "src/cgetrf/cgetrf_kernel.cpp").read_text()
    start = cached.index("__simt_callee__ __aicore__ inline void ReduceCachedPivot")
    end = cached.index("__simt_callee__ __aicore__ inline void FactorCachedColumn")
    body += cached[start:end]
    return body


def test_pivot_ranges_across_production_paths(tmp_path):
    (tmp_path / "production.inc").write_text(pivot_source())
    compiler = shutil.which("g++")
    assert compiler is not None
    acl = os.environ.get("ACL_INCLUDE_PATH")
    assert acl and (Path(acl) / "acl/acl.h").is_file()
    binary = tmp_path / "pivot"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fno-fast-math",
            "-pthread",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "include"),
            "-I" + acl,
            str(Path(__file__).with_name("complex_pivot_range.cpp")),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    run = subprocess.run(
        [str(binary)], text=True, capture_output=True, check=False, timeout=30
    )
    assert run.returncode == 0, run.stdout + run.stderr
