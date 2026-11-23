# Copyright (c) 2026 Xztxzt
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute verbatim production scalar functions on CPU, not a SIMT simulator."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _acl_include(toolkit_roots):
    configured = os.environ.get("ACL_INCLUDE_PATH")
    if configured:
        include = Path(configured)
        assert (include / "acl/acl.h").is_file(), (
            f"ACL_INCLUDE_PATH={configured} does not contain acl/acl.h"
        )
        return str(include)
    suffixes = (
        "aarch64-linux/include",
        "x86_64-linux/include",
        "include",
        "runtime/include",
        "runtime/include/external",
    )
    for root in toolkit_roots:
        for suffix in suffixes:
            include = root / suffix
            if (include / "acl/acl.h").is_file():
                return str(include)
    return None


def scalar_source():
    kernel = (ROOT / "src/utils/complex_solver_kernel.hpp").read_text()
    register = (ROOT / "src/utils/complex_solver_register.hpp").read_text()
    start = kernel.index("template <typename Pointer>")
    end = kernel.index(
        "__simt_callee__ __aicore__ inline __gm__ aclFloatComplex* MatrixPointer"
    )
    column = kernel.index(
        "template <typename MatrixPointer>\n__simt_callee__ __aicore__ inline void DivideColumn"
    )
    column_end = kernel.index(
        "template <typename MatrixPointer>\n__simt_callee__ __aicore__ inline void UpdatePanel",
        column,
    )
    prepared = register.index("struct PreparedDivisor")
    prepared_end = register.index(
        "template <int Width>\n__simt_callee__ __aicore__ inline void RegisterSolve",
        prepared,
    )
    cached = (ROOT / "src/cgetrf/cgetrf_kernel.cpp").read_text()
    cached_start = cached.index(
        "__simt_callee__ __aicore__ inline aclFloatComplex NormalizeCachedValue"
    )
    cached_end = cached.index(
        "__simt_callee__ __aicore__ inline void ReduceCachedPivot", cached_start
    )
    return (
        kernel[start:end]
        + kernel[column:column_end]
        + register[prepared:prepared_end]
        + cached[cached_start:cached_end]
    )


@pytest.mark.parametrize("optimization", ("-O0", "-O2"))
def test_divide_and_prepared_divide_range(optimization, tmp_path, toolkit_roots):
    # Use actual SDK ABI header types. ACL_INCLUDE_PATH may point to an SDK or runtime/include/external.
    acl_include = _acl_include(toolkit_roots)
    if acl_include is None:
        pytest.skip(
            "acl/acl.h not found; set ACL_INCLUDE_PATH to a CANN include directory "
            "that contains acl/acl.h"
        )
    (tmp_path / "production.inc").write_text(scalar_source())
    binary = tmp_path / "range"
    command = [
        "g++",
        "-std=c++17",
        optimization,
        "-fno-fast-math",
        "-fsanitize=undefined",
        "-fno-sanitize-recover=all",
        "-I" + str(tmp_path),
        "-I" + str(ROOT / "include"),
        "-I" + acl_include,
        str(Path(__file__).with_name("complex_divide_range.cpp")),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    run = subprocess.run([str(binary)], text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stdout + run.stderr
