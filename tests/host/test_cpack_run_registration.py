# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real CPack External and pinned makeself; the payload is an opaque Host fixture."""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

import cmake


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def package_tools():
    binaries = Path(cmake.CMAKE_BIN_DIR)
    makeself = Path(
        os.environ.get("SOLVER_MAKESELF_PATH", str(ROOT / "build/third_party/makeself"))
    )
    assert (makeself / "makeself.sh").is_file(), (
        "Provide repository-pinned makeself via SOLVER_MAKESELF_PATH"
    )
    return binaries, makeself


def write_host_project(directory, makeself):
    directory.mkdir()
    (directory / "payload.txt").write_text(
        "Opaque Host packaging fixture; not a compiled Solver library.\n"
    )
    (directory / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.19)\n"
        "project(solver_host_fixture LANGUAGES NONE)\n"
        'install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/payload.txt" DESTINATION lib64)\n'
        "set(CPACK_GENERATOR External)\n"
        "set(CPACK_PACKAGE_NAME solver_host_fixture)\n"
        "set(CPACK_PACKAGE_VERSION 1.0)\n"
        "set(CPACK_PACKAGE_FILE_NAME solver-host-fixture)\n"
        'set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}/output")\n'
        "set(CPACK_EXTERNAL_ENABLE_STAGING TRUE)\n"
        "set(CPACK_PACKAGE_CHECKSUM SHA256)\n"
        'set(CPACK_EXTERNAL_PACKAGE_SCRIPT "'
        + str(ROOT / "cmake/makeself.cmake")
        + '")\n'
        'set(CPACK_MAKESELF_PATH "' + str(makeself) + '")\n'
        "include(CPack)\n"
    )


def run_checked(command, directory):
    run = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr
    return run


def test_real_cpack_registers_run_and_its_checksum(package_tools, tmp_path):
    binaries, makeself = package_tools
    project = tmp_path / "project"
    write_host_project(project, makeself)
    # The real script needs its help/startup files present in the staging tree.
    with (project / "CMakeLists.txt").open("a") as handle:
        handle.write(
            'install(PROGRAMS "'
            + str(ROOT / "scripts/package/install.sh")
            + '" DESTINATION share/info/ops_solver)\n'
        )
        handle.write(
            'install(FILES "'
            + str(ROOT / "scripts/package/help.info")
            + '" DESTINATION share/info/ops_solver)\n'
        )
    build = project / "build"
    run_checked(
        [str(binaries / "cmake"), "-S", str(project), "-B", str(build)], project
    )
    run = run_checked(
        [str(binaries / "cpack"), "--config", str(build / "CPackConfig.cmake")], build
    )
    archive = build / "output/solver-host-fixture.run"
    checksum = archive.with_suffix(".run.sha256")
    assert archive.is_file()
    assert checksum.is_file(), (
        "Created run must also be registered with CPack, including its checksum"
    )
    assert (
        checksum.read_text().split()[0]
        == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert str(archive) + " generated." in run.stdout


@pytest.mark.parametrize("missing", ["makeself", "staging"])
def test_missing_real_packaging_inputs_still_fail(package_tools, tmp_path, missing):
    binaries, makeself = package_tools
    staging = tmp_path / "staging"
    if missing != "staging":
        staging.mkdir()
    if missing == "makeself":
        makeself = tmp_path / "missing-makeself"
    command = [
        str(binaries / "cmake"),
        "-DCPACK_MAKESELF_PATH=" + str(makeself),
        "-DCPACK_TEMPORARY_DIRECTORY=" + str(staging),
        "-DCPACK_PACKAGE_DIRECTORY=" + str(tmp_path / "output"),
        "-DCPACK_PACKAGE_FILE_NAME=fixture",
        "-P",
        str(ROOT / "cmake/makeself.cmake"),
    ]
    run = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert run.returncode != 0
    assert (
        "makeself.sh not found"
        if missing == "makeself"
        else "Staging directory not found"
    ) in run.stderr
