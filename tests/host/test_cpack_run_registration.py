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

try:
    from cmake import CMAKE_BIN_DIR
except ImportError:  # Repo-root cmake/ shadows the package when absent.
    CMAKE_BIN_DIR = None


ROOT = Path(__file__).resolve().parents[2]

CMAKE_BINARIES_MISSING = (
    "cmake and cpack not found; install requirements.txt or add them to PATH"
)
MAKESELF_MISSING = (
    "makeself.sh not found; set SOLVER_MAKESELF_PATH to a directory containing "
    "makeself.sh"
)


def _cmake_tools_run(binaries):
    for name in ("cmake", "cpack"):
        run = subprocess.run(
            [str(binaries / name), "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        if run.returncode != 0:
            return False
    return True


def _cmake_binaries():
    candidates = []
    if CMAKE_BIN_DIR:
        candidates.append(Path(CMAKE_BIN_DIR))
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        if directory:
            candidates.append(Path(directory))
    for binaries in candidates:
        if (binaries / "cmake").is_file() and (binaries / "cpack").is_file():
            if _cmake_tools_run(binaries):
                return binaries
    return None


def _makeself_directory(toolkit_roots):
    configured = os.environ.get("SOLVER_MAKESELF_PATH")
    if configured:
        makeself = Path(configured)
        assert (makeself / "makeself.sh").is_file(), (
            f"SOLVER_MAKESELF_PATH={configured} does not contain makeself.sh"
        )
        return makeself
    candidates = [ROOT / "build/third_party/makeself"]
    for root in toolkit_roots:
        for arch in ("aarch64-linux", "x86_64-linux"):
            candidates.append(
                root / arch / "tikcpp/ascendc_kernel_cmake/fwk_modules/util/makeself"
            )
        candidates.append(
            root / "tools/op_project_templates/op_project_tmpl/cmake/util/makeself"
        )
    for makeself in candidates:
        if (makeself / "makeself.sh").is_file():
            return makeself
    return None


@pytest.fixture
def cmake_binaries():
    return _cmake_binaries()


@pytest.fixture
def makeself_directory(toolkit_roots):
    return _makeself_directory(toolkit_roots)


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


def test_real_cpack_registers_run_and_its_checksum(
    cmake_binaries, makeself_directory, tmp_path
):
    if cmake_binaries is None:
        pytest.skip(CMAKE_BINARIES_MISSING)
    if makeself_directory is None:
        pytest.skip(MAKESELF_MISSING)
    binaries = cmake_binaries
    makeself = makeself_directory
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
def test_missing_real_packaging_inputs_still_fail(
    cmake_binaries, toolkit_roots, tmp_path, missing
):
    if cmake_binaries is None:
        pytest.skip(CMAKE_BINARIES_MISSING)
    binaries = cmake_binaries
    staging = tmp_path / "staging"
    if missing != "staging":
        staging.mkdir()
    if missing == "makeself":
        makeself = tmp_path / "missing-makeself"
    else:
        makeself = _makeself_directory(toolkit_roots)
        if makeself is None:
            pytest.skip(MAKESELF_MISSING)
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
