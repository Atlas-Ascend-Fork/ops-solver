# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Complete build.sh with only the external CMake process as an argv boundary."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


def bash_executable():
    executable = shutil.which("bash")
    assert executable is not None and Path(executable).is_absolute()
    return executable


@pytest.fixture
def driver(tmp_path):
    def prepare(kind, failure=""):
        repo = tmp_path / ("source with spaces" if kind == "source" else "source")
        repo.mkdir()
        original = Path(__file__).resolve().parents[2]
        for filename in ["build.sh", "CMakeLists.txt"]:
            shutil.copyfile(original / filename, repo / filename)
        sdk = tmp_path / ("sdk with spaces" if kind in {"sdk", "prefix"} else "sdk")
        sdk.mkdir()
        prefix = sdk / "lib64/cmake"
        if kind == "prefix":
            (prefix / "ASC").mkdir(parents=True)
            (prefix / "ASC/ASCConfig.cmake").write_text(
                "# File-discovery boundary; not an SDK.\n"
            )
        tools = tmp_path / "tools"
        tools.mkdir()
        recorder = tools / "cmake"
        recorder.write_text(
            "#!" + sys.executable + "\nimport json,os,sys\n"
            'with open(os.environ["ARGV_LOG"], "a") as handle: '
            'handle.write(json.dumps(sys.argv[1:])+"\\n")\n'
            'sys.exit(int(os.environ.get("BOUNDARY_FAILURE", "0")) '
            'if os.environ.get("FAIL_STAGE") in sys.argv else 0)\n'
        )
        recorder.chmod(0o755)
        env = dict(
            os.environ,
            PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
            ASCEND_HOME_PATH=str(sdk),
            ASCEND_TOOLKIT_HOME="",
            ASC_DIR="",
            ARGV_LOG=str(tmp_path / "argv.jsonl"),
            FAIL_STAGE=failure,
            BOUNDARY_FAILURE="17",
        )
        if kind == "asc_dir":
            env["ASC_DIR"] = str(tmp_path / "asc directory")
        return repo, sdk, prefix, env

    return prepare


def recorded_calls(env):
    path = Path(env.get("ARGV_LOG"))
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


@pytest.mark.parametrize("kind", ["plain", "source", "sdk", "asc_dir", "prefix"])
@pytest.mark.parametrize("pack", [False, True])
def test_complete_driver_preserves_paths_and_cmake_list(driver, kind, pack):
    repo, sdk, prefix, env = driver(kind)
    command = [
        bash_executable(),
        str(repo / "build.sh"),
        "--soc=ascend950",
        "--ops=sgetrf,cgetrf",
    ]
    if pack:
        command.append("--pkg")
    run = subprocess.run(
        command, cwd=repo.parent, env=env, text=True, capture_output=True, check=False
    )
    assert run.returncode == 0, run.stderr
    calls = recorded_calls(env)
    assert len(calls) == 3
    configure = calls[0]
    assert configure[:4] == ["-S", str(repo), "-B", str(repo / "build")]
    assert "-DASCEND_CANN_PACKAGE_PATH=" + str(sdk) in configure
    assert "-DTEST_NAMES=sgetrf;cgetrf" in configure
    assert calls[1] == ["--build", str(repo / "build"), "-j"]
    assert calls[2] == (
        ["--build", str(repo / "build"), "--target", "package"]
        if pack
        else ["--install", str(repo / "build")]
    )
    if kind == "asc_dir":
        assert "-DASC_DIR=" + env.get("ASC_DIR") in configure
    if kind == "prefix":
        assert "-DCMAKE_PREFIX_PATH=" + str(prefix) in configure
    if pack:
        assert "-DCANN_3RD_LIB_PATH=" + str(repo / "build/third_party") in configure


@pytest.mark.parametrize(
    "stage,expected_calls", [("-S", 1), ("--build", 2), ("--install", 3)]
)
def test_cmake_failure_still_stops_driver(driver, stage, expected_calls):
    repo, _, _, env = driver("plain", stage)
    run = subprocess.run(
        [bash_executable(), str(repo / "build.sh")],
        cwd=repo.parent,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 17
    assert len(recorded_calls(env)) == expected_calls
