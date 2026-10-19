# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Complete uninstall CLI; rm's failure is an explicit portable external boundary."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


FILES = {
    "library": "lib64/libops_solver.so",
    "header": "include/cann_ops_solver.h",
    "metadata": "share/info/ops_solver/version.info",
}


@pytest.fixture
def installation(tmp_path):
    entry = tmp_path / "package/share/info/ops_solver/install.sh"
    entry.parent.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    shutil.copyfile(root / "scripts/package/install.sh", entry)
    target = tmp_path / "target"
    paths = {}
    for kind, relative in FILES.items():
        path = target / "cann" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Opaque self-owned Host file, not an executable library.\n")
        paths[kind] = path
    return entry, target, paths


def rm_failure_boundary(tmp_path, suffix):
    real_rm = shutil.which("rm")
    assert real_rm
    tools = tmp_path / "tools"
    tools.mkdir()
    command = tools / "rm"
    command.write_text(
        "#!" + sys.executable + "\nimport os,sys\n"
        'suffix=os.environ.get("FAIL_DELETE_SUFFIX")\n'
        "if suffix and any(arg.endswith(suffix) for arg in sys.argv[1:]): sys.exit(17)\n"
        'os.execv(os.environ.get("ACTUAL_RM"), ["rm", *sys.argv[1:]])\n'
    )
    command.chmod(0o755)
    return dict(
        os.environ,
        PATH=str(tools) + os.pathsep + os.environ.get("PATH", ""),
        ACTUAL_RM=real_rm,
        FAIL_DELETE_SUFFIX=suffix,
    )


def run_uninstall(installation, env=None, quiet=True, answer=None):
    entry, target, _ = installation
    executable = shutil.which("bash")
    assert executable is not None and Path(executable).is_absolute()
    command = [executable, str(entry), "--uninstall", "--install-path=" + str(target)]
    if quiet:
        command.append("--quiet")
    return subprocess.run(
        command, env=env, input=answer, text=True, capture_output=True, check=False
    )


@pytest.mark.parametrize(
    "kind,suffix",
    [
        ("library", "libops_solver.so"),
        ("header", "cann_ops_solver.h"),
    ],
)
def test_failed_delete_does_not_report_success(installation, tmp_path, kind, suffix):
    run = run_uninstall(installation, rm_failure_boundary(tmp_path, suffix))
    path = installation[2].get(kind)
    assert path is not None and path.is_file()
    assert run.returncode != 0
    assert "uninstalled from" not in run.stdout
    assert "uninstall is incomplete" in run.stdout


def test_metadata_delete_failure_already_stops_uninstall(installation, tmp_path):
    run = run_uninstall(installation, rm_failure_boundary(tmp_path, "/ops_solver"))
    metadata = installation[2].get("metadata")
    assert metadata is not None and metadata.is_file()
    assert run.returncode != 0
    assert "uninstalled from" not in run.stdout


def test_success_keeps_unrelated_toolkit_files(installation):
    _, target, paths = installation
    unrelated = target / "cann/lib64/libunrelated.so"
    unrelated.write_text("Other package data\n")
    run = run_uninstall(installation)
    assert run.returncode == 0, run.stderr
    assert "uninstalled from" in run.stdout
    assert all(not path.exists() for path in paths.values())
    assert unrelated.read_text() == "Other package data\n"


def test_cancelled_uninstall_preserves_all_payloads(installation):
    run = run_uninstall(installation, quiet=False, answer="n\n")
    assert run.returncode == 0
    assert all(path.is_file() for path in installation[2].values())
    assert "uninstalled from" not in run.stdout


def test_absent_installation_remains_success(installation):
    _, target, _ = installation
    shutil.rmtree(target)
    run = run_uninstall(installation)
    assert run.returncode == 0
    assert "No installation found" in run.stdout
