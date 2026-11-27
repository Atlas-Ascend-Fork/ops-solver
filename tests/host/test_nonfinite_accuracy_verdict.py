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
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "test" / "utils"))
from lu_verify_utils import solve_lu  # noqa: E402

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
    # 参数序与 #377 修正后的 verify 脚本对齐：(n, batchSize)——此前按 (batch, n)
    # 传参使两个 batched 算子的 18 个用例因 reshape 崩溃全红（issue #447）。
    arguments = (
        [str(n), str(n)]
        if operator.endswith("getrf")
        else [str(n), str(batch)]
        if operator.endswith("batched")
        else [str(n)]
    )
    script = ROOT / "test" / operator / "data/verify_result.py"
    # cgetri 的数据定位已对齐全仓 cwd 相对路径契约（issue #448），不再需要
    # 按 __file__ 特例拷贝整套模块到临时目录。
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


CGETRS_SCRIPT = (
    Path(__file__).resolve().parents[2] / "test/cgetrs/data/verify_result.py"
)
SGETRS_SCRIPT = (
    Path(__file__).resolve().parents[2] / "test/sgetrs/data/verify_result.py"
)
CPOTRF_SCRIPT = (
    Path(__file__).resolve().parents[2] / "test/cpotrf/data/verify_result.py"
)


def identity_inputs(n=2, nrhs=1, dtype=np.complex64):
    lu = np.eye(n, dtype=dtype)
    rhs = np.ones((n, nrhs), dtype=dtype)
    return {
        "LU": lu,
        "B": rhs,
        "X": rhs.copy(),
        "piv": np.arange(1, n + 1, dtype=np.int32),
    }


def run_cli(directory, arrays, operator="cgetrs"):
    data = directory / f"test/{operator}/data"
    (data / "input").mkdir(parents=True, exist_ok=True)
    (data / "output").mkdir(exist_ok=True)
    for name, location in {
        "LU": "input/A_gm.bin",
        "B": "input/B_gm.bin",
        "X": "output/X_gm.bin",
        "piv": "input/W_gm.bin",
    }.items():
        if arrays.get(name) is not None:
            arrays[name].tofile(data / location)
    n = arrays["LU"].shape[0]
    nrhs = arrays["B"].shape[1]
    script = SGETRS_SCRIPT if operator == "sgetrs" else CGETRS_SCRIPT
    return subprocess.run(
        [sys.executable, str(script), str(n), str(nrhs)],
        cwd=directory,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("field", ("LU", "B", "X"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_cgetrs_nonfinite_input_or_solution_is_rejected(tmp_path, field, bad):
    arrays = identity_inputs()
    arrays[field][0, 0] = bad
    run = run_cli(tmp_path, arrays)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize("case", ("overflow", "zero_division"))
def test_cgetrs_nonfinite_reference_from_finite_arrays_is_rejected(tmp_path, case):
    arrays = identity_inputs()
    arrays["LU"][0, 0] = 1e-30 if case == "overflow" else 0
    arrays["B"][0, 0] = 1e30 if case == "overflow" else 0
    arrays["X"].fill(0)
    assert all(np.isfinite(value).all() for value in arrays.values())
    run = run_cli(tmp_path, arrays)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Non-finite reference solution" in run.stdout
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize(
    "case", ("identity", "zero_rhs", "complex", "small_error", "large_error", "pivot")
)
def test_cgetrs_finite_reference_preserves_accuracy_verdict(tmp_path, case):
    arrays = identity_inputs()
    if case == "zero_rhs":
        arrays["B"].fill(0)
        arrays["X"].fill(0)
    elif case == "complex":
        arrays["LU"] *= 2 + 1j
        arrays["B"] *= 2 + 1j
    elif case in ("small_error", "large_error"):
        arrays["X"][0, 0] += 5e-4 if case == "small_error" else 0.1
    elif case == "pivot":
        arrays["LU"] = np.array([[1, 1], [0, 2]], dtype=np.complex64)
        arrays["B"] = np.array([[4], [3]], dtype=np.complex64)
        arrays["X"] = np.array([[1], [2]], dtype=np.complex64)
        arrays["piv"] = np.array([2, 2], dtype=np.int32)
    run = run_cli(tmp_path, arrays)
    passed = case != "large_error"
    assert run.returncode == (0 if passed else 1), run.stdout + run.stderr
    assert ("[Success]" if passed else "[Failed]") in run.stdout


def test_cgetrs_existing_finite_mismatch_fraction_is_preserved(tmp_path):
    arrays = identity_inputs(128, 128)
    arrays["X"][0, 0] += 0.1
    run = run_cli(tmp_path, arrays)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_cgetrs_nonfinite_solution_is_not_diluted_by_mismatch_fraction(tmp_path, bad):
    arrays = identity_inputs(128, 128)
    arrays["X"][0, 0] = bad
    run = run_cli(tmp_path, arrays)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize("field", ("LU", "B", "X"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_sgetrs_nonfinite_input_or_solution_is_rejected(tmp_path, field, bad):
    arrays = identity_inputs(dtype=np.float32)
    arrays[field][0, 0] = bad
    run = run_cli(tmp_path, arrays, operator="sgetrs")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize("case", ("overflow", "zero_division"))
def test_sgetrs_nonfinite_reference_from_finite_arrays_is_rejected(tmp_path, case):
    arrays = identity_inputs(dtype=np.float32)
    arrays["LU"][0, 0] = 1e-30 if case == "overflow" else 0
    arrays["B"][0, 0] = 1e30 if case == "overflow" else 0
    arrays["X"].fill(0)
    assert all(np.isfinite(value).all() for value in arrays.values())
    run = run_cli(tmp_path, arrays, operator="sgetrs")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Non-finite reference solution" in run.stdout
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize(
    "case", ("identity", "zero_rhs", "small_error", "large_error", "pivot")
)
def test_sgetrs_finite_reference_preserves_accuracy_verdict(tmp_path, case):
    arrays = identity_inputs(dtype=np.float32)
    if case == "zero_rhs":
        arrays["B"].fill(0)
        arrays["X"].fill(0)
    elif case in ("small_error", "large_error"):
        arrays["X"][0, 0] += 5e-4 if case == "small_error" else 0.1
    elif case == "pivot":
        arrays["LU"] = np.array([[1, 1], [0, 2]], dtype=np.float32)
        arrays["B"] = np.array([[4], [3]], dtype=np.float32)
        arrays["X"] = np.array([[1], [2]], dtype=np.float32)
        arrays["piv"] = np.array([2, 2], dtype=np.int32)
    run = run_cli(tmp_path, arrays, operator="sgetrs")
    passed = case != "large_error"
    assert run.returncode == (0 if passed else 1), run.stdout + run.stderr
    assert ("[Success]" if passed else "[Failed]") in run.stdout


def test_sgetrs_existing_finite_mismatch_fraction_is_preserved(tmp_path):
    arrays = identity_inputs(128, 128, dtype=np.float32)
    arrays["X"][0, 0] += 0.1
    run = run_cli(tmp_path, arrays, operator="sgetrs")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_sgetrs_nonfinite_solution_is_not_diluted_by_mismatch_fraction(tmp_path, bad):
    arrays = identity_inputs(128, 128, dtype=np.float32)
    arrays["X"][0, 0] = bad
    run = run_cli(tmp_path, arrays, operator="sgetrs")
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout


def write_cpotrf_matrix(path, matrix):
    np.stack((matrix.real, matrix.imag), axis=-1).astype(np.float32).tofile(path)


def run_cpotrf_cli(directory, upper, gm, std, n):
    data = directory / "test/cpotrf/data/output"
    data.mkdir(parents=True)
    clean = np.triu(np.ones((n, n), dtype=np.complex64) * 3)
    prefix, other = ("U", "L") if upper else ("L", "U")
    write_cpotrf_matrix(data / f"{prefix}_gm.bin", gm)
    write_cpotrf_matrix(data / f"{prefix}_std.bin", std)
    write_cpotrf_matrix(data / f"{other}_gm.bin", clean)
    write_cpotrf_matrix(data / f"{other}_std.bin", clean)
    return subprocess.run(
        [sys.executable, str(CPOTRF_SCRIPT), str(n)],
        cwd=directory,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("upper", (True, False))
@pytest.mark.parametrize("field", ("gm", "std"))
@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_cpotrf_nonfinite_output_or_reference_is_rejected(tmp_path, upper, field, bad):
    n = 4
    factor = np.triu(np.ones((n, n), dtype=np.complex64) * 3)
    gm = factor.copy()
    std = factor.copy()
    if field == "gm":
        gm[0, 1] = bad
    else:
        std[0, 1] = bad
    run = run_cpotrf_cli(tmp_path, upper, gm, std, n)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Non-finite output or reference" in run.stdout
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize("bad", (float("nan"), float("inf")))
def test_cpotrf_nonfinite_solution_is_not_diluted_by_mismatch_fraction(tmp_path, bad):
    n = 256
    factor = np.triu(np.ones((n, n), dtype=np.complex64) * 3)
    gm = factor.copy()
    gm[0, 1] = bad
    run = run_cpotrf_cli(tmp_path, True, gm, factor, n)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "Non-finite output or reference" in run.stdout
    assert "[Failed]" in run.stdout


def test_cpotrf_existing_finite_mismatch_fraction_is_preserved(tmp_path):
    n = 256
    factor = np.triu(np.ones((n, n), dtype=np.complex64) * 3)
    gm = factor.copy()
    gm[0, 1] += 1e-3
    run = run_cpotrf_cli(tmp_path, True, gm, factor, n)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout


def test_cpotrf_finite_match_passes(tmp_path):
    n = 4
    factor = np.triu(np.ones((n, n), dtype=np.complex64) * 3)
    run = run_cpotrf_cli(tmp_path, True, factor, factor.copy(), n)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout


def test_cpotrf_finite_large_error_still_fails(tmp_path):
    n = 4
    factor = np.triu(np.ones((n, n), dtype=np.complex64) * 3)
    gm = factor.copy()
    gm[1, 2] += 1.0
    run = run_cpotrf_cli(tmp_path, True, gm, factor, n)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout


@pytest.mark.parametrize("operator", ("sgetrs", "cgetrs"))
@pytest.mark.parametrize(
    "location",
    ("input/A_gm.bin", "input/B_gm.bin", "input/W_gm.bin", "output/X_gm.bin"),
)
def test_getrs_rejects_partial_binary_elements(tmp_path, operator, location):
    dtype = np.float32 if operator == "sgetrs" else np.complex64
    arrays = identity_inputs(dtype=dtype)
    data = tmp_path / f"test/{operator}/data"
    (data / "input").mkdir(parents=True)
    (data / "output").mkdir()
    for name, target in {
        "LU": "input/A_gm.bin",
        "B": "input/B_gm.bin",
        "X": "output/X_gm.bin",
        "piv": "input/W_gm.bin",
    }.items():
        arrays[name].tofile(data / target)
    corrupted = data / location
    corrupted.write_bytes(corrupted.read_bytes() + b"\x00")
    script = SGETRS_SCRIPT if operator == "sgetrs" else CGETRS_SCRIPT
    run = subprocess.run(
        [sys.executable, str(script), "2", "1"],
        cwd=tmp_path,
        env=dict(os.environ, OPENBLAS_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 1, run.stdout + run.stderr
    assert "corrupted binary file" in run.stdout


@pytest.mark.parametrize("dtype", (np.float32, np.complex64))
def test_solve_lu_transposed_chain_matches_numpy(dtype):
    """trans≠0 转置链参考解与 numpy 转置方程一致（随机矩阵含主元交换，issue #637）。

    矩阵不做对角占优偏置（检视采纳）：严格对角占优会使部分主元法恒取对角行、
    piv 恒为单位置换，inverse_permutation 的置换 gather 退化为拷贝而零真实覆盖；
    此处断言 piv 非单位置换，保证逆序 laswp / P^T 还原解序路径被真实触发。
    """
    rng = np.random.RandomState(42)
    n = 16
    trans = 1 if dtype is np.float32 else 2
    if dtype is np.float32:
        a = rng.rand(n, n).astype(np.float32)
        b = rng.rand(n, 3).astype(dtype)
    else:
        a = (rng.rand(n, n) + 1j * rng.rand(n, n)).astype(np.complex64)
        b = (rng.rand(n, 3) + 1j * rng.rand(n, 3)).astype(dtype)
    target = np.complex128 if np.iscomplexobj(a) else np.float64
    lu, piv = scipy.linalg.lu_factor(a.astype(target))
    assert not np.array_equal(piv, np.arange(1, n + 1)), (
        "piv is identity, no real pivot swap"
    )
    x = solve_lu(lu.astype(dtype), b, piv.astype(np.int32), trans)
    lhs = a.T if trans == 1 else a.conj().T
    expected = np.linalg.solve(lhs.astype(target), b.astype(target))
    np.testing.assert_allclose(x, expected, rtol=1e-3, atol=1e-4)


def make_nontrivial_getrs_case(operator):
    """构造含真实主元交换的非平凡 getrs 用例（检视采纳），返回 (dtype, a, lu, piv, rhs)。

    [[1, 2], [3, 4]] 实数 / 对角含虚部的复数变体，scipy 部分主元法首列必选
    第 2 行（piv=[2, 2]），置换路径被真实触发；cgetrs 对角虚部使 trans=2 共轭档非平凡。
    """
    dtype = np.float32 if operator == "sgetrs" else np.complex64
    n = 2
    if operator == "sgetrs":
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    else:
        a = np.array([[1 + 1j, 2.0], [3.0, 4 - 0.5j]], dtype=np.complex128)
    lu_np, piv_np = scipy.linalg.lu_factor(a)
    lu = lu_np.astype(dtype)
    # scipy lu_factor 返回 0-based 交换伙伴索引，W_gm.bin 契约为 1-based LAPACK ipiv（与 gen_data 口径一致）
    piv = (piv_np + 1).astype(np.int32)
    assert not np.array_equal(piv, np.arange(1, n + 1)), (
        "piv is identity, no real pivot swap"
    )
    rhs = np.array([[1.0], [2.0]], dtype=dtype)
    return dtype, a, lu, piv, rhs


@pytest.mark.parametrize("operator", ("sgetrs", "cgetrs"))
def test_getrs_trans_outputs_are_verified_per_case(tmp_path, operator):
    """逐档输出文件与转置链参考解一致（含真实主元交换的非平凡 LU，issue #637）。"""
    dtype, a, lu, piv, rhs = make_nontrivial_getrs_case(operator)
    expected = {
        0: np.linalg.solve(a, rhs),
        1: np.linalg.solve(a.T, rhs),
        2: np.linalg.solve(a.conj().T, rhs),
    }
    trans_list = (0, 1) if operator == "sgetrs" else (0, 1, 2)
    arrays = {
        "LU": lu,
        "B": rhs,
        "X": expected[0].astype(dtype),
        "piv": piv,
    }
    # 仅提供 X_gm.bin 的历史 fixture 形态：trans=0 参与比对，非零档缺文件自动跳过
    run = run_cli(tmp_path, arrays, operator)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout
    assert "not found, skipped" in run.stdout
    # 逐档补齐输出文件后：全部档位参与比对
    output_dir = tmp_path / f"test/{operator}/data/output"
    for trans in trans_list:
        name = "X_gm.bin" if trans == 0 else f"X_gm_trans{trans}.bin"
        reference = expected.get(trans)
        assert reference is not None, f"missing expected reference for trans={trans}"
        reference.astype(dtype).tofile(output_dir / name)
    run = run_cli(tmp_path, arrays, operator)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout
    for trans in trans_list:
        assert f"verify trans={trans}" in run.stdout


@pytest.mark.parametrize("operator", ("sgetrs", "cgetrs"))
def test_getrs_wrong_trans_output_fails(tmp_path, operator):
    """trans=1 档输出错误必须判失败且失败可归因于该档（issue #637，检视采纳）。

    trans=0 输出提供正确解：失败轮中唯一错误档为 trans=1，[Failed] 可归因于
    非零档确实被读取比对，而非 trans=0 自身错误混入。
    """
    dtype, a, lu, piv, rhs = make_nontrivial_getrs_case(operator)
    x0 = np.linalg.solve(a, rhs).astype(dtype)
    arrays = {"LU": lu, "B": rhs, "X": x0, "piv": piv}
    # 第一轮：trans=0 正确、非零档缺文件跳过 → 通过
    run = run_cli(tmp_path, arrays, operator)
    assert run.returncode == 0, run.stdout + run.stderr
    assert "[Success]" in run.stdout
    # 第二轮：trans=1 写入错误解（cgetrs 的 trans=2 提供正确解，失败仅归因 trans=1）
    output_dir = tmp_path / f"test/{operator}/data/output"
    x_trans1 = np.linalg.solve(a.T, rhs)
    wrong = x_trans1.copy()
    wrong[0, 0] += 0.1
    wrong.astype(dtype).tofile(output_dir / "X_gm_trans1.bin")
    if operator == "cgetrs":
        x_trans2 = np.linalg.solve(a.conj().T, rhs)
        x_trans2.astype(dtype).tofile(output_dir / "X_gm_trans2.bin")
    run = run_cli(tmp_path, arrays, operator)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout
    assert "verify trans=1" in run.stdout


@pytest.mark.parametrize("operator", ("sgetrs", "cgetrs"))
def test_getrs_missing_all_trans_outputs_fails(tmp_path, operator):
    """全部档位输出缺失必须判失败：零比对不得假通过（issue #637，检视采纳）。

    算子未产出任何 X 输出（调用失败/提前返回/落盘路径不一致）时，旧口径
    all_pass 初值 True 会静默判过；新口径零命中返回 False 并明确报错。
    """
    dtype = np.float32 if operator == "sgetrs" else np.complex64
    if operator == "sgetrs":
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    else:
        a = np.array([[1 + 1j, 2.0], [3.0, 4 - 0.5j]], dtype=np.complex128)
    lu_np, piv_np = scipy.linalg.lu_factor(a)
    arrays = {
        "LU": lu_np.astype(dtype),
        "B": np.array([[1.0], [2.0]], dtype=dtype),
        # 0-based 交换伙伴转 1-based LAPACK ipiv（与 gen_data 口径一致）
        "piv": (piv_np + 1).astype(np.int32),
    }
    run = run_cli(tmp_path, arrays, operator)
    assert run.returncode == 1, run.stdout + run.stderr
    assert "[Failed]" in run.stdout
    assert "no trans output was compared" in run.stdout
