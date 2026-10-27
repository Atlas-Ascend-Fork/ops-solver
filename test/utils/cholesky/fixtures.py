# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared data hooks for the five native repository tests."""

import itertools
import json
import logging
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import lapack

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Elementwise tolerances are diagnostics; taskbook section 3.2.2 is the final gate.
PRECISION_TOLERANCES = ((2**-10, 2**-16), (2**-13, 2**-13))
MIN_MATCH_RATE = 0.99
MAX_ABSOLUTE_ERROR = 1e-2
MAX_ERROR_ULPS = 32
PADDING_SENTINEL = complex(93, -47)
RANDOM_SEED = 950
FLOAT32_EPSILON = 2**-23
CPU_RESIDUAL_FACTOR = 5
MEAN_RESIDUAL_FACTOR = 3
INVERSE_RESIDUAL_FLOOR = 0.1
# Launch-capacity transitions and the last partial warp must both remain valid.
BUFFERED_TEST_ORDERS = (864, 865, 2**10 - 1)
BUFFERED_TEST_BATCH = 17
THROUGHPUT_TEST_BATCH = 2**10 + 1
THROUGHPUT_TEST_ORDERS = (2**6 + 2**3 + 1, 2**6 + 2**5 - 1)
DIAGONAL_STATE_TEST_ORDERS = (2**7 + 16, 2**7 + 17)
DIAGONAL_STATE_TEST_BATCH = 2**5 + 1
# Neighbors of fused/native dispatch thresholds and both sides of the workspace limit.
SOLVE_TRANSITION_ORDERS = (511, 512, 513, 1024, 1025, 2048, 2049)
SINGLE_BOUNDARY_ORDERS = (511, 512, 513, 4095, 4096)
BATCH_BOUNDARY_ORDERS = (4095, 4096)
BATCH_BOUNDARY_TEST_BATCH = 2
RHS_TEST_COUNTS = (0, 8, 9, 16, 17, 128)
BUFFERED_LARGE_TEST_ORDER = 1100
BUFFERED_LARGE_TEST_BATCH = 257
STATUS_SUCCESS = 0
STATUS_INVALID_VALUE = 3
INFO_SENTINEL = -999
EDGE_TEST_ORDER = 33
ILL_CONDITIONED_MIN_EIGENVALUE = 1e-4


@dataclass(frozen=True)
class CaseSpec:
    index: int
    n: int
    nrhs: int
    batch: int
    upper: bool
    lda: int
    ldb: int
    expected_status: int = STATUS_SUCCESS
    kind: str = "positive"
    expected_info: int = 0


def cpu_reference(op, a, b, upper, kind):
    if kind == "singular":
        # POTRI consumes an already factored triangular input, not a Hermitian matrix.
        factor = a.copy()
    else:
        factor, info = lapack.cpotrf(a, lower=not upper)
        if info != 0 and kind != "nonpositive":
            raise RuntimeError("CPU input factorization failed")
    if op == "cpotrf":
        golden, golden_info = lapack.zpotrf(a.astype(np.complex128), lower=not upper)
        if info != golden_info:
            raise RuntimeError("CPU precisions disagree on expected info")
        return a, factor, golden, info
    if op == "cpotrs":
        if not b.shape[-1]:
            return factor, b.copy(), b.astype(np.complex128), 0
        output, info = lapack.cpotrs(factor, b, lower=not upper)
        golden, golden_info = lapack.zpotrs(
            factor.astype(np.complex128), b.astype(np.complex128), lower=not upper
        )
    else:
        output, info = lapack.cpotri(factor, lower=not upper)
        golden, golden_info = lapack.zpotri(
            factor.astype(np.complex128), lower=not upper
        )
    if info != golden_info or (info != 0 and kind != "singular"):
        raise RuntimeError("CPU reference failed or expected info disagrees")
    return factor, output, golden, info


def pack(matrices, ld):
    batch, rows, columns = matrices.shape
    packed = np.full((batch, columns, ld), PADDING_SENTINEL, dtype=np.complex64)
    packed[:, :, :rows] = matrices.transpose(0, 2, 1)
    return packed


@dataclass
class MatrixData:
    inputs: list[np.ndarray]
    right_hand_sides: list[np.ndarray]
    references: list[np.ndarray]
    originals: list[np.ndarray]
    cpu_outputs: list[np.ndarray]
    infos: list[int]


def generate_matrices(base, spec, rng):
    n, nrhs, batch, upper = spec.n, spec.nrhs, spec.batch, spec.upper
    solve = base == "cpotrs"
    data = MatrixData(
        inputs=[],
        right_hand_sides=[],
        references=[],
        originals=[],
        cpu_outputs=[],
        infos=[],
    )
    for matrix in range(batch):
        z = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        if spec.kind == "ill_conditioned":
            q, _ = np.linalg.qr(z)
            eigenvalues = np.geomspace(1, ILL_CONDITIONED_MIN_EIGENVALUE, n)
            a = ((q * eigenvalues) @ q.conj().T).astype(np.complex64)
        elif spec.kind in ("nonpositive", "singular"):
            a = np.eye(n, dtype=np.complex64)
            # Batched POTRF mixes successful matrices and failures at different pivots.
            if batch == 1 or matrix % 3 == 0:
                pivot = 1 + matrix % (n - 1)
                a[pivot, pivot] = -1 if spec.kind == "nonpositive" else 0
        else:
            a = (z.conj().T @ z + max(1, n) * np.eye(n)).astype(np.complex64)
        b = (rng.normal(size=(n, nrhs)) + 1j * rng.normal(size=(n, nrhs))).astype(
            np.complex64
        )
        data.originals.append(a @ a.conj().T if spec.kind == "singular" else a.copy())
        if n:
            prepared, cpu, golden, info = cpu_reference(
                base, a, b if solve else None, upper, spec.kind
            )
        else:
            prepared, golden = a, np.empty((0, nrhs if solve else 0), np.complex128)
            cpu, info = golden.astype(np.complex64), 0
        unused = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
        prepared[unused] = complex(np.nan, np.nan)
        data.inputs.append(prepared)
        data.right_hand_sides.append(b)
        data.references.append(golden)
        data.cpu_outputs.append(cpu)
        data.infos.append(info)
    return data


def base_case_shapes(op):
    batched = op.endswith("Batched")
    batch = 7 if batched else 1
    specs = [(0, False, 0, batch)]
    orders = (1, 32, 33, 64, 65, 129, 257)
    for n, upper, padding in itertools.product(orders, (False, True), (0, 3)):
        case_batch = batch
        if op == "cpotrfBatched" and n == 64:
            case_batch = THROUGHPUT_TEST_BATCH
        specs.append((n, upper, padding, case_batch))
    if op == "cpotrfBatched":
        # Adjacent short tails exercise padded panel slots and a partial matrix group.
        specs.extend(
            (n, upper, 3, THROUGHPUT_TEST_BATCH)
            for n, upper in itertools.product(THROUGHPUT_TEST_ORDERS, (False, True))
        )
        # Both sides of the packed final-panel boundary must preserve the compensated diagonal state.
        specs.extend(
            (n, upper, 3, DIAGONAL_STATE_TEST_BATCH)
            for n, upper in itertools.product(DIAGONAL_STATE_TEST_ORDERS, (False, True))
        )
    if op == "cpotrsBatched":
        # Exercise both sides of the UB launch limit and a partial final warp.
        for solve_orders in (BUFFERED_TEST_ORDERS, SOLVE_TRANSITION_ORDERS):
            specs.extend(
                (n, upper, 3, BUFFERED_TEST_BATCH)
                for n, upper in itertools.product(solve_orders, (False, True))
            )
    cases = [(*spec, "positive") for spec in specs]
    cases.extend(
        (EDGE_TEST_ORDER, upper, 3, batch, "ill_conditioned") for upper in (False, True)
    )
    failure = "nonpositive" if op.removesuffix("Batched") == "cpotrf" else "singular"
    if op.removesuffix("Batched") != "cpotrs":
        cases.extend(
            (EDGE_TEST_ORDER, upper, 3, batch, failure) for upper in (False, True)
        )
    return cases


def boundary_case_shapes(op):
    batched = op.endswith("Batched")
    cases = []
    if batched:
        cases.extend(
            (n, upper, 3, BATCH_BOUNDARY_TEST_BATCH, "positive")
            for n, upper in zip(BATCH_BOUNDARY_ORDERS, (True, False))
        )
    else:
        cases.extend(
            (n, upper, 3, 1, "positive")
            for n, upper in itertools.product(SINGLE_BOUNDARY_ORDERS, (False, True))
        )
    if op == "cpotrsBatched":
        cases.extend(
            (BUFFERED_LARGE_TEST_ORDER, upper, 3, BUFFERED_LARGE_TEST_BATCH, "positive")
            for upper in (False, True)
        )
    return cases


def make_spec(op, index, shape):
    n, upper, padding, batch, kind = shape
    solve = op.removesuffix("Batched") == "cpotrs"
    nrhs = (1 if op.endswith("Batched") else 1 + 2 * (index % 2)) if solve else 0
    return CaseSpec(
        index,
        n,
        nrhs,
        batch,
        upper,
        max(1, n + padding),
        max(1, n + 2 * padding),
        kind=kind,
    )


def invalid_specs(op, start_index):
    batch = 7 if op.endswith("Batched") else 1
    solve = op.removesuffix("Batched") == "cpotrs"
    result = []
    for upper in (False, True):
        shape = (-1, upper, 0, batch, "positive")
        result.append(
            replace(
                make_spec(op, start_index + len(result), shape),
                expected_status=STATUS_INVALID_VALUE,
                expected_info=-2,
            )
        )
        shape = (EDGE_TEST_ORDER, upper, 0, batch, "positive")
        result.append(
            replace(
                make_spec(op, start_index + len(result), shape),
                lda=EDGE_TEST_ORDER - 1,
                expected_status=STATUS_INVALID_VALUE,
                expected_info=-5 if solve else -4,
            )
        )
        if op == "cpotrsBatched":
            result.append(
                replace(
                    make_spec(op, start_index + len(result), shape),
                    nrhs=2,
                    expected_status=STATUS_INVALID_VALUE,
                    expected_info=-3,
                )
            )
    return result


def case_specs(op):
    # Append new coverage so existing case indices and RNG seeds remain stable.
    shapes = base_case_shapes(op) + boundary_case_shapes(op)
    result = [make_spec(op, index, shape) for index, shape in enumerate(shapes)]
    if op == "cpotrs":
        for nrhs, upper in itertools.product(RHS_TEST_COUNTS, (False, True)):
            shape = (EDGE_TEST_ORDER, upper, 3, 1, "positive")
            result.append(replace(make_spec(op, len(result), shape), nrhs=nrhs))
    return result + invalid_specs(op, len(result))


def generate_invalid(root, op, spec):
    """Keep valid backing storage while passing the deliberately invalid API dimensions."""
    n, batch = max(1, spec.n), max(1, spec.batch)
    solve = op.removesuffix("Batched") == "cpotrs"
    nrhs = max(1, spec.nrhs) if solve else 0
    a = np.broadcast_to(np.eye(n, dtype=np.complex64), (batch, n, n))
    b = np.ones((batch, n, nrhs), dtype=np.complex64)
    pack(a, max(n, spec.lda)).tofile(root / "input" / f"{spec.index}-A.bin")
    pack(b, max(n, spec.ldb)).tofile(root / "input" / f"{spec.index}-B.bin")
    infos = np.full(batch if op == "cpotrfBatched" else 1, INFO_SENTINEL, np.int32)
    infos[0] = spec.expected_info
    infos.tofile(root / "input" / f"{spec.index}-info.bin")


def generate(root, op=None):
    op = op or root.parent.name.replace("_batched", "Batched")
    base = op.removesuffix("Batched")
    for directory in ("input", "output", "golden"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    manifest = []
    # Adjacent orders cover register/panel boundaries; padding and seed are fixture data.
    for spec in case_specs(op):
        index, n, nrhs, batch, upper, lda, ldb = (
            spec.index,
            spec.n,
            spec.nrhs,
            spec.batch,
            spec.upper,
            spec.lda,
            spec.ldb,
        )
        manifest.append(
            [index, n, nrhs, batch, int(upper), lda, ldb, spec.expected_status]
        )
        if spec.expected_status != STATUS_SUCCESS:
            generate_invalid(root, op, spec)
            continue
        rng = np.random.default_rng(RANDOM_SEED + index)
        data = generate_matrices(base, spec, rng)
        a_packed = pack(np.stack(data.inputs), lda)
        b_packed = pack(np.stack(data.right_hand_sides), ldb)
        a_packed.tofile(root / "input" / f"{index}-A.bin")
        b_packed.tofile(root / "input" / f"{index}-B.bin")
        expected_info = np.array(
            data.infos if op == "cpotrfBatched" else data.infos[:1], dtype=np.int32
        )
        expected_info.tofile(root / "input" / f"{index}-info.bin")
        np.savez(
            root / "golden" / f"{index}.npz",
            golden=np.stack(data.references),
            original_a=np.stack(data.originals),
            cpu_fp32=np.stack(data.cpu_outputs),
        )
    (root / "input/manifest.tsv").write_text(
        "".join("\t".join(map(str, row)) + "\n" for row in manifest)
    )
    logging.info(json.dumps({"op": op, "generated_cases": len(manifest)}))


def same_bits(a, b):
    return np.array_equal(a.view(np.uint32), b.view(np.uint32))


def norm1(matrix):
    return np.max(np.sum(np.abs(matrix), axis=-2), axis=-1)


def lapack_ratio(op, a, output, b, upper):
    """Compute each matrix's residual in complex128 on the actual FP32 inputs."""
    a, output = a.astype(np.complex128), output.astype(np.complex128)
    n = a.shape[-1]
    if not n or (op == "cpotrs" and not output.shape[-1]):
        return np.zeros(a.shape[:-2], dtype=np.float64)
    if op == "cpotrs":
        numerator = np.sum(np.abs(b.astype(np.complex128) - a @ output), axis=-2)
        denominator = (
            norm1(a)[:, None] * np.sum(np.abs(output), axis=-2) * FLOAT32_EPSILON
        )
    else:
        triangle = np.triu(output) if upper else np.tril(output)
        adjoint = triangle.swapaxes(-1, -2).conj()
        if op == "cpotrf":
            reconstructed = adjoint @ triangle if upper else triangle @ adjoint
            numerator = norm1(reconstructed - a)
            denominator = n * norm1(a) * FLOAT32_EPSILON
        else:
            inverse = triangle + adjoint
            diagonal = np.arange(n)
            inverse[:, diagonal, diagonal] = triangle[:, diagonal, diagonal].real
            numerator = norm1(np.eye(n) - a @ inverse)
            denominator = n * norm1(a) * norm1(inverse) * FLOAT32_EPSILON
    ratio = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.inf),
        where=denominator != 0,
    )
    ratio[(denominator == 0) & (numerator == 0)] = 0
    return np.max(ratio, axis=-1) if op == "cpotrs" else ratio


def elementwise_diagnostics(actual, golden, index):
    """Report elementwise diagnostics without overriding the LAPACK decision."""
    a = np.stack((actual.real, actual.imag)).astype(np.float64)
    g = np.stack((golden.real, golden.imag))
    error = np.abs(a - g)
    maximum_index = np.unravel_index(np.argmax(error), error.shape)
    g32 = np.float32(g[maximum_index])
    ulp = abs(float(np.nextafter(g32, np.float32(np.inf))) - float(g32))
    maximum_limit = max(MAX_ABSOLUTE_ERROR, MAX_ERROR_ULPS * ulp)
    results = []
    for rtol, atol in PRECISION_TOLERANCES:
        matched = np.mean(
            error <= atol + rtol * np.abs(g),
            axis=tuple(i for i in range(error.ndim) if i not in (0, 1)),
        )
        results.append(
            bool(np.all(matched >= MIN_MATCH_RATE) and np.max(error) <= maximum_limit)
        )
    return {
        "case": index,
        "matched": results,
        "max_abs": float(np.max(error)),
        "diagnostic_only": True,
    }


def verify_residuals(op, records):
    batched = op.endswith("Batched")
    base = op.removesuffix("Batched")
    mean = np.mean([record["cpu_ratio"][0] for record in records]) if records else 0
    for record in records:
        cpu = record["cpu_ratio"]
        floor = (
            INVERSE_RESIDUAL_FLOOR
            if base == "cpotri"
            else MEAN_RESIDUAL_FACTOR * (np.mean(cpu) if batched else mean)
        )
        threshold = np.maximum(CPU_RESIDUAL_FACTOR * cpu, floor)
        passed = bool(
            np.all(np.isfinite(record["ratio"]))
            and np.all(record["ratio"] <= threshold)
        )
        logging.info(
            json.dumps(
                {
                    "case": record["case"],
                    "lapack_ratio": record["ratio"].tolist(),
                    "cpu_ratio": cpu.tolist(),
                    "threshold": threshold.tolist(),
                    "pass": passed,
                }
            )
        )
        if not passed:
            raise RuntimeError(f"LAPACK residual: {record['case']}")


def read_case(root, op, spec):
    index, n, nrhs, batch, lda, ldb = (
        spec.index,
        spec.n,
        spec.nrhs,
        spec.batch,
        spec.lda,
        spec.ldb,
    )
    solve = op.removesuffix("Batched") == "cpotrs"
    a_input = np.fromfile(root / "input" / f"{index}-A.bin", np.complex64).reshape(
        batch, n, lda
    )
    a_output = np.fromfile(root / "output" / f"{index}-A.bin", np.complex64).reshape(
        batch, n, lda
    )
    b_input = np.fromfile(root / "input" / f"{index}-B.bin", np.complex64).reshape(
        batch, nrhs, ldb
    )
    b_output = np.fromfile(root / "output" / f"{index}-B.bin", np.complex64).reshape(
        batch, nrhs, ldb
    )
    info = np.fromfile(root / "output" / f"{index}-info.bin", np.int32)
    expected_info = np.fromfile(root / "input" / f"{index}-info.bin", np.int32)
    if not (
        info.size == (batch if op == "cpotrfBatched" else 1)
        and np.array_equal(info, expected_info)
    ):
        raise RuntimeError(f"info: {index}")
    if not (same_bits(a_input[:, :, n:], a_output[:, :, n:])):
        raise RuntimeError(f"A padding: {index}")
    if not (same_bits(b_input[:, :, n:], b_output[:, :, n:])):
        raise RuntimeError(f"B padding: {index}")
    if solve and not same_bits(a_input, a_output):
        raise RuntimeError(f"read-only A: {index}")
    matrices = a_output[:, :, :n].transpose(0, 2, 1)
    rhs = b_input[:, :, :n].transpose(0, 2, 1)
    output = b_output[:, :, :n].transpose(0, 2, 1) if solve else matrices
    return output, rhs, info


def verify_invalid_case(root, spec):
    index = spec.index
    for operand in ("A", "B"):
        filename = f"{index}-{operand}.bin"
        if (root / "input" / filename).read_bytes() != (
            root / "output" / filename
        ).read_bytes():
            raise RuntimeError(f"invalid-argument operand changed: {index}/{operand}")
    info = np.fromfile(root / "output" / f"{index}-info.bin", np.int32)
    expected = np.fromfile(root / "input" / f"{index}-info.bin", np.int32)
    if not np.array_equal(info, expected):
        raise RuntimeError(f"invalid-argument info: {index}")
    logging.info(
        json.dumps(
            {
                "case": index,
                "expected_status": spec.expected_status,
                "negative_info": int(info[0]),
                "operands_unchanged": True,
                "pass": True,
            }
        )
    )


def verify_case(root, op, row):
    spec = CaseSpec(*map(int, row.split()))
    index, n, upper = spec.index, spec.n, spec.upper
    base = op.removesuffix("Batched")
    solve = base == "cpotrs"
    if spec.expected_status != STATUS_SUCCESS:
        verify_invalid_case(root, spec)
        return None
    output, rhs, info = read_case(root, op, spec)
    valid = info == 0 if op == "cpotrfBatched" else np.full(spec.batch, info[0] == 0)
    output, rhs = output[valid], rhs[valid]
    record = None
    with np.load(root / "golden" / f"{index}.npz") as data:
        golden = data["golden"][valid]
        if output.size:
            original_a, cpu = data["original_a"][valid], data["cpu_fp32"][valid]
            record = {
                "case": index,
                "ratio": lapack_ratio(base, original_a, output, rhs, upper),
                "cpu_ratio": lapack_ratio(base, original_a, cpu, rhs, upper),
            }
    if not solve:
        target = np.triu_indices(n) if upper else np.tril_indices(n)
        output = output[:, target[0], target[1]]
        golden = golden[:, target[0], target[1]]
    diagnostics = (
        elementwise_diagnostics(output, golden, index) if output.size else None
    )
    logging.info(
        json.dumps(
            {
                "case": index,
                "info": "PASS",
                "expected_positive_info": int(np.count_nonzero(info > 0)),
                "elementwise": diagnostics,
                "padding": "PASS",
            }
        )
    )
    return record


def verify(root, op=None):
    op = op or root.parent.name.replace("_batched", "Batched")
    rows = (root / "input/manifest.tsv").read_text().splitlines()
    if not rows:
        raise RuntimeError("no cases verified")
    records = [verify_case(root, op, row) for row in rows]
    verify_residuals(op, [record for record in records if record is not None])
    logging.info(json.dumps({"op": op, "verified_cases": len(rows), "pass": True}))
