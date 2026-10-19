#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import logging
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "utils")
)
from verify_utils import read_binary_strict  # noqa: E402


INPUT_A_PATH = "./test/cheevj/data/input/A_gm.bin"
OUTPUT_W_PATH = "./test/cheevj/data/output/W_gm.bin"
OUTPUT_V_PATH = "./test/cheevj/data/output/V_gm.bin"
GOLDEN_W_PATH = "./test/cheevj/data/golden/W_golden.bin"

EIGEN_RTOL = 5e-3
EIGEN_ATOL = 5e-3
SORT_ATOL = 1e-5
RESIDUAL_TOL = 1e-2
ORTHOGONAL_TOL = 1e-2
FULL_ORTHOGONAL_LIMIT = 1024
ORTHOGONAL_PROBES = 16
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)


def normalize_jobz(value):
    jobz = value.upper() if value else "V"
    if jobz not in {"N", "V"}:
        raise ValueError(f"jobz must be N or V, got {value}")
    return jobz


def normalize_uplo(value):
    uplo = value.upper() if value else "L"
    if uplo not in {"L", "U"}:
        raise ValueError(f"uplo must be L or U, got {value}")
    return uplo


def read_complex64_column_major(path, n):
    # 尾部不完整元素直接报错（issue #202）：fromfile 会静默丢弃末尾脏字节
    data = read_binary_strict(path, np.complex64)
    expected_size = n * n
    if data.size != expected_size:
        raise ValueError(
            f"{path} contains {data.size} complex64 values, expected {expected_size}"
        )
    return data.reshape((n, n), order="F")


def read_float32(path, n):
    data = read_binary_strict(path, np.float32)
    if data.size != n:
        raise ValueError(f"{path} contains {data.size} float32 values, expected {n}")
    return data


def load_golden_eigenvalues(matrix, n):
    if os.path.exists(GOLDEN_W_PATH):
        return read_float32(GOLDEN_W_PATH, n)
    eigenvalues, _ = np.linalg.eigh(matrix)
    return eigenvalues.astype(np.float32)


def reconstruct_hermitian_from_uplo(matrix, uplo):
    full = matrix.copy()
    n = matrix.shape[0]
    diag = np.diag_indices(n)
    full[diag] = full[diag].real.astype(np.float32)
    if uplo == "U":
        rows, cols = np.triu_indices(n, 1)
        full[cols, rows] = np.conj(full[rows, cols])
    else:
        rows, cols = np.tril_indices(n, -1)
        full[cols, rows] = np.conj(full[rows, cols])
    return full


def apply_hermitian(matrix, vectors, matrix_kind):
    if matrix_kind == "diagonal":
        return np.diag(matrix).reshape(-1, 1) * vectors
    if matrix_kind == "identity":
        return vectors
    if matrix_kind in {"repeated", "clustered", "sparse_2x2"}:
        result = np.zeros_like(vectors)
        for first in range(0, matrix.shape[0], 2):
            end = min(first + 2, matrix.shape[0])
            result[first:end, :] = matrix[first:end, first:end] @ vectors[first:end, :]
        return result
    if matrix_kind == "circulant":
        spectrum = np.fft.fft(matrix[:, 0])
        return np.fft.ifft(
            spectrum.reshape(-1, 1) * np.fft.fft(vectors, axis=0), axis=0
        ).astype(np.complex64)
    return matrix @ vectors


def orthogonality_error(vectors):
    n = vectors.shape[0]
    if n <= FULL_ORTHOGONAL_LIMIT:
        identity = np.eye(n, dtype=np.complex64)
        error = vectors.conj().T @ vectors - identity
        # Check each unit-norm/orthogonal Gram entry; averaging over n*n
        # entries can hide an entirely zero or duplicated vector column.
        return float(np.max(np.abs(error)))

    rng = np.random.default_rng(2606000 + n)
    probes = (
        rng.standard_normal((n, ORTHOGONAL_PROBES), dtype=np.float32)
        + 1j * rng.standard_normal((n, ORTHOGONAL_PROBES), dtype=np.float32)
    ).astype(np.complex64)
    probes /= np.maximum(np.linalg.norm(probes, axis=0, keepdims=True), 1.0e-12)
    column_energy = np.sum(np.abs(vectors) ** 2, axis=0)
    column_error = float(np.max(np.abs(column_energy - 1.0)))
    error = vectors.conj().T @ (vectors @ probes) - probes
    probe_error = float(
        np.linalg.norm(error, ord="fro")
        / max(float(np.linalg.norm(probes, ord="fro")), 1.0)
    )
    return max(probe_error, column_error)


def print_eigenvalue_mismatches(output, golden):
    is_close = np.isclose(
        output, golden, rtol=EIGEN_RTOL, atol=EIGEN_ATOL, equal_nan=False
    )
    mismatch_indexes = np.where(~is_close)[0]
    for index, real_index in enumerate(mismatch_indexes[:32]):
        golden_data = golden[real_index]
        output_data = output[real_index]
        denom = max(abs(float(golden_data)), 1.0)
        LOGGER.info(
            "eigenvalue index: %08d, expected: %.9f, actual: %.9f, rdiff: %.6f"
            % (
                real_index,
                golden_data,
                output_data,
                abs(float(output_data - golden_data)) / denom,
            )
        )
        if index == 31:
            break


def verify_result(n, jobz, uplo, matrix_kind):
    matrix = reconstruct_hermitian_from_uplo(
        read_complex64_column_major(INPUT_A_PATH, n), uplo
    )
    output_w = read_float32(OUTPUT_W_PATH, n)
    golden_w = load_golden_eigenvalues(matrix, n)

    sorted_ok = bool(np.all(np.diff(output_w) >= -SORT_ATOL))
    abs_error = np.abs(output_w - golden_w)
    rel_error = abs_error / np.maximum(np.abs(golden_w), 1.0)
    max_abs_error = float(np.max(abs_error))
    max_rel_error = float(np.max(rel_error))
    eigen_ok = bool(
        np.allclose(
            output_w, golden_w, rtol=EIGEN_RTOL, atol=EIGEN_ATOL, equal_nan=False
        )
    )

    LOGGER.info(f"eigenvalues sorted ascending: {sorted_ok}")
    LOGGER.info(
        "max eigenvalue abs error: %.9e, tolerance: %.9e" % (max_abs_error, EIGEN_ATOL)
    )
    LOGGER.info(
        "max eigenvalue rel error: %.9e, tolerance: %.9e" % (max_rel_error, EIGEN_RTOL)
    )
    if not eigen_ok:
        print_eigenvalue_mismatches(output_w, golden_w)

    residual_ok = True
    orthogonal_ok = True
    if jobz == "V":
        output_v = read_complex64_column_major(OUTPUT_V_PATH, n)
        residual = apply_hermitian(
            matrix, output_v, matrix_kind
        ) - output_v * output_w.reshape(1, n)
        residual_norm = float(
            np.linalg.norm(residual, ord="fro")
            / max(float(np.linalg.norm(matrix, ord="fro")), 1.0)
        )
        orthogonal_norm = orthogonality_error(output_v)
        residual_ok = residual_norm <= RESIDUAL_TOL
        orthogonal_ok = orthogonal_norm <= ORTHOGONAL_TOL
        LOGGER.info(
            "residual norm: %.9e, tolerance: %.9e" % (residual_norm, RESIDUAL_TOL)
        )
        LOGGER.info(
            "orthogonality norm: %.9e, tolerance: %.9e"
            % (orthogonal_norm, ORTHOGONAL_TOL)
        )

    return sorted_ok and eigen_ok and residual_ok and orthogonal_ok


if __name__ == "__main__":
    try:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
        jobz = normalize_jobz(sys.argv[2] if len(sys.argv) > 2 else "V")
        uplo = normalize_uplo(sys.argv[3] if len(sys.argv) > 3 else "L")
        matrix_kind = (
            sys.argv[4].strip().lower() if len(sys.argv) > 4 else "random_hermitian"
        )
        if n <= 0:
            raise ValueError("n must be greater than 0")

        if not verify_result(n, jobz, uplo, matrix_kind):
            LOGGER.info("[Failed] Case accuracy verification failed.")
            sys.exit(1)

        LOGGER.info("[Success] Case accuracy verification passed.")
        sys.exit(0)
    except Exception as err:
        LOGGER.info(err)
        sys.exit(1)
