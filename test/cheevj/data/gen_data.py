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

_D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")


INPUT_A_PATH = f"{_D}input/A_gm.bin"
GOLDEN_W_PATH = f"{_D}golden/W_golden.bin"
GOLDEN_V_PATH = f"{_D}golden/V_golden.bin"
EXACT_EIGH_LIMIT = 128
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

MATRIX_KIND_ALIASES = {
    "random": "random_hermitian",
    "random_hermitian": "random_hermitian",
    "eigenspectrum": "random_hermitian",
    "circulant": "circulant",
    "dense_circulant": "circulant",
    "sparse_2x2": "sparse_2x2",
    "sparse_block": "sparse_2x2",
    "diagonal": "diagonal",
    "identity": "identity",
    "repeated": "repeated",
    "repeated_eigenvalues": "repeated",
    "clustered": "clustered",
    "near_degenerate": "clustered",
    "near-degenerate": "clustered",
    "clustered_near_degenerate": "clustered",
    "tiny": "tiny",
    "tiny_edge": "tiny",
    "cross_scale": "cross_scale",
    "wide_spectrum": "cross_scale",
}


def normalize_option(value, default, choices):
    option = value.upper() if value else default
    if option not in choices:
        raise ValueError(f"option must be one of {sorted(choices)}, got {value}")
    return option


def normalize_kind(value):
    kind = (value or "random_hermitian").strip().lower()
    if kind not in MATRIX_KIND_ALIASES:
        raise ValueError(
            f"matrix kind must be one of {sorted(MATRIX_KIND_ALIASES)}, got {value}"
        )
    return MATRIX_KIND_ALIASES[kind]


def write_complex64_column_major(path, matrix):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.asarray(matrix, dtype=np.complex64).ravel(order="F").tofile(path)


def write_float32(path, vector):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    np.asarray(vector, dtype=np.float32).tofile(path)


def keep_requested_triangle(matrix, uplo):
    input_matrix = matrix.copy()
    for column in range(matrix.shape[0]):
        if uplo == "U":
            input_matrix[slice(column + 1, None), column] = np.complex64(0.0 + 0.0j)
        else:
            input_matrix[:column, column] = np.complex64(0.0 + 0.0j)
    return input_matrix


def force_hermitian(matrix):
    matrix = ((matrix + matrix.conj().T) * 0.5).astype(np.complex64)
    diag = np.diag_indices(matrix.shape[0])
    matrix[diag] = matrix[diag].real.astype(np.float32)
    return matrix


def make_random_hermitian_matrix(n, seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, n), dtype=np.float32)
    imag = rng.standard_normal((n, n), dtype=np.float32)
    random_matrix = real + 1j * imag

    q, _ = np.linalg.qr(random_matrix.astype(np.complex64))
    eigenvalues = np.linspace(-float(n), float(n), n, dtype=np.float32)
    if n > 1:
        eigenvalues += rng.uniform(-0.05, 0.05, n).astype(np.float32)
        eigenvalues.sort()

    matrix = q @ np.diag(eigenvalues.astype(np.float32)) @ q.conj().T
    return force_hermitian(matrix), None


def make_diagonal_matrix(n, seed):
    del seed
    if n == 1:
        eigenvalues = np.array([-0.75], dtype=np.float32)
    else:
        eigenvalues = np.linspace(
            -float(n) * 0.25, float(n) * 0.25, n, dtype=np.float32
        )
    return np.diag(eigenvalues).astype(np.complex64), eigenvalues


def make_identity_matrix(n, seed):
    del seed
    eigenvalues = np.ones(n, dtype=np.float32)
    return np.eye(n, dtype=np.complex64), eigenvalues


def make_circulant_matrix(n, seed):
    rng = np.random.default_rng(seed)
    eigenvalues = np.linspace(-float(n), float(n), n, dtype=np.float32)
    if n > 1:
        eigenvalues += rng.uniform(-0.125, 0.125, n).astype(np.float32)
        eigenvalues.sort()

    first_column = np.fft.ifft(eigenvalues).astype(np.complex64)
    matrix = np.empty((n, n), dtype=np.complex64)
    rows = np.arange(n)
    for column in range(n):
        matrix[:, column] = first_column[(rows - column) % n]
    diagonal = np.diag_indices(n)
    matrix[diagonal] = matrix[diagonal].real.astype(np.float32)
    return matrix, eigenvalues


def make_2x2_hermitian_block(lambda0, lambda1, theta, phase_angle):
    c = np.float32(np.cos(theta))
    s = np.float32(np.sin(theta))
    phase = np.complex64(np.cos(phase_angle) + 1j * np.sin(phase_angle))
    unitary = np.array(
        [[c, s * phase], [-s * np.conj(phase), c]],
        dtype=np.complex64,
    )
    spectrum = np.diag(np.array([lambda0, lambda1], dtype=np.float32)).astype(
        np.complex64
    )
    return unitary @ spectrum @ unitary.conj().T


def make_block_hermitian_matrix(eigenvalues, seed):
    rng = np.random.default_rng(seed)
    n = eigenvalues.size
    matrix = np.zeros((n, n), dtype=np.complex64)
    block_order = np.arange(n)
    if n > 2:
        rng.shuffle(block_order)

    for out_index in range(0, n - 1, 2):
        eig0 = eigenvalues[block_order[out_index]]
        eig1 = eigenvalues[block_order[out_index + 1]]
        theta = rng.uniform(0.25, 0.95)
        phase_angle = rng.uniform(-np.pi, np.pi)
        block_slice = slice(out_index, out_index + 2)
        matrix[block_slice, block_slice] = make_2x2_hermitian_block(
            eig0,
            eig1,
            theta,
            phase_angle,
        )

    if n % 2 == 1:
        matrix[-1, -1] = np.complex64(eigenvalues[block_order[-1]])

    return force_hermitian(matrix), np.sort(eigenvalues.astype(np.float32))


def make_sparse_2x2_matrix(n, seed):
    rng = np.random.default_rng(seed)
    eigenvalues = np.arange(n, dtype=np.float32) - np.float32(n // 2)
    matrix = np.diag(eigenvalues).astype(np.complex64)
    if n > 1:
        lambda0 = np.float32(-(n // 2) - 2)
        lambda1 = np.float32(-(n // 2) - 1)
        theta = rng.uniform(0.25, 0.95)
        phase_angle = rng.uniform(-np.pi, np.pi)
        matrix[:2, :2] = make_2x2_hermitian_block(lambda0, lambda1, theta, phase_angle)
        eigenvalues[:2] = (lambda0, lambda1)
    return force_hermitian(matrix), np.sort(eigenvalues)


def make_repeated_eigenvalue_matrix(n, seed):
    base = np.array([-3.0, -0.5, 1.25, 2.75], dtype=np.float32)
    eigenvalues = np.resize(base, n).astype(np.float32)
    return make_block_hermitian_matrix(eigenvalues, seed)


def make_clustered_matrix(n, seed):
    indexes = np.arange(n, dtype=np.float32)
    cluster_ids = np.floor(indexes / 4.0)
    offsets = np.mod(indexes, 4.0) * np.float32(1.0e-3)
    if n == 1:
        eigenvalues = np.array([0.0], dtype=np.float32)
    else:
        cluster_center = (float(cluster_ids[0]) + float(cluster_ids[-1])) * 0.5
        eigenvalues = (
            (cluster_ids - cluster_center) * np.float32(1.5) + offsets
        ).astype(np.float32)
    return make_block_hermitian_matrix(eigenvalues, seed)


def make_tiny_matrix(n, seed):
    del seed
    if n == 1:
        eigenvalues = np.array([-1.25], dtype=np.float32)
        return np.array([[-1.25 + 0.0j]], dtype=np.complex64), eigenvalues
    if n == 2:
        matrix = np.array(
            [[2.0 + 0.0j, -0.75 + 0.5j], [-0.75 - 0.5j, -1.0 + 0.0j]],
            dtype=np.complex64,
        )
        return force_hermitian(matrix), None
    raise ValueError("tiny matrix kind is only valid for n=1 or n=2")


def make_cross_scale_matrix(n, seed):
    base = np.array([1e-8, 1e-4, 1e-2, 1.0, 1e2, 1e4, 1e8], dtype=np.float32)
    eigenvalues = np.resize(base, n).astype(np.float32)
    eigenvalues.sort()
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((n, n), dtype=np.float32)
    imag = rng.standard_normal((n, n), dtype=np.float32)
    unitary, _ = np.linalg.qr((real + 1j * imag).astype(np.complex64))
    matrix = unitary @ np.diag(eigenvalues) @ unitary.conj().T
    return force_hermitian(matrix), eigenvalues


def make_matrix(n, seed, kind):
    if kind == "random_hermitian":
        return make_random_hermitian_matrix(n, seed)
    if kind == "diagonal":
        return make_diagonal_matrix(n, seed)
    if kind == "identity":
        return make_identity_matrix(n, seed)
    if kind == "circulant":
        return make_circulant_matrix(n, seed)
    if kind == "sparse_2x2":
        return make_sparse_2x2_matrix(n, seed)
    if kind == "repeated":
        return make_repeated_eigenvalue_matrix(n, seed)
    if kind == "clustered":
        return make_clustered_matrix(n, seed)
    if kind == "tiny":
        return make_tiny_matrix(n, seed)
    if kind == "cross_scale":
        return make_cross_scale_matrix(n, seed)
    raise ValueError(f"unsupported matrix kind: {kind}")


def parse_seed_and_kind(argv):
    seed = 42
    kind = "random_hermitian"
    if len(argv) > 6:
        raise ValueError("usage: gen_data.py [n] [jobz] [uplo] [seed] [matrix_kind]")
    if len(argv) == 5:
        try:
            seed = int(argv[4])
        except ValueError:
            kind = argv[4]
    elif len(argv) == 6:
        try:
            seed = int(argv[4])
        except ValueError as exc:
            raise ValueError(f"seed must be an integer, got: {argv[4]}") from exc
        kind = argv[5]
    return seed, normalize_kind(kind)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    jobz = normalize_option(sys.argv[2] if len(sys.argv) > 2 else "V", "V", {"N", "V"})
    uplo = normalize_option(sys.argv[3] if len(sys.argv) > 3 else "L", "L", {"L", "U"})
    seed, kind = parse_seed_and_kind(sys.argv)
    if n <= 0:
        raise ValueError("n must be greater than 0")

    matrix, exact_eigenvalues = make_matrix(n, seed, kind)
    eigenvectors = None
    if kind == "cross_scale":
        # Use the spectrum of the complex64 matrix passed to the operator.
        eigenvalues = np.linalg.eigvalsh(matrix.astype(np.complex128))
    elif exact_eigenvalues is None or n <= EXACT_EIGH_LIMIT:
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    else:
        eigenvalues = np.sort(exact_eigenvalues.astype(np.float32))

    write_complex64_column_major(INPUT_A_PATH, keep_requested_triangle(matrix, uplo))
    write_float32(GOLDEN_W_PATH, eigenvalues)
    if jobz == "V" and eigenvectors is not None:
        write_complex64_column_major(GOLDEN_V_PATH, eigenvectors)
    elif os.path.exists(GOLDEN_V_PATH):
        os.remove(GOLDEN_V_PATH)

    LOGGER.info(
        f"Generated cheevj data: n={n}, jobz={jobz}, uplo={uplo}, seed={seed}, kind={kind}"
    )
    LOGGER.info(f"Input A: {INPUT_A_PATH}")
    LOGGER.info(f"Golden W: {GOLDEN_W_PATH}")
    if jobz == "V" and eigenvectors is not None:
        LOGGER.info(f"Golden V: {GOLDEN_V_PATH}")


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        LOGGER.info(err)
        sys.exit(1)
