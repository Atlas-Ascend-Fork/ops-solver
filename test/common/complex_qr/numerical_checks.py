# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Numerical reference helpers for cross-operator QR integration regressions."""

import ctypes as ct
from collections import namedtuple

import numpy as np
from scipy.linalg.lapack import cgels, cgeqrf, cungqr, cunmqr, zungqr
from runtime import check
from test_support import require

ApplyReference = namedtuple("ApplyReference", "da dt qr tau q ref ref_tau ref_q")
EPS = 2.0**-24


def norm1(a):
    return (
        float(np.max(np.sum(np.abs(a.astype(np.complex128)), axis=0)))
        if a.size
        else 0.0
    )


def ratio(numerator, denominator):
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("inf")
    return (
        numerator / denominator
        if denominator
        else (0.0 if numerator == 0 else float("inf"))
    )


def matrix(rng, m, n):
    parts = []
    for _ in range(2):
        uniform = rng.uniform(-5, 5, (m, n))
        normal = rng.normal(rng.uniform(-5, 5), rng.uniform(0.1, 2), (m, n))
        parts.append(np.where(rng.random((m, n)) < 0.5, uniform, normal))
    return np.asfortranarray((parts[0] + 1j * parts[1]).astype(np.complex64))


def pad(a, ld):
    out = np.full((ld, a.shape[1]), complex(97, -83), dtype=np.complex64, order="F")
    out[: a.shape[0]] = a
    return out


def cpu_factor(a):
    qr, tau, _, info = cgeqrf(a)
    require(info == 0, "cpu_factor info == 0")
    return qr, tau


def cpu_q(qr, tau, n, high=False):
    m = qr.shape[0]
    a = np.zeros((m, n), dtype=np.complex128 if high else np.complex64, order="F")
    a[:, : len(tau)] = qr[:, : len(tau)]
    fn = zungqr if high else cungqr
    q, _, info = fn(a, tau.astype(a.dtype))
    require(info == 0, "cpu_q info == 0")
    return q


def device_factor(rt, original, ld):
    a = pad(original, ld)
    da = rt.upload(a)
    dt = rt.alloc(min(original.shape) * 8)
    length = ct.c_int()
    m, n = original.shape
    check(rt.lib.aclsolverCgeqrf_bufferSize(rt.handle, m, n, da, ld, ct.byref(length)))
    info = rt.alloc(4)
    check(
        rt.lib.aclsolverCgeqrf(
            rt.handle, m, n, da, ld, dt, rt.alloc(length.value * 8), length.value, info
        )
    )
    require(
        rt.download(info, (1,), np.int32)[0] == 0,
        "device_factor devInfo == 0",
    )
    got = rt.download(da, a.shape)
    require(np.array_equal(got[m:], a[m:]), "A padding modified")
    return da, dt, got[:m], rt.download(dt, (min(m, n),))


def device_q(rt, qr, tau, n, ld):
    m = qr.shape[0]
    a = np.zeros((m, n), np.complex64, order="F")
    a[:, : len(tau)] = qr[:, : len(tau)]
    padded = pad(a, ld)
    da, dt = rt.upload(padded), rt.upload(tau)
    rt.run("aclsolverCungqr", [m, n, len(tau), da, ld, dt])
    got = rt.download(da, padded.shape)
    require(
        rt.download(dt, tau.shape).tobytes() == tau.tobytes(), "tau modified by ungqr"
    )
    require(np.array_equal(got[m:], padded[m:]), "Q padding modified")
    return got[:m]


def orthogonality(q):
    q = q.astype(np.complex128)
    return ratio(norm1(np.eye(q.shape[1]) - q.conj().T @ q), q.shape[0] * EPS)


def qr_ratio(a, qr, tau):
    q = cpu_q(qr, tau, a.shape[0], True)
    r = np.triu(qr.astype(np.complex128))
    return ratio(
        norm1(r - q.conj().T @ a.astype(np.complex128)), a.shape[0] * norm1(a) * EPS
    )


def gels_ratio(a, b, x):
    a, b, x = [v.astype(np.complex128) for v in (a, b, x)]
    m, n = a.shape
    residual = b - a @ x
    if m == n:
        return max(
            ratio(
                norm1(residual[:, slice(j, j + 1)]),
                m * norm1(a) * norm1(x[:, slice(j, j + 1)]) * EPS,
            )
            for j in range(b.shape[1])
        )
    return ratio(
        norm1(residual.conj().T @ a), norm1(a) * norm1(b) * max(m, n, b.shape[1]) * EPS
    )


def evaluate_factor(rt, c, rng):
    m, n = c["m"], c["n"]
    a = matrix(rng, m, n)
    _, _, qr, tau = device_factor(rt, a, c["lda"])
    ref, ref_tau = cpu_factor(a)
    return qr_ratio(a, qr, tau), qr_ratio(a, ref, ref_tau), 5, 0.1


def evaluate_generate(rt, c, rng):
    m, n = c["m"], c["n"]
    k = c["k"]
    a = matrix(rng, m, max(k, 1))
    _, _, qr, tau = device_factor(rt, a, c["lda"])
    qr, tau = qr[:, :k], tau[:k]
    q = device_q(rt, qr, tau, n, c["lda"])
    ref_q = cpu_q(qr, tau, n)
    # Same reflectors: prevent an unrelated unitary output from passing.
    np.testing.assert_allclose(q, ref_q, rtol=2e-4, atol=3e-5)
    ref, ref_tau = cpu_factor(a)
    return (
        orthogonality(q),
        orthogonality(cpu_q(ref[:, :k], ref_tau[:k], n)),
        5,
        1.5,
    )


def apply_reference(rt, c, rng):
    m, n = c["m"], c["n"]
    side, k = c["side"], c["k"]
    order = m if side == 0 else n
    a = matrix(rng, order, max(k, 1))
    da, dt, qr, tau = device_factor(rt, a, c["lda"])
    qr, tau = qr[:, :k], tau[:k]
    q = device_q(rt, qr, tau, order, c["lda"])
    ref, ref_tau = cpu_factor(a)
    ref, ref_tau = ref[:, :k], ref_tau[:k]
    ref_q = cpu_q(ref, ref_tau, order)
    require(
        orthogonality(q) <= max(5 * orthogonality(ref_q), 1.5),
        "ungqr prerequisite failed",
    )
    return ApplyReference(da, dt, qr, tau, q, ref, ref_tau, ref_q)


def evaluate_apply(rt, c, rng):
    m, n = c["m"], c["n"]
    da, dt, qr, tau, q, ref, ref_tau, ref_q = apply_reference(rt, c, rng)
    side, trans, k = c["side"], c["trans"], c["k"]
    order = m if side == 0 else n
    original = matrix(rng, m, n)
    padded = pad(original, c["ldc"])
    dc = rt.upload(padded)
    a_before = rt.download(da, (c["lda"], max(k, 1)))
    tau_before = rt.download(dt, (min(order, max(k, 1)),))
    rt.run("aclsolverCunmqr", [side, trans, m, n, k, da, c["lda"], dt, dc, c["ldc"]])
    got = rt.download(dc, padded.shape)
    require(
        rt.download(da, a_before.shape).tobytes() == a_before.tobytes(),
        "A modified by unmqr",
    )
    require(
        rt.download(dt, tau_before.shape).tobytes() == tau_before.tobytes(),
        "tau modified by unmqr",
    )
    require(np.array_equal(got[m:], padded[m:]), "C padding modified")
    q = q.astype(np.complex128)
    ref_q = ref_q.astype(np.complex128)
    if trans == 2:
        q = q.conj().T
        ref_q = ref_q.conj().T
    expected = q @ original if side == 0 else original @ q
    ref_expected = ref_q @ original if side == 0 else original @ ref_q
    args = (
        b"L" if side == 0 else b"R",
        b"N" if trans == 0 else b"C",
        ref,
        ref_tau,
        original,
    )
    _, work, status = cunmqr(*args, -1)
    require(status == 0, "cunmqr workspace query status == 0")
    ref_out, _, status = cunmqr(*args, int(work[0].real))
    require(status == 0, "cunmqr reference status == 0")
    denom = order * norm1(original) * EPS
    return (
        ratio(norm1(expected - got[:m]), denom),
        ratio(norm1(ref_expected - ref_out), denom),
        5,
        0.1,
    )


def evaluate_solve(rt, c, rng):
    m, n = c["m"], c["n"]
    nrhs = c["nrhs"]
    a, b = matrix(rng, m, n), matrix(rng, m, nrhs)
    pa, pb, px = (
        pad(a, c["ldda"]),
        pad(b, c["lddb"]),
        pad(np.zeros((n, nrhs), np.complex64), c["lddx"]),
    )
    da, db, dx = rt.upload(pa), rt.upload(pb), rt.upload(px)
    rt.run(
        "aclsolverCCgels",
        [m, n, nrhs, da, c["ldda"], db, c["lddb"], dx, c["lddx"]],
        True,
    )
    x = rt.download(dx, px.shape)
    require(np.array_equal(x[n:], px[n:]), "X padding modified")
    require(np.array_equal(rt.download(db, pb.shape), pb), "B modified")
    require(np.array_equal(rt.download(da, pa.shape)[m:], pa[m:]), "A padding modified")
    _, ref, info = cgels(a, b)
    require(info == 0, "cgels reference info == 0")
    return (
        gels_ratio(a, b, x[:n]),
        gels_ratio(a, b, ref[:n]),
        5 if m == n else 10,
        0.015 if m == n else 0,
    )


def evaluate(rt, op, c, rng):
    operations = {
        "cgeqrf": evaluate_factor,
        "cungqr": evaluate_generate,
        "cunmqr": evaluate_apply,
        "cgels": evaluate_solve,
    }
    operation = operations.get(op)
    if operation is None:
        raise ValueError(f"Unsupported operator: {op}")
    return operation(rt, c, rng)
