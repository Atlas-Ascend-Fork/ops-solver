# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Supplementary device smoke tests; NOT a replacement for official acceptance cases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import argparse
import ctypes as ct
import json

import numpy as np
from scipy.linalg.lapack import cungqr
from test_support import report, require


from runtime import Runtime, check


def random_matrix(rng, m, n):
    return np.asfortranarray(
        (rng.normal(size=(m, n)) + 1j * rng.normal(size=(m, n))).astype(np.complex64)
    )


def smoke_factor(rt, rng, m, n, results):
    report(json.dumps(dict(starting="cgeqrf", m=m, n=n)))
    original = random_matrix(rng, m, n)
    padded = np.full((m + 3, n), 17 + 19j, dtype=np.complex64, order="F")
    padded[:m] = original
    a = rt.upload(padded)
    tau = rt.alloc(min(m, n) * 8)
    length = ct.c_int()
    check(
        rt.lib.aclsolverCgeqrf_bufferSize(rt.handle, m, n, a, m + 3, ct.byref(length))
    )
    info = rt.alloc(4)
    check(
        rt.lib.aclsolverCgeqrf(
            rt.handle,
            m,
            n,
            a,
            m + 3,
            tau,
            rt.alloc(length.value * 8),
            length.value,
            info,
        )
    )
    require(
        rt.download(info, (1,), np.int32)[0] == 0,
        "smoke.py:158: rt.download(info, (1,), np.int32)[0] == 0",
    )
    fact = rt.download(a, padded.shape)
    factors = rt.download(tau, (min(m, n),))
    require(
        np.array_equal(fact[m:], padded[m:]),
        "smoke.py:161: np.array_equal(fact[m:], padded[m:])",
    )
    k = min(m, n)
    q, _, status = cungqr(np.asfortranarray(fact[:m, :k]), factors)
    require(status == 0, "smoke.py:164: status == 0")
    relative = np.linalg.norm(
        q.astype(np.complex128) @ np.triu(fact[:k]) - original
    ) / np.linalg.norm(original)
    require(relative < 3e-6, relative)
    results.append(dict(op="cgeqrf", m=m, n=n, relative=float(relative)))
    return fact, k, a, tau, q


def smoke_apply(rt, rng, shape, factors, results):
    m, n = shape
    fact, k, a, tau, q = factors
    q_input = np.zeros((m + 3, m), np.complex64, order="F")
    q_input[:, :k] = fact[:, :k]
    q_device = rt.upload(q_input)
    report(json.dumps(dict(starting="cungqr", m=m, n=m, k=k)))
    rt.run("aclsolverCungqr", [m, m, k, q_device, m + 3, tau])
    q_full = rt.download(q_device, q_input.shape)[:m]
    np.testing.assert_allclose(q_full[:, :k], q, rtol=3e-5, atol=3e-6)
    results.append(dict(op="cungqr", m=m, n=m, k=k))
    for side in (0, 1):
        for trans in (0, 2):
            c = random_matrix(rng, m, 5) if side == 0 else random_matrix(rng, 5, m)
            dev_c = rt.upload(c)
            rt.run(
                "aclsolverCunmqr",
                [side, trans, *c.shape, k, a, m + 3, tau, dev_c, c.shape[0]],
            )
            got = rt.download(dev_c, c.shape)
            opq = q_full if trans == 0 else q_full.conj().T
            expected = opq @ c if side == 0 else c @ opq
            np.testing.assert_allclose(got, expected, rtol=3e-5, atol=8e-6)
            results.append(
                dict(op="cunmqr", side=side, trans=trans, m=c.shape[0], n=c.shape[1])
            )


def run_smoke(rt):
    rng = np.random.default_rng(1729)
    results = []
    for m, n in [(1, 1), (7, 3), (3, 7), (32, 32), (65, 33)]:
        fact, k, a, tau, q = smoke_factor(rt, rng, m, n, results)
        smoke_apply(rt, rng, (m, n), (fact, k, a, tau, q), results)
        rt.free_all()
    for m, n in [(7, 3), (16, 16), (65, 33)]:
        a = random_matrix(rng, m, n)
        b = random_matrix(rng, m, 3)
        da, db = rt.upload(a), rt.upload(b)
        dx = rt.alloc(n * 3 * 8)
        rt.run("aclsolverCCgels", [m, n, 3, da, m, db, m, dx, n], True)
        got = rt.download(dx, (n, 3))
        expected = np.linalg.lstsq(
            a.astype(np.complex128), b.astype(np.complex128), rcond=None
        )[0]
        np.testing.assert_allclose(got, expected, rtol=5e-5, atol=5e-6)
        require(
            np.array_equal(rt.download(db, b.shape), b),
            "smoke.py:208: np.array_equal(rt.download(db, b.shape), b)",
        )
        results.append(dict(op="ccgels", m=m, n=n, nrhs=3))
        rt.free_all()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    runtime = Runtime(args.library, args.device)
    try:
        report(json.dumps({"supplementary_smoke": run_smoke(runtime)}, indent=2))
    finally:
        runtime.close()
