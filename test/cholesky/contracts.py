# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================
"""Boundary, pointer-array, partial-factor, padding and determinism checks."""

import argparse
import ctypes as C
import json
import logging
import sys
from functools import partial
from itertools import product
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from runner import Solver, Int32, require
from lapack_residual import paired


def record_result(rows, name, body):
    try:
        body()
        row = {"test": name, "pass": True}
    except Exception as exc:
        row = {"test": name, "pass": False, "error": str(exc)}
    rows.append(row)
    logging.info(json.dumps(row))


def expect(solver, fn, arguments, negative=None, success=False):
    info = torch.full((4,), -777, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    call = [solver.handle] + list(arguments)
    call[-1 if "Batched" not in fn else -2] = info.data_ptr()
    status = getattr(solver.lib, fn)(*call)
    require((status == 0) == success, f"status {status}")
    solver.sync()
    actual = info.cpu().numpy()
    expected = 0 if success else negative
    require(actual[0] == expected, f"info {actual.tolist()} expected {expected}")


def status_only(solver, fn, arguments, status):
    actual = getattr(solver.lib, fn)(*arguments)
    require(actual == status, f"status {actual} expected {status}")
    solver.sync()


def singular_inverse(solver, n, k, upper, diagonal=0.0):
    factor = np.eye(n, dtype=np.float32)
    factor[k - 1, k - 1] = diagonal
    ignored = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
    factor[ignored] = np.nan
    got, info = solver.run("spotri", factor, upper=upper, padding=3)
    require(info[0] == k, f"info {info} expected {k}")
    require(
        np.array_equal(got, factor, equal_nan=True), "singular inverse overwrote input"
    )


def partial_factor(solver, n, k, upper, original):
    completed = k - 1
    x = original.copy()
    x[completed, completed] -= 10000
    # Poison the ignored triangle; no arithmetic may depend on it.
    ignored = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
    x[ignored] = np.nan
    got, info = solver.run("spotrf", x, upper=upper, padding=3)
    require(int(info[0]) == k, f"info {info} expected {k}")
    if k > 1:
        first = la.cholesky(
            original[:completed, :completed].astype(np.float64), lower=True
        )
        lower = np.zeros((n, completed))
        lower[:completed] = first
        lower[completed:] = la.solve_triangular(
            first, original[completed:, :completed].astype(np.float64).T, lower=True
        ).T
        logical = got.T if upper else got
        idx = np.tril_indices(n, 0)
        selected = idx[1] < completed
        ii, jj = idx[0][selected], idx[1][selected]
        require(
            np.allclose(
                logical[ii, jj], lower[ii, jj], rtol=2 ** (-10), atol=2 ** (-16)
            ),
            "contract condition failed",
        )


def deterministic(solver, n, upper, original):
    previous = None
    for _ in range(3):
        out, info = solver.run("spotrf", original, upper=upper, padding=3)
        idx = np.triu_indices(n) if upper else np.tril_indices(n)
        state = out[idx].tobytes() + info.tobytes()
        require(previous is None or state == previous, "contract condition failed")
        previous = state


def pointers(solver, rng, n, upper):
    ld, count = n + 3, 7
    inputs, buffers = [], []
    for q in range(count):
        r = rng.normal(size=(n, n)) / np.sqrt(n)
        x = (r @ r.T + np.eye(n) * (1 + q)).astype(np.float32)
        raw = np.full((n, ld), -12345.125, dtype=np.float32)
        raw[:, :n] = x.T
        inputs.append(x)
        buffers.append(torch.from_numpy(raw.reshape(-1)).to("npu"))
    order = [6, 2, 4, 0, 5, 1, 3]
    array = torch.tensor(
        [buffers[q].data_ptr() for q in order], dtype=torch.int64, device="npu"
    )
    infos = torch.full((count,), -777, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    solver.check(
        solver.lib.aclsolverSpotrfBatched(
            solver.handle, int(upper), n, array.data_ptr(), ld, infos.data_ptr(), count
        )
    )
    solver.sync()
    require(np.all(infos.cpu().numpy() == 0), "contract condition failed")
    for q in order:
        raw = buffers[q].cpu().numpy().reshape(n, ld)
        require(np.all(raw[:, n:] == -12345.125), "contract condition failed")
        gold = la.cholesky(inputs[q].astype(np.float64), lower=not upper)
        idx = np.triu_indices(n) if upper else np.tril_indices(n)
        require(
            np.allclose(raw[:, :n].T[idx], gold[idx], rtol=2 ** (-10), atol=2 ** (-16)),
            "contract condition failed",
        )


def batch_partial(solver, rng, n, k, upper):
    completed = k - 1
    r = rng.normal(size=(n, n)) / np.sqrt(n)
    original = (r @ r.T + np.eye(n) * 2).astype(np.float32)
    matrices = np.stack([original.copy() for _ in range(3)])
    matrices[1, completed, completed] -= 10000
    ignored = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
    matrices[:, ignored[0], ignored[1]] = np.nan
    got, info = solver.run("spotrfbatched", matrices, upper=upper, padding=3)
    require(info.tolist() == [0, k, 0], f"info {info} expected [0,{k},0]")
    first = la.cholesky(original[:completed, :completed].astype(np.float64), lower=True)
    lower = np.zeros((n, completed))
    lower[:completed] = first
    lower[completed:] = la.solve_triangular(
        first, original[completed:, :completed].astype(np.float64).T, lower=True
    ).T
    logical = got[1].T if upper else got[1]
    ii, jj = np.tril_indices(n)
    selected = jj < completed
    require(
        np.allclose(
            logical[ii[selected], jj[selected]],
            lower[ii[selected], jj[selected]],
            rtol=2 ** (-10),
            atol=2 ** (-16),
        ),
        "contract condition failed",
    )
    gold = la.cholesky(original.astype(np.float64), lower=not upper)
    index = np.triu_indices(n) if upper else np.tril_indices(n)
    for q in (0, 2):
        require(
            np.allclose(got[q][index], gold[index], rtol=2 ** (-10), atol=2 ** (-16)),
            "contract condition failed",
        )


def solve_pointers(solver, rng, n, upper):
    ld, count = n + 3, 7
    factors, right, references, originals = [], [], [], []
    for q in range(count):
        r = rng.normal(size=(n, n)) / np.sqrt(n)
        x = r @ r.T + np.eye(n) * (1 + q)
        f = la.cholesky(x, lower=not upper).astype(np.float32)
        rhs = rng.normal(size=n).astype(np.float32)
        references.append(
            la.cho_solve((f.astype(np.float64), not upper), rhs.astype(np.float64))
        )
        ignored = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
        f[ignored] = np.nan
        raw = np.full((n, ld), -12345.125, dtype=np.float32)
        raw[:, :n] = f.T
        originals.append(raw.reshape(-1).copy())
        factors.append(torch.from_numpy(raw.reshape(-1)).to("npu"))
        rhs_raw = np.full(ld, -12345.125, dtype=np.float32)
        rhs_raw[:n] = rhs
        right.append(torch.from_numpy(rhs_raw).to("npu"))
    order = [6, 2, 4, 0, 5, 1, 3]
    fa = torch.tensor(
        [factors[q].data_ptr() for q in order], dtype=torch.int64, device="npu"
    )
    ba = torch.tensor(
        [right[q].data_ptr() for q in order], dtype=torch.int64, device="npu"
    )
    info = torch.full((1,), -777, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    solver.check(
        solver.lib.aclsolverSpotrsBatched(
            solver.handle,
            int(upper),
            n,
            1,
            fa.data_ptr(),
            ld,
            ba.data_ptr(),
            ld,
            info.data_ptr(),
            count,
        )
    )
    solver.sync()
    require(info.cpu().item() == 0, "contract condition failed")
    verify_solve_pointers(order, right, references, factors, originals, n)


def verify_solve_pointers(order, right, references, factors, originals, n):
    for q in order:
        raw = right[q].cpu().numpy()
        require(np.all(raw[n:] == -12345.125), "contract condition failed")
        require(
            np.allclose(raw[:n], references[q], rtol=2 ** (-10), atol=2 ** (-16)),
            "contract condition failed",
        )
        require(
            np.array_equal(factors[q].cpu().numpy(), originals[q], equal_nan=True),
            "contract condition failed",
        )


def pool_stream_lifetime(solver, rng):
    streams, buffers, arrays, infos, inputs = [], [], [], [], []
    n, count = 512, 11
    try:
        for upper in (False, True):
            stream = C.c_void_p()
            solver.check(solver.acl.aclrtCreateStream(C.byref(stream)))
            streams.append(stream)
            r = rng.normal(size=(n, n)) / np.sqrt(n)
            x = (r @ r.T + np.eye(n) * (2 + int(upper))).astype(np.float32)
            inputs.append(x)
            raw = np.tile(x.T.reshape(-1), count)
            buffer = torch.from_numpy(raw).to("npu")
            buffers.append(buffer)
            arrays.append(
                torch.tensor(
                    [buffer.data_ptr() + q * n * n * 4 for q in range(count)],
                    dtype=torch.int64,
                    device="npu",
                )
            )
            infos.append(torch.full((count,), -777, dtype=torch.int32, device="npu"))
        torch.npu.synchronize()
        handle = C.c_void_p()
        solver.check(solver.lib.aclsolverCreate(C.byref(handle)))
        try:
            for q in range(2):
                solver.check(solver.lib.aclsolverSetStream(handle, streams[q]))
                solver.check(
                    solver.lib.aclsolverSpotrfBatched(
                        handle,
                        q,
                        n,
                        arrays[q].data_ptr(),
                        n,
                        infos[q].data_ptr(),
                        count,
                    )
                )
        finally:
            solver.check(solver.lib.aclsolverDestroy(handle))
        verify_pool_streams(solver, streams, buffers, infos, inputs, n, count)
    finally:
        for stream in streams:
            solver.check(solver.acl.aclrtDestroyStream(stream))


def verify_pool_streams(solver, streams, buffers, infos, inputs, n, count):
    for q in range(2):
        solver.check(solver.acl.aclrtSynchronizeStream(streams[q]))
        require(np.all(infos[q].cpu().numpy() == 0), "contract condition failed")
        raw = buffers[q].cpu().numpy().reshape(count, n, n)
        gold = la.cholesky(inputs[q].astype(np.float64), lower=q == 0)
        idx = np.tril_indices(n) if q == 0 else np.triu_indices(n)
        for output in raw:
            require(
                np.allclose(output.T[idx], gold[idx], rtol=2 ** (-10), atol=2 ** (-16)),
                "contract condition failed",
            )


def parameter_buffers():
    # Parameter positions exclude handle. Combined failures check first-index priority.
    a = torch.eye(17, dtype=torch.float32, device="npu").reshape(-1)
    b = torch.ones(17, dtype=torch.float32, device="npu")
    w = torch.empty(4096, dtype=torch.float32, device="npu")
    torch.npu.synchronize()
    factor_input = torch.eye(129, dtype=torch.float32, device="npu").reshape(-1)
    factor_workspace = torch.empty(65536, dtype=torch.float32, device="npu")
    return a, b, w, factor_input, factor_workspace


def check_handles(solver, record, buffers):
    check = record
    status_call = partial(status_only, solver)
    a, b, w, factor_input, factor_workspace = buffers
    ap, bp = a.data_ptr(), b.data_ptr()
    calls = {
        "Spotrf": [0, 17, ap, 17, None, 0, None],
        "Spotri": [0, 17, ap, 17, w.data_ptr(), 4096, None],
        "Spotrs": [0, 17, 1, ap, 17, bp, 17, None],
        "SpotrfBatched": [0, 0, None, 1, None, 0],
        "SpotrsBatched": [0, 0, 1, None, 1, None, 1, None, 0],
    }
    for name, values in calls.items():
        check(
            name + "/null-handle",
            lambda name=name, values=values: status_call(
                "aclsolver" + name, [None] + values, 9
            ),
        )
        # Factor batchCount=0 allows a null info pointer; other APIs require info.
        if name != "SpotrfBatched":
            check(
                name + "/null-info",
                lambda name=name, values=values: status_call(
                    "aclsolver" + name, [solver.handle] + values, 3
                ),
            )


def check_queries(solver, record, buffers):
    check = record
    expect_call = partial(expect, solver)
    status_call = partial(status_only, solver)
    a, _, _, factor_input, factor_workspace = buffers
    ap = a.data_ptr()
    for name in ("Spotrf", "Spotri"):
        lw = Int32(-777)
        query = "aclsolver" + name + "_bufferSize"
        check(
            query + "/null-handle",
            lambda query=query: status_call(
                query, [None, 0, 17, ap, 17, C.byref(lw)], 9
            ),
        )
        check(
            query + "/null-lwork",
            lambda query=query: status_call(
                query, [solver.handle, 0, 17, ap, 17, None], 3
            ),
        )
        check(
            query + "/above-max",
            lambda query=query: status_call(
                query, [solver.handle, 0, 4097, ap, 4097, C.byref(lw)], 3
            ),
        )
        solver.check(
            getattr(solver.lib, query)(
                solver.handle, 0, 129, factor_input.data_ptr(), 129, C.byref(lw)
            )
        )
        check(
            name + "/workspace-one-short",
            lambda name=name, lw=lw.value: expect_call(
                "aclsolver" + name,
                [
                    0,
                    129,
                    factor_input.data_ptr(),
                    129,
                    factor_workspace.data_ptr(),
                    lw - 1,
                    None,
                ],
                -6,
            ),
        )


def check_factor_dimensions(solver, record, buffers):
    a, _, _, factor_input, factor_workspace = buffers
    ap = a.data_ptr()
    check = record
    expect_call = partial(expect, solver)
    for fn in ("aclsolverSpotrf", "aclsolverSpotri"):
        dimension = 129 if fn == "aclsolverSpotrf" else 17
        base = [
            0,
            dimension,
            factor_input.data_ptr() if dimension == 129 else ap,
            dimension,
            factor_workspace.data_ptr(),
            65536,
            None,
        ]
        for index, value in (
            (1, 7),
            (2, -1),
            (3, None),
            (4, dimension - 1),
            (5, None),
            (6, 0),
        ):
            values = base.copy()
            values[index - 1] = value
            check(
                fn + f"/invalid-{index}",
                lambda fn=fn, values=values, index=index: expect_call(
                    fn, values, -index
                ),
            )
        check(
            fn + "/empty",
            lambda fn=fn: expect_call(fn, [0, 0, None, 1, None, 0, None], success=True),
        )


def check_solve_dimensions(solver, record, buffers):
    a, b, _, _, _ = buffers
    ap, bp = a.data_ptr(), b.data_ptr()
    check = record
    expect_call = partial(expect, solver)
    for index, value in (
        (1, 9),
        (2, -1),
        (3, 129),
        (4, None),
        (5, 16),
        (6, None),
        (7, 16),
    ):
        values = [0, 17, 1, ap, 17, bp, 17, None]
        values[index - 1] = value
        check(
            f"Spotrs/invalid-{index}",
            lambda values=values, index=index: expect_call(
                "aclsolverSpotrs", values, -index
            ),
        )


def check_empty_dimensions(solver, record):
    check = record
    expect_call = partial(expect, solver)
    status_call = partial(status_only, solver)
    check(
        "Spotrs/empty-nrhs",
        lambda: expect_call(
            "aclsolverSpotrs", [0, 17, 0, None, 17, None, 17, None], success=True
        ),
    )
    check(
        "Spotrs/empty-n",
        lambda: expect_call(
            "aclsolverSpotrs", [0, 0, 1, None, 1, None, 1, None], success=True
        ),
    )
    check(
        "SpotrfBatched/empty-n",
        lambda: expect_call(
            "aclsolverSpotrfBatched", [0, 0, None, 1, None, 4], success=True
        ),
    )
    check(
        "SpotrfBatched/empty-count",
        lambda: status_call(
            "aclsolverSpotrfBatched", [solver.handle, 0, 17, None, 17, None, 0], 0
        ),
    )
    check(
        "SpotrsBatched/empty-count",
        lambda: expect_call(
            "aclsolverSpotrsBatched",
            [0, 17, 1, None, 17, None, 17, None, 0],
            success=True,
        ),
    )
    check(
        "SpotrsBatched/empty-n",
        lambda: expect_call(
            "aclsolverSpotrsBatched", [0, 0, 1, None, 1, None, 1, None, 7], success=True
        ),
    )


def check_empty_batches(solver, record):
    check = record
    expect_call = partial(expect, solver)
    for nrhs in (0, 2):
        check(
            f"SpotrsBatched/nrhs-{nrhs}",
            lambda nrhs=nrhs: expect_call(
                "aclsolverSpotrsBatched", [0, 17, nrhs, None, 17, None, 17, None, 7], -3
            ),
        )
    check(
        "SpotrfBatched/negative-count",
        lambda: expect_call("aclsolverSpotrfBatched", [0, 17, None, 17, None, -1], -6),
    )
    check(
        "SpotrsBatched/negative-count",
        lambda: expect_call(
            "aclsolverSpotrsBatched", [0, 17, 1, None, 17, None, 17, None, -1], -9
        ),
    )


def check_numerical_contracts(solver, record):
    rng = np.random.default_rng(91350)
    for upper in (False, True):
        for n in (65, 129, 257):
            for k in (1, 65, n):
                record(
                    f"Spotri/zero-diagonal/{n}/{k}/{upper}",
                    partial(singular_inverse, solver, n, k, upper),
                )
        for n in (16, 65, 130):
            r = rng.normal(size=(n, n)) / np.sqrt(n)
            original = (r @ r.T + np.eye(n) * 2).astype(np.float32)
            for k in sorted({1, min(n, 17), min(n, 65), min(n, 66), n}):
                record(
                    f"Spotrf/partial/{n}/{k}/{upper}",
                    partial(partial_factor, solver, n, k, upper, original),
                )
            record(
                f"Spotrf/deterministic/{n}/{upper}",
                partial(deterministic, solver, n, upper, original),
            )
            record(
                f"SpotrfBatched/noncontiguous/{n}/{upper}",
                partial(pointers, solver, rng, n, upper),
            )
        for n in (129, 201, 257, 513):
            for k in sorted({17, 65, 66, min(n, 129), n}):
                record(
                    f"SpotrfBatched/partial/{n}/{k}/{upper}",
                    partial(batch_partial, solver, rng, n, k, upper),
                )
        for n in (33, 129, 257):
            record(
                f"SpotrsBatched/noncontiguous/{n}/{upper}",
                partial(solve_pointers, solver, rng, n, upper),
            )
    record(
        "scratch-pool/two-streams-and-pending-destroy",
        partial(pool_stream_lifetime, solver, rng),
    )


def null_solve_slots(solver, n, upper, missing_a, missing_b, count=7):
    ld = n + 3
    matrix = np.full((count, n, ld), -12345.125, dtype=np.float32)
    matrix[:, :, :n] = np.eye(n, dtype=np.float32)
    factors = torch.from_numpy(matrix.reshape(-1)).to("npu")
    rhs = torch.ones((count * ld,), dtype=torch.float32, device="npu")
    before = rhs.cpu().numpy().copy()
    ap = [factors.data_ptr() + q * n * ld * 4 for q in range(count)]
    bp = [rhs.data_ptr() + q * ld * 4 for q in range(count)]
    if missing_a:
        ap[-1] = 0
    if missing_b:
        bp[1] = 0
    aa = torch.tensor(ap, dtype=torch.int64, device="npu")
    ba = torch.tensor(bp, dtype=torch.int64, device="npu")
    info = torch.full((1,), -777, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    for _ in range(3):
        solver.check(
            solver.lib.aclsolverSpotrsBatched(
                solver.handle,
                int(upper),
                n,
                1,
                aa.data_ptr(),
                ld,
                ba.data_ptr(),
                ld,
                info.data_ptr(),
                count,
            )
        )
        solver.sync()
        require(
            info.cpu().item() == (-4 if missing_a else -6),
            "non-deterministic scalar pointer error",
        )
        require(
            np.array_equal(rhs.cpu().numpy(), before),
            "invalid pointer batch modified RHS",
        )


def factor_slot_inputs(n, upper, count, missing_pattern):
    rng = np.random.default_rng(51000 + n)
    r = rng.normal(size=(n, n)) / np.sqrt(n)
    original = (r @ r.T + np.eye(n) * 2).astype(np.float32)
    raw = np.full((n, n + 3), -12345.125, dtype=np.float32)
    raw[:, :n] = original.T
    ignored = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
    raw[ignored[1], ignored[0]] = -12345.125
    storage = torch.from_numpy(raw).to("npu").unsqueeze(0).repeat(count, 1, 1)
    missing = np.full(count, missing_pattern == "all", dtype=bool)
    if missing_pattern == "mixed":
        for q in (0, 1, 31, 32, 55, 56, 127, 128, 1023, 1024, 4095, count - 1):
            if q < count:
                missing[q] = True
    # Reverse the physical order so info must follow pointer slots, not storage.
    addresses = np.array(
        [storage.data_ptr() + q * raw.nbytes for q in reversed(range(count))],
        dtype=np.int64,
    )
    addresses[missing] = 0
    return original, raw, storage, addresses, missing


def verify_factor_slot_outputs(original, raw, outputs, missing, upper):
    physical_missing = missing[::-1]
    healthy = np.flatnonzero(~physical_missing)
    reference = outputs[healthy[0]] if healthy.size else raw
    if healthy.size:
        check = paired(
            "spotrfbatched", original, reference[:, : len(original)].T, upper
        )
        require(
            np.isfinite(check["ratio"]) and check["ratio"] <= 5 * check["ratio_cpu"],
            "healthy factor failed LAPACK residual",
        )
        n = len(original)
        ignored = np.tril_indices(n, -1) if upper else np.triu_indices(n, 1)
        require(np.array_equal(reference[:, n:], raw[:, n:]), "padding overwritten")
        require(
            np.array_equal(
                reference[ignored[1], ignored[0]], raw[ignored[1], ignored[0]]
            ),
            "ignored triangle overwritten",
        )
    for start in range(0, len(outputs), 32):
        end = start + 32
        chunk = outputs[start:end].view(np.uint32)
        invalid = physical_missing[start:end]
        require(
            np.all(chunk[invalid] == raw.view(np.uint32)), "null-slot backing modified"
        )
        require(
            np.all(chunk[~invalid] == reference.view(np.uint32)),
            "null slots affected other matrices",
        )


def null_factor_slots(solver, n, upper, count=7, missing_pattern="mixed"):
    original, raw, storage, addresses, missing = factor_slot_inputs(
        n, upper, count, missing_pattern
    )
    pointer_array = torch.from_numpy(addresses).to("npu")
    guarded_info = torch.full((count + 2,), -777, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    solver.check(
        solver.lib.aclsolverSpotrfBatched(
            solver.handle,
            int(upper),
            n,
            pointer_array.data_ptr(),
            n + 3,
            guarded_info.data_ptr() + 4,
            count,
        )
    )
    solver.sync()
    info = guarded_info.cpu().numpy()
    require(info[0] == -777 and info[-1] == -777, "info guard overwritten")
    require(np.array_equal(info[1:-1], np.where(missing, -3, 0)), "incorrect slot info")
    require(
        np.array_equal(pointer_array.cpu().numpy(), addresses), "pointer array modified"
    )
    verify_factor_slot_outputs(original, raw, storage.cpu().numpy(), missing, upper)


def null_factor_array(solver):
    count = 7
    info = torch.full((count + 2,), -777, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    status = solver.lib.aclsolverSpotrfBatched(
        solver.handle, 0, 17, None, 17, info.data_ptr() + 4, count
    )
    require(status == 3, f"null Aarray status {status}, expected INVALID_VALUE")
    solver.sync()
    actual = info.cpu().numpy()
    require(actual[0] == -777 and actual[-1] == -777, "host error overwrote info guard")
    require(np.all(actual[1:-1] == -3), "host null Aarray did not fill every info")


def check_factor_slots(solver, record):
    cases = [(n, 7, "mixed") for n in (1, 8, 17, 33, 65, 129, 257)]
    cases += [(n, count, "mixed") for n, count in product((8, 17), (129, 1025, 4097))]
    cases += [(n, 4097, "mixed") for n in (65, 129, 257)]
    cases += [(n, 7, "all") for n in (1, 17, 33, 65, 129)]
    cases += [(n, 7, "none") for n in (33, 65, 129, 257)]
    for (n, count, missing_pattern), upper in product(cases, (False, True)):
        record(
            f"SpotrfBatched/null-slots/{n}/{count}/{upper}/{missing_pattern}",
            partial(null_factor_slots, solver, n, upper, count, missing_pattern),
        )
    record("SpotrfBatched/null-array-host-error", partial(null_factor_array, solver))


def batch_parameter_infos(solver):
    count = 7
    info = torch.full((count,), 9876, dtype=torch.int32, device="npu")
    torch.npu.synchronize()
    status = solver.lib.aclsolverSpotrfBatched(
        solver.handle, 9, 17, None, 17, info.data_ptr(), count
    )
    require(status == 3, f"status {status}")
    solver.sync()
    require(
        np.all(info.cpu().numpy() == -1),
        "parameter error did not fill complete infoArray",
    )


def poisoned_workspace(solver, n, op, upper):
    rng = np.random.default_rng(31000 + n)
    x = rng.normal(size=(n, n)) / np.sqrt(n)
    a = (x @ x.T + np.eye(n) * 2).astype(np.float32)
    factor = la.cholesky(a.astype(np.float64), lower=not upper).astype(np.float32)
    for fill in (float("nan"), -12345.125):
        out, info = solver.run(
            op,
            a if op == "spotrf" else factor,
            upper=upper,
            padding=3,
            workspace_fill=fill,
        )
        check = paired(op, a, out, upper)
        threshold = (
            max(5 * check["ratio_cpu"], 0.1)
            if op == "spotri"
            else 5 * check["ratio_cpu"]
        )
        require(
            np.all(info == 0) and check["ratio"] <= threshold,
            "product read uninitialized workspace contents",
        )


def check_defensive_cases(solver, record):
    for count, upper, (a, b) in product(
        (129, 1025, 4097), (False, True), ((True, False), (False, True), (True, True))
    ):
        record(
            f"SpotrsBatched/null-slots-large-batch/{count}/{upper}/{a}/{b}",
            partial(null_solve_slots, solver, 8, upper, a, b, count),
        )
    for n, upper, (a, b) in product(
        (1, 8, 17, 33, 128, 129, 257),
        (False, True),
        ((True, False), (False, True), (True, True)),
    ):
        record(
            f"SpotrsBatched/null-slots/{n}/{upper}/{a}/{b}",
            partial(null_solve_slots, solver, n, upper, a, b),
        )
    for n, upper, value in product(
        (8, 65, 129), (False, True), (-1.0, float("nan"), float("inf"), float("-inf"))
    ):
        record(
            f"Spotri/invalid-diagonal/{n}/{upper}/{value}",
            partial(singular_inverse, solver, n, n // 2, upper, value),
        )
    for n, upper, op in product((65, 129, 257), (False, True), ("spotrf", "spotri")):
        record(
            f"{op}/poisoned-workspace/{n}/{upper}",
            partial(poisoned_workspace, solver, n, op, upper),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--library", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    solver = Solver(args.library)
    rows = []
    record = partial(record_result, rows)
    try:
        buffers = parameter_buffers()
        check_handles(solver, record, buffers)
        check_queries(solver, record, buffers)
        check_factor_dimensions(solver, record, buffers)
        check_solve_dimensions(solver, record, buffers)
        check_empty_dimensions(solver, record)
        check_empty_batches(solver, record)
        check_numerical_contracts(solver, record)
        record(
            "SpotrfBatched/all-parameter-infos", partial(batch_parameter_infos, solver)
        )
        check_defensive_cases(solver, record)
        check_factor_slots(solver, record)
    finally:
        solver.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))
    raise SystemExit(0 if rows and all(x["pass"] for x in rows) else 1)


if __name__ == "__main__":
    main()
