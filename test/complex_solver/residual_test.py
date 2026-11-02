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


"""Device API residual tests against SciPy LAPACK; accepts external case JSON files."""

import argparse
import ctypes as ct
import json
import logging
import sys
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.linalg.lapack import cgetrf, cgetri

BOUNDARY_ORDERS = (
    1,
    2,
    3,
    7,
    8,
    15,
    16,
    17,
    31,
    32,
    33,
    63,
    64,
    65,
    127,
    128,
    129,
    255,
    256,
    257,
    512,
)

EPS = 2.0**-24
PTR = ct.c_void_p
INT = ct.c_int


def check(status, operation):
    if status:
        raise RuntimeError(f"{operation}: status={status}")


class Runtime:
    def __init__(self, library):
        self.acl = ct.CDLL("libascendcl.so", mode=ct.RTLD_GLOBAL)
        self.solver = ct.CDLL(str(Path(library).resolve()), mode=ct.RTLD_GLOBAL)
        signatures = {
            "aclInit": [PTR],
            "aclrtSetDevice": [INT],
            "aclrtCreateStream": [ct.POINTER(PTR)],
            "aclrtSynchronizeStream": [PTR],
            "aclrtDestroyStream": [PTR],
            "aclrtResetDevice": [INT],
            "aclFinalize": [],
            "aclrtMalloc": [ct.POINTER(PTR), ct.c_size_t, INT],
            "aclrtFree": [PTR],
            "aclrtMemcpy": [PTR, ct.c_size_t, PTR, ct.c_size_t, INT],
            "aclrtCreateEventWithFlag": [ct.POINTER(PTR), ct.c_uint],
            "aclrtRecordEvent": [PTR, PTR],
            "aclrtSynchronizeEvent": [PTR],
            "aclrtDestroyEvent": [PTR],
            "aclrtEventElapsedTime": [ct.POINTER(ct.c_float), PTR, PTR],
        }
        for name, args in signatures.items():
            getattr(self.acl, name).argtypes = args
        signatures = {
            "aclsolverCreate": [ct.POINTER(PTR)],
            "aclsolverDestroy": [PTR],
            "aclsolverSetStream": [PTR, PTR],
            "aclsolverCgetrf_bufferSize": [PTR, INT, INT, PTR, INT, ct.POINTER(INT)],
            "aclsolverCgetrf": [PTR, INT, INT, PTR, INT, PTR, PTR, PTR],
            "aclsolverCgetri_bufferSize": [PTR, INT, PTR, INT, ct.POINTER(INT)],
            "aclsolverCgetri": [PTR, INT, PTR, INT, PTR, PTR, INT, PTR],
            "aclsolverCgetriBatched": [PTR, INT, PTR, INT, PTR, PTR, INT, PTR, INT],
            "aclsolverCmatinvBatched": [PTR, INT, PTR, INT, PTR, INT, PTR, INT],
        }
        for name, args in signatures.items():
            getattr(self.solver, name).argtypes = args
        self.stream, self.handle = PTR(), PTR()
        self.allocations = []
        check(self.acl.aclInit(None), "aclInit")
        check(self.acl.aclrtSetDevice(0), "aclrtSetDevice")
        check(self.acl.aclrtCreateStream(ct.byref(self.stream)), "create stream")
        check(self.solver.aclsolverCreate(ct.byref(self.handle)), "create handle")
        check(self.solver.aclsolverSetStream(self.handle, self.stream), "set stream")

    def sync(self):
        check(self.acl.aclrtSynchronizeStream(self.stream), "synchronize")

    def allocate(self, array):
        array = np.ascontiguousarray(array)
        pointer = PTR()
        # HUGE_FIRST permits large acceptance buffers and falls back to normal pages.
        check(
            self.acl.aclrtMalloc(ct.byref(pointer), max(array.nbytes, 1), 0),
            "Device allocation",
        )
        self.allocations.append(pointer)
        self.upload(pointer, array)
        return pointer

    def upload(self, pointer, array):
        check(
            self.acl.aclrtMemcpy(
                pointer, array.nbytes, PTR(array.ctypes.data), array.nbytes, 1
            ),
            "upload",
        )

    def download(self, pointer, array):
        self.sync()
        check(
            self.acl.aclrtMemcpy(
                PTR(array.ctypes.data), array.nbytes, pointer, array.nbytes, 2
            ),
            "download",
        )
        return array

    def release(self):
        self.sync()
        for pointer in reversed(self.allocations):
            check(self.acl.aclrtFree(pointer), "Device free")
        self.allocations.clear()

    def close(self):
        self.release()
        check(self.solver.aclsolverDestroy(self.handle), "destroy handle")
        check(self.acl.aclrtDestroyStream(self.stream), "destroy stream")
        check(self.acl.aclrtResetDevice(0), "reset Device")
        check(self.acl.aclFinalize(), "aclFinalize")

    def time_interval(self, invoke, reset, iterations):
        start, end = PTR(), PTR()
        check(self.acl.aclrtCreateEventWithFlag(ct.byref(start), 8), "start event")
        check(self.acl.aclrtCreateEventWithFlag(ct.byref(end), 8), "end event")
        values = []
        try:
            for _ in range(iterations + 5):
                reset()
                check(self.acl.aclrtRecordEvent(start, self.stream), "record start")
                invoke()
                check(self.acl.aclrtRecordEvent(end, self.stream), "record end")
                check(self.acl.aclrtSynchronizeEvent(end), "sync end")
                elapsed = ct.c_float()
                check(
                    self.acl.aclrtEventElapsedTime(ct.byref(elapsed), start, end),
                    "elapsed time",
                )
                values.append(elapsed.value)
        finally:
            self.acl.aclrtDestroyEvent(start)
            self.acl.aclrtDestroyEvent(end)
        # This is a stream interval. Host dispatch gaps/metadata transfers may be included.
        return float(np.mean(values[5:]))


def lu_residual(a, lu, pivots):
    m, n = a.shape
    k = min(m, n)
    lower = np.tril(lu[:, :k].astype(np.complex128), -1) + np.eye(m, k)
    upper = np.triu(lu[:k, :].astype(np.complex128))
    restored = lower @ upper
    for row in range(k - 1, -1, -1):
        pivot = int(pivots[row])
        restored[[row, pivot]] = restored[[pivot, row]]
    return float(np.linalg.norm(restored - a, 1) / (n * np.linalg.norm(a, 1) * EPS))


def inverse_residual(a, inverse):
    a, inverse = a.astype(np.complex128), inverse.astype(np.complex128)
    n = a.shape[0]
    error = np.linalg.norm(np.eye(n) - inverse @ a, 1)
    return float(error / (n * np.linalg.norm(a, 1) * np.linalg.norm(inverse, 1) * EPS))


def reference_lu(a, pivot):
    if pivot:
        lu, pivots, info = cgetrf(a)
        check(info, "CPU cgetrf")
        return lu, pivots
    lu = a.copy()
    m, n = a.shape
    for row in range(min(m, n)):
        trailing = slice(row + 1, None)
        lu[trailing, row] /= lu[row, row]
        lu[trailing, trailing] -= lu[trailing, row, None] * lu[None, row, trailing]
    return lu, np.arange(min(m, n), dtype=np.int32)


def matrix_distribution(m, n, seed, normal):
    random = np.random.default_rng(seed)
    if normal:
        a = random.normal(0.5, 0.7, (m, n)) + 1j * random.normal(-0.5, 0.7, (m, n))
    else:
        a = random.uniform(-5, 5, (m, n)) + 1j * random.uniform(-5, 5, (m, n))
    return a.astype(np.complex64)


def make_matrix(m, n, seed, normal):
    a = matrix_distribution(m, n, seed, normal)
    a[np.arange(min(m, n)), np.arange(min(m, n))] += n
    return a


def prepare_reference(op, case, index):
    n = case["n"]
    d = SimpleNamespace(n=n, m=case.get("m", n), batch=case.get("batch", 1))
    d.lda = case.get("lda", d.m)
    d.ldc = case.get("ldc", case.get("lda_inv", n))
    d.pivot = case.get("pivot", True)
    generator = make_matrix if case.get("dominant", True) else matrix_distribution
    d.original = generator(
        d.m, n, case.get("seed", index + 97), case.get("normal", index % 2 != 0)
    )
    lu, d.cpu_pivot = reference_lu(d.original, d.pivot)
    if op == "cgetrf":
        d.cpu_residual = lu_residual(d.original.astype(np.complex128), lu, d.cpu_pivot)
        d.source = d.original
    else:
        inverse, info = cgetri(lu, d.cpu_pivot)
        check(info, "CPU cgetri")
        d.cpu_residual = inverse_residual(d.original, inverse)
        d.source = d.original if op == "cmatinvbatched" else lu
    return d


def prepare_device(runtime, op, d):
    d.sentinel = np.complex64(7 - 11j)
    d.matrix = np.full((d.batch, d.n, d.lda), d.sentinel, np.complex64)
    d.matrix[:, :, : d.m] = d.source.T
    d.output = np.full((d.batch, d.n, d.ldc), d.sentinel, np.complex64)
    d.da = runtime.allocate(d.matrix)
    pivots = np.broadcast_to(
        (d.cpu_pivot + 1).astype(np.int32), (d.batch, min(d.m, d.n))
    ).copy()
    d.dp = runtime.allocate(pivots) if d.pivot else None
    d.info = runtime.allocate(np.zeros(d.batch, np.int32))
    d.dc = runtime.allocate(d.output) if "batched" in op else d.da


def single_invocation(runtime, op, d):
    lwork = INT()
    if op == "cgetrf":
        status = runtime.solver.aclsolverCgetrf_bufferSize(
            runtime.handle, d.m, d.n, d.da, d.lda, ct.byref(lwork)
        )
    else:
        status = runtime.solver.aclsolverCgetri_bufferSize(
            runtime.handle, d.n, d.da, d.lda, ct.byref(lwork)
        )
    check(status, op + " query")
    workspace = None
    if lwork.value or op == "cgetri":
        workspace = runtime.allocate(np.zeros(lwork.value, np.complex64))
    if op == "cgetrf":
        api = runtime.solver.aclsolverCgetrf
        args = (runtime.handle, d.m, d.n, d.da, d.lda, workspace, d.dp, d.info)
    else:
        api = runtime.solver.aclsolverCgetri
        args = (runtime.handle, d.n, d.da, d.lda, d.dp, workspace, lwork, d.info)

    def invoke():
        check(api(*args), op)

    return invoke


def batch_invocation(runtime, op, d):
    aa = runtime.allocate(
        np.arange(d.batch, dtype=np.uint64) * d.n * d.lda * 8 + d.da.value
    )
    cc = runtime.allocate(
        np.arange(d.batch, dtype=np.uint64) * d.n * d.ldc * 8 + d.dc.value
    )
    if op == "cmatinvbatched":
        api = runtime.solver.aclsolverCmatinvBatched
        args = (runtime.handle, d.n, aa, d.lda, cc, d.ldc, d.info, d.batch)
    else:
        api = runtime.solver.aclsolverCgetriBatched
        args = (runtime.handle, d.n, aa, d.lda, d.dp, cc, d.ldc, d.info, d.batch)

    def invoke():
        check(api(*args), op)

    return invoke


def validate_result(runtime, op, d, result):
    rows = d.m if op == "cgetrf" else d.n
    if not np.all(result[:, :, rows:] == d.sentinel):
        raise AssertionError("padding changed")
    if op == "cgetrf":
        pivot = (
            runtime.download(d.dp, np.empty(min(d.m, d.n), np.int32)) - 1
            if d.pivot
            else d.cpu_pivot
        )
        residual = lu_residual(
            d.original.astype(np.complex128), result[0, :, : d.m].T, pivot
        )
    else:
        if not np.all(result == result[0]):
            raise AssertionError("equal batch inputs produced different results")
        residual = inverse_residual(d.original, result[0, :, : d.n].T)
    floor = 0.05 if op in ("cgetri", "cgetribatched") else 0.5
    limit = max(5 * d.cpu_residual, floor)
    if not np.isfinite(residual) or residual > limit:
        raise AssertionError(
            f"residual={residual}, CPU={d.cpu_residual}, limit={limit}"
        )
    return residual, limit


def metadata_result(runtime, d):
    info = runtime.download(d.info, np.empty(d.batch, np.int32))
    pivots = (
        runtime.download(d.dp, np.empty((d.batch, min(d.m, d.n)), np.int32))
        if d.dp is not None
        else None
    )
    return info, pivots


def validate_metadata_repeat(runtime, d, info, pivots):
    repeated_info, repeated_pivots = metadata_result(runtime, d)
    if not np.array_equal(info, repeated_info):
        raise AssertionError("info determinism failed")
    if pivots is not None and not np.array_equal(pivots, repeated_pivots):
        raise AssertionError("pivot determinism failed")


def run_case(runtime, op, case, index, iterations):
    d = prepare_reference(op, case, index)
    prepare_device(runtime, op, d)
    invoke = (
        batch_invocation(runtime, op, d)
        if "batched" in op
        else single_invocation(runtime, op, d)
    )
    started = time.monotonic()
    invoke()
    values, pivots = metadata_result(runtime, d)
    if np.any(values):
        raise AssertionError(f"Device info {values[:16]}")
    result = runtime.download(
        d.dc, np.empty_like(d.output if "batched" in op else d.matrix)
    )
    residual, limit = validate_result(runtime, op, d, result)
    runtime.upload(d.da, d.matrix)
    invoke()
    repeat = runtime.download(d.dc, np.empty_like(result))
    validate_metadata_repeat(runtime, d, values, pivots)
    if not np.array_equal(result.view(np.uint32), repeat.view(np.uint32)):
        raise AssertionError("determinism failed")
    interval = (
        runtime.time_interval(
            invoke, lambda: runtime.upload(d.da, d.matrix), iterations
        )
        if iterations
        else None
    )
    runtime.release()
    return {
        "op": op,
        "case": case,
        "residual": residual,
        "cpu_residual": d.cpu_residual,
        "limit": limit,
        "stream_interval_ms": interval,
        "elapsed_s": time.monotonic() - started,
        "passed": True,
    }


def resident_regression_cases(batched):
    # Normal inputs expose cancellation in the forward/backward triangular sums.
    cases = []
    for n, seed in (
        (80, 108),
        (90, 178),
        (124, 196),
        (124, 205),
        (212, 276),
        (242, 294),
    ):
        cases.append(
            {
                "n": n,
                "batch": 3 if batched else 1,
                "lda": n + 3,
                "ldc": n + 5,
                "seed": seed,
                "normal": True,
            }
        )
    return cases


def panel_regression_cases():
    # Cover distributed reuse, distributed-to-local transition, and width reduction.
    cases = []
    for index, (m, n) in enumerate(((1000, 100), (800, 64), (1000, 40))):
        for padding in (8, 32):
            for pivot in (False, True):
                cases.append(
                    {
                        "m": m,
                        "n": n,
                        "lda": m + padding,
                        "pivot": pivot,
                        "dominant": not pivot,
                        "seed": 197 + index,
                    }
                )
    return cases


def rectangular_boundary_cases(pivots):
    cases = []
    for m, n in ((31, 65), (65, 31), (127, 129), (129, 127)):
        for padding in (8, 32):
            for pivot in pivots:
                cases.append({"m": m, "n": n, "lda": m + padding, "pivot": pivot})
    return cases


def boundary_regression_cases(op):
    maximum = 32 if op == "cmatinvbatched" else 512
    batched = "batched" in op
    pivots = (True,) if op == "cmatinvbatched" else (False, True)
    cases = []
    batch = 3 if batched else 1
    for n in BOUNDARY_ORDERS:
        if n > maximum:
            continue
        for padding in (8, 32):
            for pivot in pivots:
                cases.append(
                    {
                        "n": n,
                        "lda": n + padding,
                        "ldc": n + padding,
                        "batch": batch,
                        "pivot": pivot,
                    }
                )
    if op == "cgetrf":
        cases += rectangular_boundary_cases(pivots)
        cases += panel_regression_cases()
    if batched:
        cases += [
            {"n": 8, "lda": 16, "ldc": 40, "batch": batch}
            for batch in (1, 8, 128, 1024, 3000)
        ]
    return cases


def adversarial_regression_cases(op):
    orders = (2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32)
    if op != "cmatinvbatched":
        orders += (33, 65, 129, 257)
    return [
        {
            "n": n,
            "lda": n + 3,
            "ldc": n + 5,
            "batch": 67 if "batched" in op else 1,
            "seed": 97 + seed,
            "normal": seed % 2 != 0,
            "dominant": False,
        }
        for n in orders
        for seed in range(4)
    ]


def selected_cases(args, parser):
    if args.panel_regression:
        if args.op != "cgetrf":
            parser.error("--panel-regression requires --op cgetrf")
        return panel_regression_cases()
    if args.adversarial_regression:
        return adversarial_regression_cases(args.op)
    if args.boundary_regression:
        return boundary_regression_cases(args.op)
    if args.resident_regression:
        if args.op not in ("cgetri", "cgetribatched"):
            parser.error("--resident-regression requires a LU-inverse interface")
        return resident_regression_cases(args.op == "cgetribatched")
    else:
        if not args.case_file:
            parser.error("provide --case-file or a regression option")
        return json.loads(Path(args.case_file).read_text())["cases"]


def evaluate_case(runtime, args, case, index):
    try:
        return run_case(runtime, args.op, case, index, args.iterations)
    except Exception as error:
        runtime.release()
        return {"op": args.op, "case": case, "passed": False, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--op",
        choices=["cgetrf", "cgetri", "cgetribatched", "cmatinvbatched"],
        required=True,
    )
    parser.add_argument("--case-file")
    parser.add_argument("--resident-regression", action="store_true")
    parser.add_argument("--boundary-regression", action="store_true")
    parser.add_argument("--panel-regression", action="store_true")
    parser.add_argument("--adversarial-regression", action="store_true")
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cases = selected_cases(args, parser)
    runtime = Runtime(args.library)
    failed = 0
    try:
        with open(args.output, "w", encoding="utf-8") as output:
            for index, case in enumerate(cases[: args.limit]):
                result = evaluate_case(runtime, args, case, index)
                failed += not result["passed"]
                logging.info(json.dumps(result))
                output.write(json.dumps(result) + "\n")
                output.flush()
    finally:
        runtime.close()
    return bool(failed)


if __name__ == "__main__":
    raise SystemExit(main())
