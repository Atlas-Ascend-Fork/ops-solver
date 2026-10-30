# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ================================================================================================================
"""Direct C API runner. TorchNPU is used only by this test harness for device buffers."""

import argparse
import ctypes as C
import json
import logging
import sys
from itertools import product
from pathlib import Path

import numpy as np
import scipy.linalg as la
import torch
from lapack_residual import paired, finalize

VoidPointer = C.c_void_p
Int32 = C.c_int

QUERY_ARGS = [VoidPointer, Int32, Int32, VoidPointer, Int32, C.POINTER(Int32)]
COMPUTE_ARGS = QUERY_ARGS[:5] + [VoidPointer, Int32, VoidPointer]
SOLVE_ARGS = [VoidPointer] + [Int32] * 3 + [VoidPointer, Int32] * 2 + [VoidPointer]
SOLVER_ARGTYPES = {
    "aclsolverCreate": [C.POINTER(VoidPointer)],
    "aclsolverDestroy": [VoidPointer],
    "aclsolverSetStream": [VoidPointer, VoidPointer],
    "aclsolverSpotrf_bufferSize": QUERY_ARGS,
    "aclsolverSpotri_bufferSize": QUERY_ARGS,
    "aclsolverSpotrf": COMPUTE_ARGS,
    "aclsolverSpotri": COMPUTE_ARGS,
    "aclsolverSpotrs": SOLVE_ARGS,
    "aclsolverSpotrfBatched": QUERY_ARGS[:5] + [VoidPointer, Int32],
    "aclsolverSpotrsBatched": SOLVE_ARGS + [Int32],
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


class Solver:
    def __init__(self, library):
        torch.npu.set_device(0)
        self.acl = C.CDLL("libascendcl.so", mode=C.RTLD_GLOBAL)
        self.lib = C.CDLL(str(Path(library).resolve()))
        for name, args in SOLVER_ARGTYPES.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, Int32
        self.acl.aclrtCreateStream.argtypes = [C.POINTER(VoidPointer)]
        self.acl.aclrtSynchronizeStream.argtypes = [VoidPointer]
        self.acl.aclrtDestroyStream.argtypes = [VoidPointer]
        self.acl.aclrtGetMemInfo.argtypes = [
            Int32,
            C.POINTER(C.c_size_t),
            C.POINTER(C.c_size_t),
        ]
        self.acl.aclrtGetDeviceInfo.argtypes = [C.c_uint, Int32, C.POINTER(C.c_int64)]
        self.handle, self.stream = VoidPointer(), VoidPointer()
        self.check(self.lib.aclsolverCreate(C.byref(self.handle)))
        self.check(self.acl.aclrtCreateStream(C.byref(self.stream)))
        self.check(self.lib.aclsolverSetStream(self.handle, self.stream))

    @staticmethod
    def check(status):
        if status != 0:
            raise RuntimeError(f"Runtime / solver status {status}")

    @staticmethod
    def rhs_buffers(b32, count, ld):
        b = bp = None
        nrhs = 0
        if b32 is not None:
            b32 = np.asarray(b32, dtype=np.float32)
            n, nrhs = b32.shape[-2:]
            bs = np.full((count, nrhs, ld), np.float32(-12345.125))
            bs[:, :, :n] = b32.transpose(0, 2, 1)
            b = torch.from_numpy(bs.reshape(-1)).to("npu")
            bp = torch.tensor(
                [b.data_ptr() + q * nrhs * ld * 4 for q in range(count)],
                dtype=torch.int64,
                device="npu",
            )
        return b, bp, nrhs

    @staticmethod
    def read_result(output, info, count, columns, ld, n, batched):
        raw = output.cpu().numpy().reshape(count, columns, ld)
        require(np.all(raw[:, :, n:] == np.float32(-12345.125)), "padding overwritten")
        result = raw[:, :, :n].transpose(0, 2, 1).copy()
        infos = info.cpu().numpy().copy()
        return (result if batched else result[0]), infos

    def sync(self):
        self.check(self.acl.aclrtSynchronizeStream(self.stream))

    def memory(self):
        free, total = C.c_size_t(), C.c_size_t()
        self.check(
            self.acl.aclrtGetMemInfo(1, C.byref(free), C.byref(total))
        )  # ACL_HBM_MEM
        return {"free_bytes": free.value, "total_bytes": total.value}

    def vector_cores(self):
        value = C.c_int64()
        self.check(
            self.acl.aclrtGetDeviceInfo(0, 201, C.byref(value))
        )  # ACL_DEV_ATTR_VECTOR_CORE_NUM
        require(value.value > 0, "invalid vector core count")
        return value.value

    def close(self):
        self.sync()
        self.check(self.lib.aclsolverDestroy(self.handle))
        self.check(self.acl.aclrtDestroyStream(self.stream))

    def workspace(self, op, upper, n, a, ld, fill):
        workspace, lwork = None, Int32()
        if op in ("spotrf", "spotri"):
            query = getattr(self.lib, "aclsolver" + op.capitalize() + "_bufferSize")
            self.check(
                query(self.handle, int(upper), n, a.data_ptr(), ld, C.byref(lwork))
            )
            workspace = torch.empty((lwork.value,), dtype=torch.float32, device="npu")
            if fill is not None:
                workspace.fill_(fill)
        return workspace, lwork

    def run(self, op, a32, b32=None, upper=False, padding=0, workspace_fill=None):
        batched = op.endswith("batched")
        a32 = np.asarray(a32, dtype=np.float32)
        if not batched:
            a32 = a32[None, ...]
            if b32 is not None:
                b32 = np.asarray(b32, dtype=np.float32)[None, ...]
        count, n, _ = a32.shape
        ld = n + padding
        storage = np.full((count, n, ld), np.float32(-12345.125))
        storage[:, :, :n] = a32.transpose(0, 2, 1)
        a = torch.from_numpy(storage.reshape(-1)).to("npu")
        initial = a.clone()
        info = torch.full(
            (count if op == "spotrfbatched" else 1,),
            -777,
            dtype=torch.int32,
            device="npu",
        )
        ap = torch.tensor(
            [a.data_ptr() + q * n * ld * 4 for q in range(count)],
            dtype=torch.int64,
            device="npu",
        )
        b, bp, nrhs = self.rhs_buffers(b32, count, ld)
        workspace, lwork = self.workspace(op, upper, n, a, ld, workspace_fill)
        torch.npu.synchronize()
        fn = getattr(
            self.lib,
            "aclsolver"
            + {
                "spotrf": "Spotrf",
                "spotrs": "Spotrs",
                "spotri": "Spotri",
                "spotrfbatched": "SpotrfBatched",
                "spotrsbatched": "SpotrsBatched",
            }[op],
        )

        def invoke():
            if op in ("spotrf", "spotri"):
                return fn(
                    self.handle,
                    int(upper),
                    n,
                    a.data_ptr(),
                    ld,
                    workspace.data_ptr(),
                    lwork,
                    info.data_ptr(),
                )
            if op == "spotrs":
                return fn(
                    self.handle,
                    int(upper),
                    n,
                    nrhs,
                    a.data_ptr(),
                    ld,
                    b.data_ptr(),
                    ld,
                    info.data_ptr(),
                )
            if op == "spotrfbatched":
                return fn(
                    self.handle,
                    int(upper),
                    n,
                    ap.data_ptr(),
                    ld,
                    info.data_ptr(),
                    count,
                )
            return fn(
                self.handle,
                int(upper),
                n,
                nrhs,
                ap.data_ptr(),
                ld,
                bp.data_ptr(),
                ld,
                info.data_ptr(),
                count,
            )

        self.check(invoke())
        # Each call owns fresh inputs; repetition is provided by the native benchmark.
        self.sync()
        output = b if b is not None else a
        columns = nrhs if b is not None else n
        if b is not None:
            require(
                np.array_equal(a.cpu().numpy(), initial.cpu().numpy()),
                "factor modified by solve",
            )
        return self.read_result(output, info, count, columns, ld, n, batched)


def smoke_error(op, upper, inputs, x):
    a, factor, b = inputs
    n = a.shape[0]
    if "potrf" in op:
        gold = la.cholesky(a.astype(np.float64), lower=not upper)
        x = np.triu(x) if upper else np.tril(x)
    elif "potrs" in op:
        f = factor.astype(np.float64)
        gold = la.cho_solve((f, not upper), b.astype(np.float64))
    else:
        gold = la.cho_solve((factor.astype(np.float64), not upper), np.eye(n))
        x = np.triu(x) if upper else np.tril(x)
        gold = np.triu(gold) if upper else np.tril(gold)
    error = np.abs(x - gold)
    ratio = float(np.mean(error <= 2**-16 + np.abs(gold) * 2**-10))
    return float(error.max()), ratio


def smoke_outputs(solver, op, inputs, upper, padding):
    a, factors, b = inputs
    input_a = a if "potrf" in op else factors
    input_b = b if "potrs" in op else None
    batched = op.endswith("batched")
    out, info = solver.run(
        op,
        input_a if batched else input_a[0],
        input_b if batched else input_b[0] if input_b is not None else None,
        upper,
        padding,
    )
    outputs = out if batched else out[None, ...]
    return outputs, info


def smoke(solver, n, op, options, upper=False):
    count, padding = options.batch, options.padding
    nrhs = 1 if op.endswith("batched") else options.nrhs
    rng = np.random.default_rng(950 + n)
    r = rng.normal(size=(count, n, n)) / np.sqrt(n)
    a = (r @ r.transpose(0, 2, 1) + np.eye(n) * 2).astype(np.float32)
    factors = np.stack(
        [la.cholesky(x.astype(np.float64), lower=not upper) for x in a]
    ).astype(np.float32)
    b = rng.normal(size=(count, n, nrhs)).astype(np.float32)
    batched = op.endswith("batched")
    outputs, info = smoke_outputs(solver, op, (a, factors, b), upper, padding)
    errors = []
    for q, x in enumerate(outputs):
        errors.append(smoke_error(op, upper, (a[q], factors[q], b[q]), x))
    checks = [
        paired(op, a[q], x, upper, b[q] if "potrs" in op else None)
        for q, x in enumerate(outputs)
    ]
    if batched:
        mean = float(np.mean([item["ratio_cpu"] for item in checks]))
        for item in checks:
            item["ratio_cpu_mean"] = mean
    passed = bool(np.all(info == 0))
    return {
        "op": op,
        "n": n,
        "batch": count if batched else 1,
        "nrhs": nrhs,
        "uplo": "U" if upper else "L",
        "padding": padding,
        "info": info.tolist(),
        "lapack_residual": checks,
        "max_abs": max(x[0] for x in errors),
        "ratio": min(x[1] for x in errors),
        "pass": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--sizes", default="1,2,8,16,17,31,64,65,127,128,129,257")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ops", default="spotrf,spotrs,spotri,spotrfbatched,spotrsbatched"
    )
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--padding", type=int, default=3)
    parser.add_argument("--nrhs", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    solver = Solver(args.library)
    rows = []
    try:
        for n, u, op in product(
            map(int, args.sizes.split(",")), (False, True), args.ops.split(",")
        ):
            row = smoke(solver, n, op, args, u)
            rows.append(row)
        for op in args.ops.split(","):
            selected = [row for row in rows if row["op"] == op]
            finalize(selected, op, complete_set=True)
            for row in selected:
                row["pass"] = row["pass"] and row["final_precision_pass"]
    finally:
        solver.close()
        Path(args.output).write_text(json.dumps(rows, indent=2))
    for row in rows:
        logging.info(json.dumps(row))
    raise SystemExit(0 if rows and all(x["pass"] for x in rows) else 1)


if __name__ == "__main__":
    main()
