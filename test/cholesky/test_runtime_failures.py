# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Real NPU calls with test-only runtime failure injection and cache reuse checks."""

import argparse
import ctypes as C
import json
import logging
import sys
from pathlib import Path
import numpy as np
import torch
from runner import Solver, require


def buffers(n):
    return (
        torch.eye(n, dtype=torch.float32, device="npu"),
        torch.ones(n * 2, dtype=torch.float32, device="npu"),
    )


def solve(solver, n, a, b, info):
    return solver.lib.aclsolverSpotrs(
        solver.handle, 0, n, 2, a.data_ptr(), n, b.data_ptr(), n, info.data_ptr()
    )


def check_growth(library, hooks, kind):
    solver = Solver(library)
    try:
        small_a, small_b = buffers(129)
        large_a, large_b = buffers(257)
        info = torch.empty(1, dtype=torch.int32, device="npu")
        torch.npu.synchronize()

        solver.check(solve(solver, 129, small_a, small_b, info))
        solver.sync()
        hooks.choleskyTestArm(kind)
        require(
            solve(solver, 257, large_a, large_b, info) == 2,
            "growth failure did not report ALLOC_FAILED",
        )
        solver.sync()
        require(
            hooks.choleskyTestMallocCount() == 1,
            "unexpected allocation count on failed growth",
        )
        solver.check(solve(solver, 129, small_a, small_b, info))
        solver.sync()
        require(
            hooks.choleskyTestMallocCount() == 1,
            "failed growth destroyed reusable old cache",
        )
        require(np.all(small_b.cpu().numpy() == 1), "cache reuse corrupted the solve")
    finally:
        hooks.choleskyTestArm(0)
        solver.close()


def check_launch(library, hooks, pooled):
    solver = Solver(library)
    try:
        n = 129 if pooled else 8
        a, b = buffers(n)
        info = torch.empty(1, dtype=torch.int32, device="npu")
        torch.npu.synchronize()
        hooks.choleskyTestArm(3)
        if pooled:
            status = solve(solver, n, a, b, info)
        else:
            status = solver.lib.aclsolverSpotrf(
                solver.handle, 0, n, a.data_ptr(), n, None, 0, info.data_ptr()
            )
        require(status == 5, "thread-level launch error was reported as SUCCESS")
        solver.sync()
    finally:
        hooks.choleskyTestArm(0)
        solver.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    hooks = C.CDLL(None)
    hooks.choleskyTestArm.argtypes = [C.c_int]
    hooks.choleskyTestMallocCount.restype = C.c_uint64
    rows = []
    cases = [
        ("cache/malloc-failure", lambda: check_growth(args.library, hooks, 1)),
        ("cache/event-failure", lambda: check_growth(args.library, hooks, 2)),
        ("launch/pooled", lambda: check_launch(args.library, hooks, True)),
        ("launch/no-pool", lambda: check_launch(args.library, hooks, False)),
    ]
    for name, body in cases:
        try:
            body()
            rows.append({"test": name, "pass": True})
        except Exception as exc:
            rows.append({"test": name, "pass": False, "error": str(exc)})
    args.output.write_text(json.dumps(rows, indent=2))
    logging.info(json.dumps(rows))
    raise SystemExit(0 if all(row["pass"] for row in rows) else 1)


if __name__ == "__main__":
    main()
