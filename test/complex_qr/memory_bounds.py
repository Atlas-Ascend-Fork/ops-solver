# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Supplementary allocation guards; does not replace device memory instrumentation.

Every input, output and queried workspace gets 256-byte prefix/suffix canaries.
The original accuracy evaluator also verifies matrix padding and read-only inputs.
These checks detect writes to the guards, not arbitrary invalid reads or races.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import argparse
import ctypes as ct
import hashlib
import json

import numpy as np
from numerical_checks import evaluate
from runtime import Runtime, check
from test_support import report


class GuardedRuntime(Runtime):
    GUARD = 256

    def __init__(self, library, device):
        super().__init__(library, device)
        self.guards = []
        self.requested_bytes = 0

    def alloc(self, nbytes):
        size = max(int(nbytes), 1)
        base = ct.c_void_p()
        check(self.acl.aclrtMalloc(ct.byref(base), size + 2 * self.GUARD, 0))
        self.allocations.append(base)
        canary = np.arange(self.GUARD, dtype=np.uint8) ^ np.uint8(0xA5)
        for offset in (0, self.GUARD + size):
            check(
                self.acl.aclrtMemcpy(
                    base.value + offset, self.GUARD, canary.ctypes.data, self.GUARD, 1
                )
            )
        self.guards.append((base.value, size, canary))
        self.requested_bytes += int(nbytes)
        return ct.c_void_p(base.value + self.GUARD)

    def verify_guards(self):
        check(self.acl.aclrtSynchronizeDevice())
        for index, (base, size, expected) in enumerate(self.guards):
            for label, offset in (("prefix", 0), ("suffix", self.GUARD + size)):
                observed = np.empty(self.GUARD, dtype=np.uint8)
                check(
                    self.acl.aclrtMemcpy(
                        observed.ctypes.data, self.GUARD, base + offset, self.GUARD, 2
                    )
                )
                if not np.array_equal(observed, expected):
                    raise AssertionError(
                        f"allocation {index} ({size} bytes): {label} guard modified"
                    )

    def free_all(self):
        try:
            self.verify_guards()
        finally:
            self.guards.clear()
            self.requested_bytes = 0
            super().free_all()


def guard_case(runtime, index, operator, case):
    result = dict(index=index, operator=operator, case=case)
    try:
        observed, reference, multiplier, floor = evaluate(
            runtime, operator, case, np.random.default_rng(951000 + index)
        )
        runtime.verify_guards()
        limit = max(multiplier * reference, floor)
        result.update(
            allocations=len(runtime.guards),
            requested_bytes=runtime.requested_bytes,
            ratio=observed,
            ratio_cpu=reference,
            limit=limit,
            passed=bool(
                np.isfinite(observed) and np.isfinite(reference) and observed <= limit
            ),
        )
    except Exception as error:
        result.update(passed=False, error=str(error))
    finally:
        try:
            runtime.free_all()
        except Exception as error:
            result.update(passed=False, error=str(error))
    return result


def guard_cases():
    cases = []
    for m, n in ((1, 1), (127, 128), (128, 128), (129, 127), (511, 513), (2048, 17)):
        cases.append(("cgeqrf", dict(m=m, n=n, lda=m + 8)))
    for m, n, k in ((65, 65, 33), (129, 127, 65), (1025, 513, 257)):
        cases.append(("cungqr", dict(m=m, n=n, k=k, lda=m + 32)))
    for side in (0, 1):
        for trans in (0, 2):
            for m, n, k in ((127, 255, 65), (1025, 513, 257)):
                cases.append(
                    (
                        "cunmqr",
                        dict(
                            m=m,
                            n=n,
                            k=k,
                            side=side,
                            trans=trans,
                            lda=(m if side == 0 else n) + 8,
                            ldc=m + 32,
                        ),
                    )
                )
    for m, n, rhs in ((31, 17, 1), (128, 128, 8), (129, 97, 32), (1025, 513, 128)):
        cases.append(
            ("cgels", dict(m=m, n=n, nrhs=rhs, ldda=m + 8, lddb=m + 32, lddx=n + 8))
        )
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    library = Path(args.library)
    identity = hashlib.sha256(library.read_bytes()).hexdigest()
    cases = guard_cases()
    runtime = GuardedRuntime(args.library, args.device)
    failures = 0
    report(
        json.dumps(
            dict(
                library_sha256=identity,
                device=args.device,
                guard_bytes=runtime.GUARD,
                scope=(
                    "allocation prefix/suffix writes, matrix padding, read-only inputs; "
                    "not read/race instrumentation"
                ),
            )
        )
    )
    try:
        for index, (operator, case) in enumerate(cases):
            result = guard_case(runtime, index, operator, case)
            failures += not result["passed"]
            report(json.dumps(result))
    finally:
        runtime.close()
    if hashlib.sha256(library.read_bytes()).hexdigest() != identity:
        raise RuntimeError("Implementation changed during guard validation")
    report(json.dumps(dict(total=len(cases), failed=failures, supplementary=True)))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
