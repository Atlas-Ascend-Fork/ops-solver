# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Supplementary API contracts: invalid arguments, finite inputs and empty work."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import argparse
import ctypes as ct
import json

import numpy as np
from runtime import Runtime, check
from test_support import report, require


class ContractChecks:
    def __init__(self, rt, results):
        self.rt = rt
        self.results = results
        self.a = self.rt.upload(np.zeros((7, 4), np.complex64, order="F"))
        self.tau = self.rt.upload(np.zeros(4, np.complex64))
        self.b = self.rt.upload(np.zeros((8, 4), np.complex64, order="F"))
        self.x = self.rt.upload(np.zeros((6, 2), np.complex64, order="F"))
        self.niter = ct.c_int(-999)
        self.info = self.rt.alloc(4)
        self.work = self.rt.alloc(4096)
        self.calls = {
            "Cgeqrf": [5, 4, self.a, 7, self.tau, self.work, 512, self.info],
            "Cungqr": [5, 4, 3, self.a, 7, self.tau, self.work, 512, self.info],
            "Cunmqr": [
                0,
                0,
                5,
                4,
                3,
                self.a,
                7,
                self.tau,
                self.b,
                8,
                self.work,
                512,
                self.info,
            ],
            "CCgels": [
                5,
                4,
                2,
                self.a,
                7,
                self.b,
                8,
                self.x,
                6,
                self.work,
                4096,
                ct.byref(self.niter),
                self.info,
            ],
        }

    @staticmethod
    def singular_cases():
        # The reviewed design reports the first exact zero R diagonal and
        # leaves X unchanged rather than dividing by zero and claiming success.
        singular_cases = (
            (4, 4, 1, 0, "single"),
            (7, 4, 2, 2, "single"),
            (129, 97, 64, 64, "single"),
            (7, 4, 2, 0, "all"),
            (129, 97, 64, 2, "multiple"),
        )
        return singular_cases

    def invoke(self, op, changes, expected, label):
        original = self.calls.get(op)
        if original is None:
            raise ValueError(op)
        values = original.copy()
        for index, value in changes.items():
            values[index - 1] = value
        sentinel = np.array([-999], np.int32)
        check(self.rt.acl.aclrtMemcpy(self.info, 4, sentinel.ctypes.data, 4, 1))
        status = getattr(self.rt.lib, "aclsolver" + op)(self.rt.handle, *values)
        got = int(self.rt.download(self.info, (1,), np.int32)[0])
        passed = (
            (status == 0 and got == 0)
            if expected == 0
            else (status == 3 and got == -expected)
        )
        self.results.append(
            dict(
                operator=op,
                label=label,
                status=status,
                info=got,
                expected_info=-expected,
                passed=passed,
            )
        )
        if not passed:
            raise AssertionError(self.results[-1])

    def invalid(self):
        # Expected indices exclude handle, matching the task's devInfo contract.
        invalid = {
            "Cgeqrf": [(1, -1), (2, -1), (3, None), (4, 4), (5, None)],
            "Cungqr": [
                (1, -1),
                (2, 6),
                (3, 5),
                (4, None),
                (5, 4),
                (6, None),
                (7, None),
                (8, 0),
            ],
            "Cunmqr": [
                (1, 2),
                (2, 1),
                (3, -1),
                (4, -1),
                (5, 6),
                (6, None),
                (7, 4),
                (8, None),
                (9, None),
                (10, 4),
            ],
            "CCgels": [
                (1, 3),
                (2, -1),
                (3, -1),
                (4, None),
                (5, 4),
                (6, None),
                (7, 4),
                (8, None),
                (9, 3),
                (10, None),
                (11, 0),
                (12, None),
            ],
        }
        for op, entries in invalid.items():
            for index, value in entries:
                self.invoke(op, {index: value}, index, f"invalid_argument_{index}")

    def null_pointers(self):
        for op, original in self.calls.items():
            values = original.copy()
            values[-1] = None
            status = getattr(self.rt.lib, "aclsolver" + op)(self.rt.handle, *values)
            require(status == 3, (op, "null_info", status))
            self.results.append(
                dict(operator=op, label="null_info", status=status, passed=True)
            )
            status = getattr(self.rt.lib, "aclsolver" + op)(None, *original)
            require(status == 9, (op, "null_handle", status))
            self.results.append(
                dict(operator=op, label="null_handle", status=status, passed=True)
            )

    def overflow_queries(self):
        maximum = 2**31 - 1
        length = ct.c_int(-1)
        byte_length = ct.c_size_t(0)
        queries = {
            "Cgeqrf": [maximum, maximum, self.a, maximum, ct.byref(length)],
            "Cungqr": [
                maximum,
                maximum,
                maximum,
                self.a,
                maximum,
                self.tau,
                ct.byref(length),
            ],
            "Cunmqr": [
                0,
                0,
                maximum,
                maximum,
                17,
                self.a,
                maximum,
                self.tau,
                self.b,
                maximum,
                ct.byref(length),
            ],
            "CCgels": [
                maximum,
                maximum,
                128,
                self.a,
                maximum,
                self.b,
                maximum,
                self.x,
                maximum,
                None,
                ct.byref(byte_length),
            ],
        }
        return queries, length, byte_length

    def overflow(self):
        queries, length, byte_length = self.overflow_queries()
        for op, values in queries.items():
            status = getattr(self.rt.lib, "aclsolver" + op + "_bufferSize")(
                self.rt.handle, *values
            )
            require(status == 3, (op, "workspace_overflow", status))
            self.results.append(
                dict(
                    operator=op, label="workspace_overflow", status=status, passed=True
                )
            )

    def nonfinite(self):
        for op, positions in {
            "Cgeqrf": (3,),
            "Cungqr": (4, 6),
            "Cunmqr": (6, 8, 9),
            "CCgels": (4, 6),
        }.items():
            for position in positions:
                for value in (complex(float("nan"), 0), complex(0, float("inf"))):
                    data = np.zeros((8, 4), np.complex64, order="F")
                    data[1, 0] = (
                        value  # Strictly below the implicit reflector diagonal.
                    )
                    invalid_ptr = self.rt.upload(data)
                    self.invoke(
                        op,
                        {position: invalid_ptr},
                        position,
                        f"nonfinite_argument_{position}_{value}",
                    )

    def least_squares_empty(self):
        self.invoke("CCgels", {1: 0, 5: 1, 7: 1}, 1, "empty_m_positive_n_invalid")
        self.invoke(
            "CCgels",
            {1: 0, 2: 0, 4: None, 5: 1, 6: None, 7: 1, 8: None, 9: 1, 10: None, 11: 0},
            0,
            "empty_m_n",
        )
        require(self.niter.value == 0, "CCgels empty problem must set niter=0")

    def empty(self):
        self.least_squares_empty()
        self.invoke(
            "Cgeqrf", {1: 0, 3: None, 4: 1, 5: None, 6: None, 7: 0}, 0, "empty_m"
        )
        self.invoke("Cgeqrf", {2: 0, 3: None, 5: None, 6: None, 7: 0}, 0, "empty_n")
        self.invoke(
            "Cungqr", {2: 0, 3: 0, 4: None, 6: None, 7: None, 8: 0}, 0, "empty_n"
        )
        self.invoke("Cunmqr", {5: 0, 6: None, 8: None, 11: None, 12: 0}, 0, "zero_k")
        self.invoke(
            "Cunmqr",
            {5: 0, 6: None, 8: None, 9: None, 11: None, 12: 0},
            0,
            "zero_k_null_c",
        )
        self.invoke(
            "CCgels", {3: 0, 4: None, 6: None, 8: None, 10: None, 11: 0}, 0, "zero_rhs"
        )
        require(self.niter.value == 0, "niter.value == 0")
        self.invoke(
            "CCgels",
            {2: 0, 4: None, 6: None, 8: None, 9: 1, 10: None, 11: 0},
            0,
            "empty_n",
        )
        self.invoke(
            "Cungqr",
            {1: 0, 2: 0, 3: 0, 4: None, 5: 1, 6: None, 7: None, 8: 0},
            0,
            "empty_m_n",
        )
        self.invoke(
            "Cunmqr",
            {3: 0, 5: 0, 6: None, 7: 1, 8: None, 9: None, 10: 1, 11: None, 12: 0},
            0,
            "empty_m",
        )
        self.invoke("Cungqr", {3: 0, 6: None, 7: None, 8: 0}, 0, "zero_k_identity")
        identity = self.rt.download(self.a, (7, 4))
        np.testing.assert_array_equal(identity[:5], np.eye(5, 4, dtype=np.complex64))
        np.testing.assert_array_equal(identity[5:], 0)

    def record_singular(self, missing, kind, got):
        self.results.append(
            dict(
                operator="CCgels",
                label=f"zero_diagonal_{missing + 1}_{kind}",
                info=got,
                expected_info=missing + 1,
                x_unchanged=True,
                passed=True,
            )
        )

    def singular(self):
        for m, n, nrhs, missing, kind in self.singular_cases():
            matrix = np.zeros((m + 8, n), np.complex64, order="F")
            matrix[:n] = np.eye(n, dtype=np.complex64)
            matrix[missing, missing] = 0
            if kind == "all":
                matrix[:m] = 0
            elif kind == "multiple":
                matrix[n - 1, n - 1] = 0
            matrix[m:] = complex(97, -83)
            rhs = np.full((m + 32, nrhs), complex(1, 2), np.complex64, order="F")
            output = np.full((n + 8, nrhs), complex(31, -29), np.complex64, order="F")
            da, db, dx = (
                self.rt.upload(matrix),
                self.rt.upload(rhs),
                self.rt.upload(output),
            )
            size = ct.c_size_t()
            arguments = [m, n, nrhs, da, m + 8, db, m + 32, dx, n + 8]
            check(
                self.rt.lib.aclsolverCCgels_bufferSize(
                    self.rt.handle, *arguments, None, ct.byref(size)
                )
            )
            status = self.rt.lib.aclsolverCCgels(
                self.rt.handle,
                *arguments,
                self.rt.alloc(size.value),
                size.value,
                ct.byref(self.niter),
                self.info,
            )
            got = int(self.rt.download(self.info, (1,), np.int32)[0])
            require(
                status == 0 and got == missing + 1 and self.niter.value == 0,
                (
                    status,
                    got,
                    missing,
                ),
            )
            np.testing.assert_array_equal(self.rt.download(dx, output.shape), output)
            np.testing.assert_array_equal(self.rt.download(db, rhs.shape), rhs)
            np.testing.assert_array_equal(
                self.rt.download(da, matrix.shape)[m:], matrix[m:]
            )
            self.record_singular(missing, kind, got)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    rt = Runtime(args.library, args.device)
    results = []
    try:
        checks = ContractChecks(rt, results)
        checks.invalid()
        checks.null_pointers()
        checks.overflow()
        checks.nonfinite()
        checks.empty()
        checks.singular()
        report(
            json.dumps(
                dict(supplementary=True, total=len(results), failed=0, cases=results)
            )
        )
    finally:
        rt.close()


if __name__ == "__main__":
    main()
