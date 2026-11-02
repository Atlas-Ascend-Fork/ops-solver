# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""ctypes binding of the unchanged solver_adapter.h v2 acceptance protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import ctypes as ct

import numpy as np
from runtime import check


class Case(ct.Structure):
    _fields_ = (
        [("version", ct.c_uint32), ("struct_size", ct.c_uint32)]
        + [(k, ct.c_int64) for k in ("n", "nrhs", "batch", "lda", "ldb")]
        + [
            ("uplo", ct.c_char),
            ("info_probe", ct.c_char_p),
            ("input_kind", ct.c_char_p),
        ]
        + [(k, ct.c_size_t) for k in ("a_bytes", "b_bytes", "out_bytes", "info_count")]
    )


TAU_FIELDS = ("in_ipiv_count", "in_tau_count", "out_ipiv_count", "out_tau_count")


class CaseV2(ct.Structure):
    _fields_ = (
        [("base", Case)]
        + [(k, ct.c_int64) for k in ("m", "k", "ldc", "ldx")]
        + [("side", ct.c_char), ("trans", ct.c_char)]
        + [(k, ct.c_size_t) for k in TAU_FIELDS]
    )


def adapter_case(case, arrays, buffers):
    a, b, tau, out, out_tau = buffers
    m, n = case["m"], case["n"]
    c = CaseV2()
    c.base = Case(
        2,
        ct.sizeof(CaseV2),
        n,
        case.get("nrhs", 0),
        1,
        case.get("lda", case.get("ldda", m)),
        case.get("lddb", m),
        b"\0",
        case.get("info_probe", "").encode(),
        b"matrix",
        a.nbytes,
        b.nbytes,
        out.nbytes,
        1,
    )
    c.m, c.k, c.ldc, c.ldx = (
        m,
        case.get("k", 0),
        case.get("ldc", m),
        case.get("lddx", n),
    )
    c.side, c.trans = (
        case.get("side", "LEFT")[0].encode(),
        case.get("trans", "N").encode(),
    )
    c.in_tau_count, c.out_tau_count = tau.size, out_tau.size
    return c


def load_adapter(directory, case):
    op = case["op"]
    lib = ct.CDLL(
        str(directory / ("lib" + ("ccgels" if op == "cgels" else op) + "_adapter.so"))
    )
    p, z = ct.c_void_p, ct.c_size_t
    lib.solver_adapter_create_v2.argtypes = [ct.POINTER(CaseV2), p, p, ct.POINTER(p)]
    lib.solver_adapter_reset_v2.argtypes = [p, p, z, p, z, p, z, p, z]
    lib.solver_adapter_execute.argtypes = [p, ct.POINTER(ct.c_int)]
    lib.solver_adapter_readback_v2.argtypes = [p, p, z, p, z, p, z, p, z]
    lib.solver_adapter_destroy.argtypes = [p]
    return lib


def adapter_buffers(case, arrays):
    op = case["op"]
    a = np.ascontiguousarray(arrays["A32"])
    b = np.ascontiguousarray(
        arrays.get("B32", arrays.get("C32", np.empty(0, np.complex64)))
    )
    tau = np.ascontiguousarray(
        arrays["tau32"] if op in ("cungqr", "cunmqr") else np.empty(0, np.complex64)
    )
    m, n = case["m"], case["n"]
    out = np.empty((n, case["nrhs"]) if op == "cgels" else (m, n), np.complex64)
    out_tau = np.empty(min(m, n) if op == "cgeqrf" else 0, np.complex64)
    return a, b, tau, out, out_tau


def execute_adapter(rt, directory, case, arrays):
    lib = load_adapter(directory, case)
    op = case["op"]
    p = ct.c_void_p
    a, b, tau, out, out_tau = adapter_buffers(case, arrays)
    c = adapter_case(case, arrays, (a, b, tau, out, out_tau))
    state = p()
    try:
        check(
            lib.solver_adapter_create_v2(
                ct.byref(c), rt.handle, rt.stream, ct.byref(state)
            )
        )
        check(
            lib.solver_adapter_reset_v2(
                state,
                a.ctypes.data,
                a.nbytes,
                b.ctypes.data,
                b.nbytes,
                None,
                0,
                tau.ctypes.data,
                tau.nbytes,
            )
        )
        status, info = ct.c_int(-999), ct.c_int(-999)
        check(lib.solver_adapter_execute(state, ct.byref(status)))
        check(rt.acl.aclrtSynchronizeDevice())
        check(
            lib.solver_adapter_readback_v2(
                state,
                out.ctypes.data,
                out.nbytes,
                ct.byref(info),
                1,
                None,
                0,
                out_tau.ctypes.data,
                out_tau.nbytes,
            )
        )
        result = dict(out32=out, info=info.value, dut_status=status.value)
        if op == "cgeqrf" and case.get("case_purpose") != "info":
            result["tau"] = out_tau
        return result
    finally:
        check(lib.solver_adapter_destroy(state))
