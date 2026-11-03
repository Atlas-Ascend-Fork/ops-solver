# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Execute the unchanged October task package inputs on the actual public NPU APIs.

This runner produces DUT outputs only. Use the supplied verify_accuracy.py for
judgement against its frozen CPU references; never replace those references.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))

import argparse
import ctypes as ct
import hashlib
import importlib.util
import json
from collections import namedtuple

import numpy as np
from runtime import Runtime, check
from test_support import report, require

PreparedCall = namedtuple(
    "PreparedCall",
    "args query_args name out_ptr out_shape active_rows dt db b",
)


def padded(a, ld):
    result = np.full((ld, a.shape[1]), 97 - 83j, np.complex64, order="F")
    result[: a.shape[0]] = a
    return result


def prepare_cgeqrf(rt, case, arrays, a, da):
    m, n = case["m"], case["n"]
    dt, db, b = None, None, None
    dt = rt.alloc(min(m, n) * 8)
    args = [m, n, da, a.shape[0], dt]
    query_args = args[:-1]
    name = "aclsolverCgeqrf"
    out_ptr, out_shape, active_rows = da, a.shape, m
    return PreparedCall(
        args, query_args, name, out_ptr, out_shape, active_rows, dt, db, b
    )


def prepare_cungqr(rt, case, arrays, a, da):
    m, n = case["m"], case["n"]
    dt, db, b = None, None, None
    dt = rt.upload(np.ascontiguousarray(arrays["tau32"]))
    args = [m, n, case["k"], da, a.shape[0], dt]
    query_args = args
    name = "aclsolverCungqr"
    out_ptr, out_shape, active_rows = da, a.shape, m
    return PreparedCall(
        args, query_args, name, out_ptr, out_shape, active_rows, dt, db, b
    )


def prepare_cunmqr(rt, case, arrays, a, da):
    m, n = case["m"], case["n"]
    dt, db, b = None, None, None
    dt = rt.upload(np.ascontiguousarray(arrays["tau32"]))
    c = padded(arrays["C32"], case["ldc"])
    dc = rt.upload(c)
    args = [
        0 if case["side"] == "LEFT" else 1,
        0 if case["trans"] == "N" else 2,
        m,
        n,
        case["k"],
        da,
        a.shape[0],
        dt,
        dc,
        c.shape[0],
    ]
    query_args = args
    name = "aclsolverCunmqr"
    out_ptr, out_shape, active_rows = dc, c.shape, m
    return PreparedCall(
        args, query_args, name, out_ptr, out_shape, active_rows, dt, db, b
    )


def prepare_cgels(rt, case, arrays, a, da):
    m, n = case["m"], case["n"]
    dt, db, b = None, None, None
    b = padded(arrays["B32"], case["lddb"])
    db = rt.upload(b)
    x = np.full((case["lddx"], case["nrhs"]), 97 - 83j, np.complex64, order="F")
    dx = rt.upload(x)
    args = [m, n, case["nrhs"], da, a.shape[0], db, b.shape[0], dx, x.shape[0]]
    query_args = args + [None]
    name = "aclsolverCCgels"
    out_ptr, out_shape, active_rows = dx, x.shape, n
    return PreparedCall(
        args, query_args, name, out_ptr, out_shape, active_rows, dt, db, b
    )


def prepare_call(rt, case, arrays, a, da):
    operations = {
        "cgeqrf": prepare_cgeqrf,
        "cungqr": prepare_cungqr,
        "cunmqr": prepare_cunmqr,
        "cgels": prepare_cgels,
    }
    operation = operations.get(case["op"])
    if operation is None:
        raise ValueError(f"Unsupported operator: {case['op']}")
    return operation(rt, case, arrays, a, da)


def execute(rt, case, arrays):
    op = case["op"]
    m, n = case["m"], case["n"]
    a = padded(arrays["A32"], case.get("lda", case.get("ldda", m)))
    da = rt.upload(a)
    extra = {}
    args, query_args, name, out_ptr, out_shape, active_rows, dt, db, b = prepare_call(
        rt, case, arrays, a, da
    )
    length = ct.c_size_t() if op == "cgels" else ct.c_int()
    check(
        getattr(rt.lib, name + "_bufferSize")(rt.handle, *query_args, ct.byref(length))
    )
    work = rt.alloc(length.value * (1 if op == "cgels" else 8))
    info = rt.upload(np.array([-999], np.int32))
    niter = ct.c_int(-999)
    if case.get("case_purpose") == "info":
        # Official QR bad_param probes require the first non-handle argument
        # to be invalid, and must observe the device's actual info write.
        args[0] = -1
    tail = [ct.byref(niter)] if op == "cgels" else []
    status = getattr(rt.lib, name)(rt.handle, *args, work, length.value, *tail, info)
    actual_info = int(rt.download(info, (1,), np.int32)[0])
    is_info = case.get("case_purpose") == "info"
    if is_info:
        require(status != 0, "invalid argument was accepted")
    else:
        check(status)
        require(actual_info == 0, actual_info)
    out = rt.download(out_ptr, out_shape)
    require(np.all(out[active_rows:] == np.complex64(97 - 83j)), "padding modified")
    if op == "cgeqrf" and not is_info:
        extra["tau"] = rt.download(dt, (min(m, n),))
    if op in ("cungqr", "cunmqr"):
        require(
            np.array_equal(rt.download(dt, arrays["tau32"].shape), arrays["tau32"]),
            'np.array_equal(rt.download(dt, arrays["tau32"].shape), arrays["tau32"])',
        )
    if op == "cunmqr":
        require(np.array_equal(rt.download(da, a.shape), a), "read-only A modified")
    if op == "cgels":
        require(np.array_equal(rt.download(db, b.shape), b), "read-only B modified")
        extra["niter"] = niter.value
        if not is_info:
            require(niter.value >= 0, "niter.value >= 0")
    return dict(
        out32=np.ascontiguousarray(out[:active_rows]),
        info=actual_info,
        status="ok",
        dut_status=status,
        **extra,
    )


def case_arrays(gen, by_id, case):
    if case.get("case_purpose") == "info":
        base_arrays = gen.build_case_arrays(by_id[case["base_case_id"]])
        arrays = gen.build_info_arrays(case, base_arrays)
        # The official bad_param payload retains A/B only. Supply
        # the unchanged base tau/C to complete the public API call;
        # the invalid first argument must be rejected before use.
        for key in ("tau32", "C32"):
            if key in base_arrays:
                arrays[key] = base_arrays[key]
    else:
        gen.validate_case(case)
        arrays = gen.build_case_arrays(case)
    return arrays


def export_inputs(args, case, arrays, cid):
    dest = args.perf_inputs / cid
    dest.mkdir(parents=True, exist_ok=True)
    saved = {"a": padded(arrays["A32"], case.get("lda", case.get("ldda", case["m"])))}
    if "tau32" in arrays:
        saved["tau"] = arrays["tau32"]
    if "B32" in arrays:
        saved["b"] = padded(arrays["B32"], case["lddb"])
    if "C32" in arrays:
        saved["b"] = padded(arrays["C32"], case["ldc"])
    hashes = {}
    for key, value in saved.items():
        data = np.asarray(value, dtype=np.complex64).tobytes(order="F")
        (dest / (key + ".bin")).write_bytes(data)
        hashes[key] = hashlib.sha256(data).hexdigest()
    (dest / "identity.json").write_text(
        json.dumps(dict(case=case, sha256=hashes), indent=2)
    )


def checked_outputs(rt, args, case, arrays):
    outputs = []
    for _ in range(2):
        outputs.append(execute(rt, case, arrays))
        rt.free_all()
    for key in outputs[0]:
        require(
            np.asarray(outputs[0][key]).tobytes()
            == np.asarray(outputs[1][key]).tobytes(),
            key,
        )
    if args.adapters:
        from adapter_runtime import execute_adapter

        adapted = execute_adapter(rt, args.adapters, case, arrays)
        for key, value in adapted.items():
            require(
                np.asarray(value).tobytes() == np.asarray(outputs[0][key]).tobytes(),
                "adapter " + key,
            )
    return outputs


def run_case(rt, args, gen, by_id, case):
    failures = []
    cid = case["case_id"]
    output_path = args.out / (cid + ".npz")
    output_path.unlink(missing_ok=True)
    try:
        arrays = case_arrays(gen, by_id, case)
        if args.perf_inputs and case.get("case_purpose") != "info":
            export_inputs(args, case, arrays, cid)
        outputs = checked_outputs(rt, args, case, arrays)
        np.savez(output_path, **outputs[0])
        report(
            json.dumps(
                dict(
                    case_id=cid,
                    execution="PASS",
                    deterministic=True,
                    sha256=hashlib.sha256(outputs[0]["out32"].tobytes()).hexdigest(),
                )
            )
        )
    except Exception as exc:
        failures.append(cid)
        report(json.dumps(dict(case_id=cid, execution="FAIL", error=str(exc))))
        rt.free_all()
    return failures


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument("--library", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=int, required=True)
    ap.add_argument("--case-id", action="append", help="Development subset only")
    ap.add_argument(
        "--perf-inputs",
        type=Path,
        help="Export identical column-major inputs for profiling",
    )
    ap.add_argument(
        "--adapters",
        type=Path,
        help="Also verify the fixed acceptance adapter against the direct API",
    )
    args = ap.parse_args()
    return args


def load_cases(args):
    spec = importlib.util.spec_from_file_location(
        "official_gen", args.package / "gen_data.py"
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    canonical = json.loads((args.package / "canonical_cases.json").read_text())
    original = canonical["cases"]
    by_id = {c["case_id"]: c for c in original}
    cases = original + gen.derive_info_cases(
        original, require_s1=canonical.get("package_scope") != "full"
    )
    if args.case_id:
        wanted = set(args.case_id)
        require(
            wanted <= {c["case_id"] for c in cases},
            'wanted <= {c["case_id"] for c in cases}',
        )
        cases = [c for c in cases if c["case_id"] in wanted]
    return gen, by_id, cases


def record_identity(args):
    identity_paths = {
        "library": Path(args.library),
        "generator": args.package / "gen_data.py",
        "cases": args.package / "canonical_cases.json",
        "references": args.package / "cases/index.json",
        "verifier": args.package / "verify_accuracy.py",
    }
    if args.adapters:
        for adapter in args.adapters.glob("lib*_adapter.so"):
            identity_paths[adapter.name] = adapter
    identity = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in identity_paths.items()
    }
    metadata_path = args.out.parent / "execution-metadata.json"
    metadata = dict(
        sha256=identity,
        device=args.device,
        subset=bool(args.case_id),
        repeat=2,
        adapter_checked=bool(args.adapters),
        completed=False,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return identity_paths, identity, metadata_path, metadata


def main():
    args = parse_args()
    gen, by_id, cases = load_cases(args)
    args.out.mkdir(parents=True, exist_ok=True)
    identity_paths, identity, metadata_path, metadata = record_identity(args)
    rt = Runtime(args.library, args.device)
    failures = []
    try:
        for case in cases:
            failures.extend(run_case(rt, args, gen, by_id, case))
    finally:
        rt.close()
    current = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in identity_paths.items()
    }
    if current != identity:
        raise RuntimeError("Implementation or official inputs changed during testing")
    metadata.update(completed=True, total=len(cases), failed=failures)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    report(
        json.dumps(dict(total=len(cases), failed=failures, subset=bool(args.case_id)))
    )
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
