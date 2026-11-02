# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared ACL runtime for independent and integration tests."""

import ctypes as ct
import hashlib
import numpy as np
from test_support import require


def check(status):
    if status:
        raise RuntimeError(f"runtime/solver status: {status}")


class Runtime:
    def __init__(self, library, device):
        self.acl = ct.CDLL("libascendcl.so", mode=ct.RTLD_GLOBAL)
        self.lib = ct.CDLL(library)
        p, i, z = ct.c_void_p, ct.c_int, ct.c_size_t
        signatures = {
            "aclInit": [p],
            "aclrtSetDevice": [i],
            "aclrtMalloc": [ct.POINTER(p), z, i],
            "aclrtFree": [p],
            "aclrtMemcpy": [p, z, p, z, i],
            "aclrtSynchronizeDevice": [],
            "aclrtResetDevice": [i],
            "aclFinalize": [],
            "aclrtCreateStream": [ct.POINTER(p)],
            "aclrtDestroyStream": [p],
        }
        for name, args in signatures.items():
            getattr(self.acl, name).argtypes = args
        signatures = {
            "aclsolverCreate": [ct.POINTER(p)],
            "aclsolverDestroy": [p],
            "aclsolverSetStream": [p, p],
            "aclsolverCgeqrf_bufferSize": [p, i, i, p, i, p],
            "aclsolverCgeqrf": [p, i, i, p, i, p, p, i, p],
            "aclsolverCungqr_bufferSize": [p, i, i, i, p, i, p, p],
            "aclsolverCungqr": [p, i, i, i, p, i, p, p, i, p],
            "aclsolverCunmqr_bufferSize": [p, i, i, i, i, i, p, i, p, p, i, p],
            "aclsolverCunmqr": [p, i, i, i, i, i, p, i, p, p, i, p, i, p],
            "aclsolverCCgels_bufferSize": [p, i, i, i, p, i, p, i, p, i, p, p],
            "aclsolverCCgels": [p, i, i, i, p, i, p, i, p, i, p, z, p, p],
        }
        for name, args in signatures.items():
            getattr(self.lib, name).argtypes = args
        check(self.acl.aclInit(None))
        check(self.acl.aclrtSetDevice(device))
        self.device = device
        self.handle = p()
        check(self.lib.aclsolverCreate(ct.byref(self.handle)))
        self.stream = p()
        check(self.acl.aclrtCreateStream(ct.byref(self.stream)))
        check(self.lib.aclsolverSetStream(self.handle, self.stream))
        self.allocations = []
        self.observed = []

    def alloc(self, nbytes):
        ptr = ct.c_void_p()
        check(self.acl.aclrtMalloc(ct.byref(ptr), max(nbytes, 32), 0))
        self.allocations.append(ptr)
        return ptr

    def upload(self, array):
        require(
            array.flags.f_contiguous or array.flags.c_contiguous,
            "smoke.py:75: array.flags.f_contiguous or array.flags.c_contiguous",
        )
        ptr = self.alloc(array.nbytes)
        check(
            self.acl.aclrtMemcpy(ptr, array.nbytes, array.ctypes.data, array.nbytes, 1)
        )
        return ptr

    def download(self, ptr, shape, dtype=np.complex64):
        check(self.acl.aclrtSynchronizeDevice())
        out = np.empty(shape, dtype=dtype, order="F")
        check(self.acl.aclrtMemcpy(out.ctypes.data, out.nbytes, ptr, out.nbytes, 2))
        self.observed.append(hashlib.sha256(out.tobytes()).hexdigest())
        return out

    def free_all(self):
        for ptr in reversed(self.allocations):
            check(self.acl.aclrtFree(ptr))
        self.allocations.clear()

    def close(self):
        self.free_all()
        check(self.lib.aclsolverDestroy(self.handle))
        check(self.acl.aclrtDestroyStream(self.stream))
        check(self.acl.aclrtResetDevice(self.device))
        check(self.acl.aclFinalize())

    def run(self, name, args, byte_workspace=False):
        length = ct.c_size_t() if byte_workspace else ct.c_int()
        query = [self.handle, *args]
        if name == "aclsolverCgeqrf":
            query.pop()  # TAU is an execution output, not a bufferSize argument.
        if byte_workspace:
            query.append(None)
        check(getattr(self.lib, name + "_bufferSize")(*query, ct.byref(length)))
        work = self.alloc(length.value * (1 if byte_workspace else 8))
        info = self.upload(np.array([-999], dtype=np.int32))
        niter = ct.c_int(-999)
        extra = [ct.byref(niter)] if byte_workspace else []
        check(
            getattr(self.lib, name)(
                self.handle, *args, work, length.value, *extra, info
            )
        )
        require(
            self.download(info, (1,), np.int32)[0] == 0,
            "smoke.py:116: self.download(info, (1,), np.int32)[0] == 0",
        )
        if byte_workspace:
            require(niter.value >= 0, "CCgels did not write a valid Host niter")
            self.observed.append(hashlib.sha256(bytes(niter)).hexdigest())
