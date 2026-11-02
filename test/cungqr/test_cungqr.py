# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import sys
from pathlib import Path
import numpy as np
from scipy.linalg.lapack import cgeqrf, cungqr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))
from test_support import require

from independent import matrix, run


def test_operator(rt):
    rng = np.random.default_rng(951)
    for m, n, k in [(1, 1, 1), (7, 5, 3), (65, 33, 17), (257, 131, 129), (7, 5, 0)]:
        qr, tau, _, info = cgeqrf(matrix(rng, m, max(k, 1)))
        require(info == 0)
        padded = np.full((m + 8, n), 17 + 19j, dtype=np.complex64, order="F")
        padded[:m] = 0
        padded[:m, :k] = qr[:, :k]
        ref, _, info = (
            cungqr(np.asfortranarray(padded[:m]), tau[:k])
            if k
            else (np.eye(m, n, dtype=np.complex64), None, 0)
        )
        require(info == 0)
        da, dt = rt.upload(padded), rt.upload(tau[:k]) if k else None
        rt.run("aclsolverCungqr", [m, n, k, da, m + 8, dt])
        got = rt.download(da, padded.shape)
        np.testing.assert_allclose(got[:m], ref, rtol=2e-4, atol=3e-5)
        np.testing.assert_array_equal(got[m:], padded[m:])
        rt.free_all()


if __name__ == "__main__":
    run(test_operator)
