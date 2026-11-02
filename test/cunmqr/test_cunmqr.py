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
    rng = np.random.default_rng(952)
    for order, columns, k in [(7, 5, 3), (65, 33, 17), (257, 131, 129)]:
        qr, tau, _, info = cgeqrf(matrix(rng, order, order))
        require(info == 0)
        q, _, info = cungqr(qr, tau[:k])
        require(info == 0)
        for side in (0, 1):
            for trans in (0, 2):
                m, n = (order, columns) if side == 0 else (columns, order)
                a = np.full((order + 8, order), 17 + 19j, dtype=np.complex64, order="F")
                a[:order] = qr
                source = matrix(rng, m, n)
                c = np.full((m + 8, n), 17 + 19j, dtype=np.complex64, order="F")
                c[:m] = source
                da, dt, dc = rt.upload(a), rt.upload(tau[:k]), rt.upload(c)
                rt.run(
                    "aclsolverCunmqr",
                    [side, trans, m, n, k, da, order + 8, dt, dc, m + 8],
                )
                got = rt.download(dc, c.shape)
                opq = q if trans == 0 else q.conj().T
                expected = opq @ source if side == 0 else source @ opq
                np.testing.assert_allclose(got[:m], expected, rtol=2e-4, atol=3e-5)
                np.testing.assert_array_equal(got[m:], c[m:])
                np.testing.assert_array_equal(rt.download(da, a.shape), a)
                rt.free_all()


if __name__ == "__main__":
    run(test_operator)
