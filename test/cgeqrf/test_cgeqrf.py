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
from scipy.linalg.lapack import cungqr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))
from test_support import require

from independent import matrix, run


def test_operator(rt):
    rng = np.random.default_rng(950)
    for m, n in [(1, 1), (7, 3), (3, 7), (65, 33), (257, 131)]:
        a = matrix(rng, m, n)
        padded = np.full((m + 8, n), 17 + 19j, dtype=np.complex64, order="F")
        padded[:m] = a
        da, dt = rt.upload(padded), rt.alloc(min(m, n) * 8)
        rt.run("aclsolverCgeqrf", [m, n, da, m + 8, dt])
        fact, tau = rt.download(da, padded.shape), rt.download(dt, (min(m, n),))
        k = min(m, n)
        q, _, info = cungqr(np.asfortranarray(fact[:m, :k]), tau)
        require(info == 0)
        np.testing.assert_allclose(
            q.astype(np.complex128) @ np.triu(fact[:k]), a, rtol=2e-4, atol=3e-5
        )
        np.testing.assert_array_equal(fact[m:], padded[m:])
        rt.free_all()


if __name__ == "__main__":
    run(test_operator)
