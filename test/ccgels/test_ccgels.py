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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))


# 命名避开 pytest 收集（issue #470）：本文件是 run_package.sh 以
# --library/--device 调用的 CLI 脚本，test_* 前缀会被 pytest 误收集报
# fixture 'rt' not found。independent 依赖上方 sys.path 铺垫，在函数内
# 局部导入以保持模块级 import 区完整（G.FMT.05）。
def check_operator(rt):
    from independent import matrix

    rng = np.random.default_rng(953)
    for m, n, rhs in [(7, 3, 1), (16, 16, 8), (65, 33, 3), (257, 131, 8)]:
        a, b = matrix(rng, m, n), matrix(rng, m, rhs)
        ap = np.full((m + 8, n), 17 + 19j, dtype=np.complex64, order="F")
        bp = np.full((m + 8, rhs), 17 + 19j, dtype=np.complex64, order="F")
        xp = np.full((n + 8, rhs), 17 + 19j, dtype=np.complex64, order="F")
        ap[:m], bp[:m] = a, b
        da, db, dx = rt.upload(ap), rt.upload(bp), rt.upload(xp)
        rt.run("aclsolverCCgels", [m, n, rhs, da, m + 8, db, m + 8, dx, n + 8], True)
        got = rt.download(dx, xp.shape)
        ref = np.linalg.lstsq(
            a.astype(np.complex128), b.astype(np.complex128), rcond=None
        )[0]
        np.testing.assert_allclose(got[:n], ref, rtol=2e-4, atol=3e-5)
        np.testing.assert_array_equal(got[n:], xp[n:])
        np.testing.assert_array_equal(rt.download(db, bp.shape), bp)
        rt.free_all()


if __name__ == "__main__":
    from independent import run

    run(check_operator)
