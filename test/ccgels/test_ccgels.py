# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import ctypes as ct
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common" / "complex_qr"))


# 命名避开 pytest 收集（issue #470）：本文件是 run_package.sh 以
# --library/--device 调用的 CLI 脚本，test_* 前缀会被 pytest 误收集报
# fixture 'rt' not found。independent/test_support 依赖上方 sys.path 铺垫，
# 一律在函数内局部导入以保持模块级 import 区完整（G.FMT.05）。
# 函数参数统一打包为元组（shape/case/spec/arg），单函数形参不超过 5 个。
def upload_sentinels(rt, shape, ab):
    """创建带哨兵填充的 A/B/X 缓冲并上传，返回 (da, db, dx, bp, xp)。

    shape=(m, n, rhs)，ab=(a, b)；X 尾部（第 n 行起）与 dB 的 17+19j
    填充供"不被改写"断言使用。
    """
    m, n, rhs = shape
    a, b = ab
    ap = np.full((m + 8, n), 17 + 19j, dtype=np.complex64, order="F")
    bp = np.full((m + 8, rhs), 17 + 19j, dtype=np.complex64, order="F")
    xp = np.full((n + 8, rhs), 17 + 19j, dtype=np.complex64, order="F")
    ap[:m], bp[:m] = a, b
    return rt.upload(ap), rt.upload(bp), rt.upload(xp), bp, xp


def check_operator(rt):
    from independent import matrix

    rng = np.random.default_rng(953)
    for m, n, rhs in [(7, 3, 1), (16, 16, 8), (65, 33, 3), (257, 131, 8)]:
        a, b = matrix(rng, m, n), matrix(rng, m, rhs)
        da, db, dx, bp, xp = upload_sentinels(rt, (m, n, rhs), (a, b))
        rt.run("aclsolverCCgels", [m, n, rhs, da, m + 8, db, m + 8, dx, n + 8], True)
        got = rt.download(dx, xp.shape)
        ref = np.linalg.lstsq(
            a.astype(np.complex128), b.astype(np.complex128), rcond=None
        )[0]
        np.testing.assert_allclose(got[:n], ref, rtol=2e-4, atol=3e-5)
        np.testing.assert_array_equal(got[n:], xp[n:])
        np.testing.assert_array_equal(rt.download(db, bp.shape), bp)
        rt.free_all()


def run_gels(rt, arg, work=None, lwork=0):
    """直调 aclsolverCCgels，返回 (status, niter, dinfo)。

    rt.run 固化要求 dinfo==0，负向与奇异路径需要自行断言 dinfo 语义
    （issue #532），故绕开 rt.run；niter/dinfo 初值置哨兵以便检测漏写。
    arg=(m, n, rhs, da, lda, db, ldb, dx, ldx)。
    """
    m, n, rhs, da, lda, db, ldb, dx, ldx = arg
    niter = ct.c_int(-999)
    info = rt.upload(np.array([-999], dtype=np.int32))
    status = rt.lib.aclsolverCCgels(
        rt.handle,
        m,
        n,
        rhs,
        da,
        lda,
        db,
        ldb,
        dx,
        ldx,
        work,
        lwork,
        ct.byref(niter),
        info,
    )
    dinfo = int(rt.download(info, (1,), np.int32)[0])
    return status, niter.value, dinfo


def query_workspace(rt, arg):
    """查询 aclsolverCCgels_bufferSize，返回 (status, length)。arg 同 run_gels。"""
    length = ct.c_size_t()
    status = rt.lib.aclsolverCCgels_bufferSize(rt.handle, *arg, None, ct.byref(length))
    return status, length


def gels_workspace(rt, arg):
    """按 bufferSize 查询并分配 workspace，返回 (work, lwork)。"""
    from test_support import require

    status, length = query_workspace(rt, arg)
    require(status == 0, f"bufferSize status={status}")
    return rt.alloc(length.value), length.value


def run_and_check(rt, case, spec):
    """哨兵缓冲上传、workspace 查询、直调与公共校验。

    case=(m, n, rhs, a, b)；spec=(expect_x, rtol, atol, expect_dinfo, tag)。
    断言 status==0、dinfo==expect_dinfo、niter==0；expect_x 为 None 时
    断言 X 整体不被回代改写（奇异路径），否则与期望解 allclose 并校验
    X 尾部哨兵与 dB 未被改写。
    """
    from test_support import require

    m, n, rhs, a, b = case
    expect_x, rtol, atol, expect_dinfo, tag = spec
    da, db, dx, bp, xp = upload_sentinels(rt, (m, n, rhs), (a, b))
    work, lwork = gels_workspace(rt, (m, n, rhs, da, m + 8, db, m + 8, dx, n + 8))
    status, niter, dinfo = run_gels(
        rt, (m, n, rhs, da, m + 8, db, m + 8, dx, n + 8), work, lwork
    )
    require(status == 0, f"{tag} status={status}")
    require(dinfo == expect_dinfo, f"{tag} dinfo={dinfo}, expected {expect_dinfo}")
    require(niter == 0, f"{tag} niter={niter}, expected 0")
    got = rt.download(dx, xp.shape)
    if expect_x is None:
        np.testing.assert_array_equal(got, xp)
    else:
        np.testing.assert_allclose(got[:n], expect_x, rtol=rtol, atol=atol)
        np.testing.assert_array_equal(got[n:], xp[n:])
    np.testing.assert_array_equal(rt.download(db, bp.shape), bp)
    rt.free_all()


def check_boundary(rt):
    """任务域上界形状（n=4096、nrhs=128）。

    A 取单位阵使解可解析（x=b），免除 4096 阶 CPU lstsq 的开销；
    断言 dinfo==0、niter==0、X 尾部哨兵与 dB 不被改写（issue #532）。
    """
    rng = np.random.default_rng(5321)
    m = n = 4096
    rhs = 128
    a = np.eye(m, n, dtype=np.complex64, order="F")
    b = np.asfortranarray(
        (rng.normal(size=(m, rhs)) + 1j * rng.normal(size=(m, rhs))).astype(
            np.complex64
        )
    )
    run_and_check(rt, (m, n, rhs, a, b), (b, 1e-5, 1e-5, 0, "boundary"))


def check_directed(rt):
    """A=I、b=e_j（j>0）与 2x2 定向写回回归（对齐 #502 验收口径，issue #532）。

    j>0 专门覆盖"只写出 x[0]、x[1..n-1] 残留 0"类缺陷：e_j 解的第 j 个
    分量为 1、其余为 0，写回缺失会直接失配；单位阵路径数值精确，可紧断言。
    """
    m = n = 16
    for j in (1, n - 1):
        a = np.eye(m, n, dtype=np.complex64, order="F")
        b = np.zeros((m, 1), dtype=np.complex64, order="F")
        b[j, 0] = 1
        expected = np.zeros((n, 1), dtype=np.complex64)
        expected[j, 0] = 1
        run_and_check(rt, (m, n, 1, a, b), (expected, 0, 1e-6, 0, f"e_{j}"))
    # 2x2 满秩：解分量互不相同且非零（4 与 4.5），写回缺失即失配
    a = np.array([[1, 2], [3, 4]], dtype=np.complex64, order="F")
    b = np.array([[5], [6]], dtype=np.complex64, order="F")
    ref = np.linalg.lstsq(a.astype(np.complex128), b.astype(np.complex128), rcond=None)[
        0
    ]
    run_and_check(rt, (2, 2, 1, a, b), (ref.astype(np.complex64), 1e-5, 1e-6, 0, "2x2"))


def check_singular(rt):
    """精确奇异（第二列全零 -> R[1][1] 恒为 0）：dinfo=2 且 X 不被回代改写。

    全零列经 Householder 变换仍为全零，R[1][1] 精确为 0，不依赖舍入路径；
    求解在首个零对角处提前返回，dinfo 为其 1 基下标（issue #532）。
    """
    m, n, rhs = 3, 2, 1
    a = np.zeros((m, n), dtype=np.complex64, order="F")
    a[:, 0] = [1, 2, 3]
    b = np.ones((m, 1), dtype=np.complex64, order="F")
    run_and_check(rt, (m, n, rhs, a, b), (None, 0, 0, 2, "singular"))


def check_rejected(rt, dims, ld, expect_dinfo, tag):
    """拒绝路径公共校验：执行与 bufferSize 同口径断言 INVALID_VALUE 与 dinfo。

    dims=(m, n, rhs)，ld=(lda, ldb, ldx)。
    """
    from test_support import require

    m, n, rhs = dims
    lda, ldb, ldx = ld
    da = rt.upload(np.zeros((lda * n,), dtype=np.complex64))
    db = rt.upload(np.zeros((ldb * rhs,), dtype=np.complex64))
    dx = rt.upload(np.zeros((ldx * rhs,), dtype=np.complex64))
    arg = (m, n, rhs, da, lda, db, ldb, dx, ldx)
    status, _, dinfo = run_gels(rt, arg)
    require(status == 3, f"{tag} status={status}, expected 3 (INVALID_VALUE)")
    require(dinfo == expect_dinfo, f"{tag} dinfo={dinfo}, expected {expect_dinfo}")
    query, _ = query_workspace(rt, arg)
    require(query == 3, f"{tag} bufferSize status={query}, expected 3")
    rt.free_all()


def check_rejections(rt):
    """任务域上界拒绝路径：nrhs=129 -> dinfo=-3，m/n 超 4096 -> dinfo=-1。

    bufferSize 与执行共用同一校验，两者都必须拒绝；设备缓冲按校验的
    字节足迹（rhs*ldx / lda*n）足额分配，避免 Overlap 判定假阳性。
    """
    # nrhs 超 CC_GELS_MAX_RHS=128：重叠校验在前，缓冲足额后命中参数位 3
    check_rejected(rt, (16, 16, 129), (16, 16, 16), -3, "nrhs=129")
    # m/n 超 CC_GELS_MAX_ORDER=4096：m>=n 保证命中的是上界分支而非 m<n
    check_rejected(rt, (4097, 4097, 1), (4097, 4097, 4097), -1, "order>4096")


def check_all(rt):
    check_operator(rt)
    check_boundary(rt)
    check_directed(rt)
    check_singular(rt)
    check_rejections(rt)


if __name__ == "__main__":
    from independent import run

    run(check_all)
