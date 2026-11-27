# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check data hook imports on the Python version running this test."""

import subprocess
import sys
import unittest
from pathlib import Path

# 子进程校验代码（python -c 载体）：用最小 numpy/scipy stub 隔离数值依赖，
# 使本测试可在最小解释器上捕获 fixtures 链的 import 期错误。提取为模块级
# 常量以控制测试方法的 nbnc 行数（issue #452 的 CodeCheck 超大函数整改）。
# stub 需提供 nan/inf 标量：#352 起 fixtures 模块级使用 np.nan/np.inf
# （INVALID_DIAGONAL_VALUES），缺失时子进程 import 期即 AttributeError。
CHECK_CODE = """
import runpy, sys, types
numpy = types.ModuleType('numpy')
numpy.ndarray = type('ndarray', (), {})
numpy.asarray = lambda value: value
numpy.all = lambda value: value
numpy.array_equal = lambda left, right: True
numpy.uint32 = object
numpy.float32 = object
numpy.complex64 = object
numpy.nan = float('nan')
numpy.inf = float('inf')
scipy = types.ModuleType('scipy')
linalg = types.ModuleType('scipy.linalg')
linalg.lapack = types.ModuleType('scipy.linalg.lapack')
linalg.lapack.spotrf = linalg.lapack.spotrs = linalg.lapack.spotri = lambda *args: None
sys.modules.update({'numpy': numpy, 'scipy': scipy, 'scipy.linalg': linalg,
                    'scipy.linalg.lapack': linalg.lapack})
fixtures = runpy.run_path(sys.argv[1])
unbatch = fixtures['unbatch_op']
assert unbatch('cpotrfBatched') == 'cpotrf'
assert unbatch('cpotrs') == 'cpotrs'
assert unbatch('BatchedInside') == 'BatchedInside'
assert unbatch('cpotrs_batched') == 'cpotrs'  # issue #658: 下划线名归一化
specs = fixtures['case_specs']
# issue #658: 下划线与驼峰 op 名产生完全一致的 spec 列表
assert list(specs('cpotrs_batched')) == list(specs('cpotrsBatched'))
residual = runpy.run_path(sys.argv[2])
evaluate = residual['evaluate']
evaluate.__globals__['paired'] = lambda *args: {'ratio': 1.0, 'ratio_cpu': 2.0}
assert evaluate({'op': 'spotrf', 'uplo': 'L'}, {'A32': None},
                {'info': 0, 'out32': None})[0]['physical_slots_checked'] == 1
verifier = runpy.run_path(sys.argv[3])
main = verifier['main']
class FakeMatrix:
    def view(self, dtype):
        return self
main.__globals__['read'] = lambda *args: FakeMatrix()
main.__globals__['paired'] = lambda *args: {'ratio': 0.0, 'ratio_cpu': 0.0}
assert main() == 0
"""


class PythonCompatibilityTest(unittest.TestCase):
    def test_import_and_operator_names(self):
        fixture_path = Path(__file__).with_name("fixtures.py")
        cholesky_path = Path(__file__).resolve().parents[2] / "cholesky"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                CHECK_CODE,
                str(fixture_path),
                str(cholesky_path / "lapack_residual.py"),
                str(cholesky_path / "data" / "verify_result.py"),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
