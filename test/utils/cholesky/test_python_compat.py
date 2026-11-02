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


class PythonCompatibilityTest(unittest.TestCase):
    def test_import_and_operator_names(self):
        # Numeric dependencies are isolated so this test can run on a minimal
        # interpreter and catch errors that occur while importing the module.
        code = """
import runpy, sys, types
numpy = types.ModuleType('numpy')
numpy.ndarray = type('ndarray', (), {})
numpy.asarray = lambda value: value
numpy.all = lambda value: value
numpy.array_equal = lambda left, right: True
numpy.uint32 = object
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
        fixture_path = Path(__file__).with_name("fixtures.py")
        cholesky_path = Path(__file__).resolve().parents[2] / "cholesky"
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
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
