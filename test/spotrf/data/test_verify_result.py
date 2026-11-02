# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host-only regression for the sample data and verifier."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class SpotrfSampleVerificationTest(unittest.TestCase):
    def test_custom_size_and_both_triangle_modes(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gen_data.py", "verify_result.py"):
                shutil.copyfile(source / name, root / name)
            subprocess.run([sys.executable, str(root / "gen_data.py"), "--n", "7"], check=True)
            matrix = np.fromfile(root / "input/A_gm.bin", dtype=np.float32).reshape(7, 7)
            lower = np.linalg.cholesky(matrix).astype(np.float32)
            output = root / "output/A_gm.bin"
            for mode, factor in (("L", lower.T), ("U", lower)):
                with self.subTest(mode=mode):
                    factor.tofile(output)
                    command = [sys.executable, str(root / "verify_result.py"), "--n", "7", "--uplo", mode]
                    self.assertEqual(subprocess.run(command).returncode, 0)
                    wrong = factor.copy()
                    wrong[0, 0] += 0.5  # still finite and positive, but no longer reconstructs A
                    wrong.tofile(output)
                    self.assertNotEqual(subprocess.run(command).returncode, 0)

    def test_correct_and_corrupt_results(self):
        source = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("gen_data.py", "verify_result.py"):
                shutil.copyfile(source / name, root / name)
            subprocess.run([sys.executable, str(root / "gen_data.py")], check=True)
            matrix = np.fromfile(root / "input/A_gm.bin", dtype=np.float32).reshape(
                255, 255
            )
            upper = np.linalg.cholesky(matrix).T.astype(np.float32)
            output = root / "output/A_gm.bin"
            for label, value, expected in (
                ("correct", upper, 0),
                ("wrong", np.zeros_like(upper), 1),
                ("negative diagonal", -upper, 1),
                ("nonfinite", np.full_like(upper, np.nan), 1),
            ):
                with self.subTest(case=label):
                    value.tofile(output)
                    result = subprocess.run(
                        [sys.executable, str(root / "verify_result.py")]
                    )
                    self.assertEqual(result.returncode, expected)
            output.write_bytes(b"bad")
            result = subprocess.run([sys.executable, str(root / "verify_result.py")])
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
