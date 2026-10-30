# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Independent checks for acceptance arithmetic and fail-closed batch coverage."""

import unittest
import numpy as np
from lapack_residual import EPS, evaluate, finalize, residual


class ResidualTests(unittest.TestCase):
    def test_upper_factor_order_and_both_storage_sides(self):
        a = np.array([[4, 2], [2, 10]], dtype=np.float32)
        u = np.array([[2, 1], [0, 3]], dtype=np.float32)
        self.assertEqual(residual("spotrf", a, u, True), 0)
        self.assertEqual(residual("spotrf", a, u.T, False), 0)
        self.assertGreater(residual("spotrf", a, u.T, True), 1)

    def test_vector_one_norm_multiple_rhs(self):
        a = np.diag(np.array([2, 4], dtype=np.float32))
        x = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        b = a @ x
        b[0, 1] += np.float32(2**-20)
        expected = (2**-20) / (4 * 4 * EPS)
        self.assertEqual(residual("spotrs", a, x, b32=b), expected)

    def test_inverse_uses_original_matrix_and_mirrors_stored_triangle(self):
        a = np.array([[4, 2], [2, 10]], dtype=np.float32)
        c = np.array([[10, -2], [-2, 4]], dtype=np.float64) / 36
        self.assertLess(residual("spotri", a, np.triu(c), True), 1e-8)
        self.assertGreater(residual("spotri", a, np.eye(2)), 1)

    def test_residual_is_final_even_when_elementwise_passed(self):
        rows = [
            {
                "purpose": "accuracy",
                "task_layer1_pass": True,
                "lapack_residual": [{"ratio": 100.0, "ratio_cpu": 0.1}],
            }
        ]
        finalize(rows, "spotrf", True)
        self.assertFalse(rows[0]["final_precision_pass"])
        rows[0]["task_layer1_pass"] = False
        rows[0]["lapack_residual"][0]["ratio"] = 0.2
        finalize(rows, "spotrf", True)
        self.assertTrue(rows[0]["final_precision_pass"])

    def test_batch_checks_nonrepresentative_physical_slots(self):
        a = np.array([np.diag([4, 9]), np.diag([16, 36])], dtype=np.float32)
        out = np.array(
            [np.diag([2, 3]), np.diag([4, 6]), np.diag([2, 3])], dtype=np.float32
        )
        arrays = {
            "A32": a,
            "sample_map": [
                {"content_idx": 0, "rep_slot": 0, "slots": [0, 2]},
                {"content_idx": 1, "rep_slot": 1, "slots": [1]},
            ],
        }
        entry = {"op": "spotrfBatched", "uplo": "L", "batch": 3}
        dut = {"out32": out, "info": np.zeros(3, dtype=np.int32)}
        self.assertEqual(
            sum(r["physical_slots_checked"] for r in evaluate(entry, arrays, dut)), 3
        )
        out[2, 0, 0] = 5
        with self.assertRaisesRegex(ValueError, "physical slots differ"):
            evaluate(entry, arrays, dut)

    def test_nonfinite_output_is_rejected(self):
        with self.assertRaises(ValueError):
            residual("spotrf", np.eye(2), np.full((2, 2), np.nan))


if __name__ == "__main__":
    unittest.main()
