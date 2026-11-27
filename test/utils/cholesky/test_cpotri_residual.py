# Copyright (c) 2026 Xztxzt.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Host regression for the complex POTRI inverse residual."""

import numpy as np
import pytest

from fixtures import lapack_ratio, verify_residuals


@pytest.mark.parametrize("upper", [False, True])
def test_inverse_residual_keeps_imaginary_diagonal(upper):
    identity = np.eye(2, dtype=np.complex64)[None]
    cpu = lapack_ratio("cpotri", identity, identity, None, upper)
    result = identity.copy()
    result[0, 0, 0] += 2j
    ratio = lapack_ratio("cpotri", identity, result, None, upper)

    assert cpu[0] == 0
    with pytest.raises(RuntimeError, match="LAPACK residual"):
        verify_residuals("cpotri", [{"case": 0, "cpu_ratio": cpu, "ratio": ratio}])
