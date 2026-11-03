# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""require() diagnostics must point at the caller, never at a stale file:line."""

from pathlib import Path
import inspect
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "test" / "common" / "complex_qr"))

from test_support import require  # noqa: E402


def test_require_prefixes_the_caller_location():
    source_lines, start_line = inspect.getsourcelines(
        test_require_prefixes_the_caller_location
    )
    call_line = start_line + next(
        index
        for index, line in enumerate(source_lines)
        if "require(False, \"diagnostic message\")" in line
    )
    with pytest.raises(AssertionError) as caught:
        require(False, "diagnostic message")
    expected_prefix = "{}:{}: ".format(Path(__file__).name, call_line)
    assert str(caught.value).startswith(expected_prefix)
    assert str(caught.value).endswith("diagnostic message")


def test_require_keeps_checks_active_under_optimization():
    # require() must not use assert, so python -O cannot silence a failure.
    with pytest.raises(AssertionError):
        require(0, "unreachable guard")


def test_require_accepts_non_string_messages():
    with pytest.raises(AssertionError) as caught:
        require(False, 3e-6)
    assert str(caught.value).endswith(": 3e-06")
