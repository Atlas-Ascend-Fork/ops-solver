# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：lu_verify_utils.run_cli 的退出码语义（issue #642）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from lu_verify_utils import run_cli


def test_run_cli_pass_returns_zero():
    def main_pass():
        return True
    result = run_cli(main_pass)
    assert result == 0


def test_run_cli_fail_returns_one():
    def main_fail():
        return False
    result = run_cli(main_fail)
    assert result == 1


def test_run_cli_exception_returns_one():
    def main_raises():
        raise ValueError("test exception")
    result = run_cli(main_raises)
    assert result == 1


def test_run_cli_none_returns_one():
    def main_none():
        return None  # falsy
    result = run_cli(main_none)
    assert result == 1
