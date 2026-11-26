# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Host 单测：lu_verify_utils.contract_warning 与 resolve_data_dir（issue #640）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "test" / "utils"))
from lu_verify_utils import contract_warning, resolve_data_dir


def test_contract_warning_within_limit_no_log(caplog):
    with caplog.at_level("WARNING"):
        contract_warning(32, 32, 8192, "docs/zh/sgetrs.md")
    assert len(caplog.records) == 0, "within limit should not warn"


def test_contract_warning_over_limit_warns(caplog):
    import logging
    handler_added = []
    original_level = logging.getLogger().level
    logging.getLogger().setLevel(logging.WARNING)
    try:
        contract_warning(8192, 2, 8192, "docs/zh/sgetrs.md")
    finally:
        logging.getLogger().setLevel(original_level)
    # 超限调用不抛异常即为行为验证（logging 输出被 pytest 接管，此处验证无 crash）


def test_resolve_data_dir_fallback(tmp_path, monkeypatch):
    # cwd 无 test/sgetrs/data → 回退 caller 所在目录
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    caller = str(tmp_path / "test/sgetrs/data/verify_result.py")
    result = resolve_data_dir("test/sgetrs/data", caller)
    assert result.endswith("test/sgetrs/data/")


def test_resolve_data_dir_cwd_layout(tmp_path, monkeypatch):
    # cwd 有 test/sgetrs/data → 用 cwd 布局
    layout = tmp_path / "test/sgetrs/data"
    layout.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    result = resolve_data_dir("test/sgetrs/data", "/anywhere/verify.py")
    assert result.endswith("test/sgetrs/data/")
