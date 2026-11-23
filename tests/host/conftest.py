# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared fixtures for Host-side tests."""

import os
from pathlib import Path

import pytest


@pytest.fixture
def toolkit_roots():
    """Return candidate CANN toolkit roots, most explicit first."""
    roots = []
    for variable in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value))
    roots.extend(
        [
            Path("/usr/local/Ascend/ascend-toolkit/latest"),
            Path.home() / "Ascend/ascend-toolkit/latest",
        ]
    )
    for parent in (Path("/usr/local/Ascend"), Path.home() / "Ascend"):
        roots.extend(sorted(parent.glob("cann-*")))
    return roots
