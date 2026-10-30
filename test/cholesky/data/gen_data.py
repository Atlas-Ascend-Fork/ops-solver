# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Generate data for the solver build.sh three-stage sample."""

import logging
import sys
from pathlib import Path
import shutil
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
root = Path(__file__).resolve().parent
output = root / "output"
if output.exists():
    shutil.rmtree(output)
output.mkdir()
(root / "input").mkdir(exist_ok=True)
n, ld = 17, 20
a = np.full((n, ld), -12345.125, dtype=np.float32)
a[:, :n] = np.eye(n, dtype=np.float32) * 3 + np.float32(0.1)
b = np.full(ld, -12345.125, dtype=np.float32)
b[:n] = 1
a.tofile(root / "input/a.bin")
b.tofile(root / "input/b.bin")
logging.info(
    "Generated n=17, lda=20; LOWER/UPPER; five C APIs and physical batch slots"
)
