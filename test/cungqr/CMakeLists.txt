# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# 本算子的精度回归以 test_cungqr.py CLI 形态经 run_tests.sh/run_package.sh 执行，
# 不产出 <op>_test 二进制：build.sh --run 依赖本 WARNING 文案把 op 记为 SKIPPED
# 而非 "Test binary not found" FAILED（issue #490）。WARNING 必须无条件发射
# （含 950），且不得 return()——下方 adapter include 还要在非 950 构造
# <op>_arch_mismatch 目标供 run_tests.sh 消费。
message(WARNING "cungqr tests require Ascend950, skipping on ${SOC_VERSION_LOWER}")

set(QR_ADAPTER_TARGET cungqr)
include(${CMAKE_CURRENT_LIST_DIR}/../complex_qr/adapter.cmake)
