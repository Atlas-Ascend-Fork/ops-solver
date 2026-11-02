#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -euo pipefail

test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:?Usage: run_tests.sh BUILD_DIR}
device=${NPU_DEVICE_ID:-0}
python_bin=${PYTHON:-python3}
library="${build_dir}/libops_solver.so"
export LD_LIBRARY_PATH="${build_dir}:${LD_LIBRARY_PATH:-}"

if [ -x "${build_dir}/test/complex_qr/complex_qr_arch_mismatch" ]; then
    exec "${build_dir}/test/complex_qr/complex_qr_arch_mismatch"
fi

if [ ! -x "${build_dir}/test/complex_qr/complex_qr_public_abi" ]; then
    echo "Complex QR tests require SOC_VERSION=ascend950 and BUILD_TEST=ON."
    exit 1
fi
"${build_dir}/test/complex_qr/complex_qr_public_abi"
"${python_bin}" "${test_dir}/contracts.py" --library "${library}" --device "${device}"
"${python_bin}" "${test_dir}/smoke.py" --library "${library}" --device "${device}"
"${python_bin}" "${test_dir}/blocked_apply_smoke.py" --library "${library}" --device "${device}" --task-boundaries
"${python_bin}" "${test_dir}/validation_coverage.py" --library "${library}" --device "${device}"
"${python_bin}" "${test_dir}/memory_bounds.py" --library "${library}" --device "${device}"
echo "Complex QR supplementary suite passed. Follow REVISED_PACKAGE.md for the official package_accuracy.py / verify_accuracy.py matrix and original msprof op performance measurements."
