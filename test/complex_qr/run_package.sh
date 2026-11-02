#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Run one operator's complete revised official precision and info matrix.
set -euo pipefail
op=${1:?operator required}
build_dir=${2:?build directory required}
case "$op" in cgeqrf|cungqr|cunmqr|ccgels) ;; *) exit 2 ;; esac
package_op=$op
if [ "$op" = ccgels ]; then package_op=cgels; fi
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cases_root=${COMPLEX_QR_CASES_ROOT:-}
results=${COMPLEX_QR_RESULTS:-$build_dir/package-results}
python_bin=${PYTHON:-python3}
adapter_dir="$build_dir/test/$op"
if [ ! -f "$adapter_dir/lib${op}_adapter.so" ]; then adapter_dir="$build_dir/test/complex_qr"; fi
export LD_LIBRARY_PATH="$build_dir:${LD_LIBRARY_PATH:-}"
if [ -x "$build_dir/test/$op/${op}_arch_mismatch" ]; then
    exec "$build_dir/test/$op/${op}_arch_mismatch"
fi
if [ -z "$cases_root" ]; then
    exec "$python_bin" "$test_dir/../$op/test_${op}.py" --library "$build_dir/libops_solver.so" --device "${NPU_DEVICE_ID:-0}"
fi
"$python_bin" "$test_dir/package_accuracy.py" --package "$cases_root/$package_op/package" \
    --library "$build_dir/libops_solver.so" --device "${NPU_DEVICE_ID:-0}" \
    --out "$results/$package_op/dut" --perf-inputs "$results/inputs" --adapters "$adapter_dir"
"$python_bin" "$cases_root/$package_op/package/verify_accuracy.py" \
    --package "$cases_root/$package_op/package" --dut-out "$results/$package_op/dut" \
    --report "$results/$package_op/accuracy.json"
