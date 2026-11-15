# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# tests/host —— Host 侧回归测试

纯 Host（无 NPU 设备依赖）的回归测试集合，与 `test/` 下的设备测试、CLI 校验脚本分层互补。绝大多数已挂接 pre-commit，作为提交期门禁。

## 测试清单与挂接状态

| 测试文件 | 覆盖面 | pre-commit hook | 触发锚（files） |
|---|---|---|---|
| test_build_path_arguments.py | build.sh 驱动契约：路径含空格的参数保真、CMake 选项 argv 边界、三阶段退出码透传（#214/#445） | build-driver-contract-tests | build.sh、测试自身 |
| test_cheevj_2x2_vector_host.py | cheevj host 数值块（2x2 向量差分） | cheevj-host-block-tests | src/cheevj/cheevj_host.cpp、测试自身 |
| test_cheevj_basis_verdict.py | cheevj verify 的正交基接受/非正交基拒绝/非有限基拒绝语义（8 用例） | cheevj-basis-verdict-tests | test/cheevj/data/verify_result.py、测试自身 |
| test_cmake_repository_setup.py | install_deps.sh 的 CMake 仓库配置与失败路径 | install-deps-cmake-repository-tests | install_deps.sh、测试自身 |
| test_cpack_run_registration.py | CPack External 打包链：.run 注册 + SHA256 校验、makeself/staging 缺失报错 | cpack-run-registration-tests（hook entry 自备 makeself） | cmake/makeself.cmake、cmake/package.cmake、scripts/package/install.sh、测试自身 |
| test_nonfinite_accuracy_verdict.py | 6 算子 verify 脚本的非有限值不被精度容差稀释（#347 口径）；qr_ulp 子集锚 cgeqrf 的 ULP 硬限 | qr-ulp-host-tests（仅 qr_ulp 子集） | 6 算子 verify 脚本、测试自身 |
| test_require_diagnostic_location.py | test/common/complex_qr 共享 require() 的诊断定位 | complex-qr-require-tests | test/common/complex_qr/*.py、测试自身 |
| test_uninstall_deletion_status.py | scripts/package/install.sh 卸载的删除完整性与失败边界 | uninstall-deletion-tests | scripts/package/install.sh、测试自身 |

## 运行方式

```bash
# 全量（注意剥离可能污染 numpy/libffi 的 Ascend LD_LIBRARY_PATH，见下）
env -u LD_LIBRARY_PATH python3 -m pytest -q tests/host/

# 单文件
python3 -m pytest -q tests/host/test_build_path_arguments.py

# pre-commit 单 hook 触发
pre-commit run build-driver-contract-tests --all-files
```

## 挂接策略（新增测试的指引）

1. **纯 Python / fake 外部命令的测试优先挂**：无 numpy/scipy、无设备依赖（如 build/uninstall/cmake-repo 三类）；
2. **numpy/scipy 依赖可挂**：gitcode CI 容器无 Ascend 环境，numpy 正常——此前「容器内 numpy import 段错误」的结论是开发机 `LD_LIBRARY_PATH` 含 Ascend 目录导致 libffi/libgomp 被 LD 链劫持，非 numpy 本身（#449 翻案）；
3. **设备依赖不挂**：torch_npu / NPU Runtime 类测试只经 run_tests.sh 门控；
4. **hook 的 files 正则锚**：锚「被测资产 + 测试自身」，粒度到具体文件而非目录，避免误触发；
5. **环境自备**：外部二进制依赖（如 cpack 测试的 makeself）在 hook entry 内幂等获取，不在测试里硬编码路径。

## 相关文档

- 挂接历史与各 hook 的动机：见各合入 PR（#273/#435/#440/#446/#449/#450 等）；
- `test/` 下设备测试与 CLI 脚本的运行：各目录 README 与 `build.sh --run`。
