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
| test_cpack_run_registration.py | CPack External 打包链：.run 注册 + SHA256 校验、makeself/staging 缺失报错 | cpack-run-registration-tests（entry 自备 makeself：下载先按 makeself-fetch.cmake 同一 SHA256 校验再解压，curl/tar 退出码显式检查，幂等判断双文件、不齐自动重下自愈——#486） | cmake/makeself.cmake、cmake/package.cmake、scripts/package/install.sh、测试自身 |
| test_nonfinite_accuracy_verdict.py | 9 算子 verify 脚本的非有限值不被精度容差稀释（#347 口径，含 sgetrs/cgetrs/cpotrf 判定用例）；qr_ulp 子集锚 cgeqrf 的 ULP 硬限 | qr-ulp-host-tests（ULP 子集）+ nonfinite-verdict-tests（全量，#484） | qr-ulp 锚 cgeqrf verify；nonfinite-verdict 锚 9 个 getrf/getri/getrs/cpotrf 族 verify 脚本与测试自身 |
| test_require_diagnostic_location.py | test/common/complex_qr 共享 require() 的诊断定位 | complex-qr-require-tests | test/common/complex_qr/*.py、测试自身 |
| test_uninstall_deletion_status.py | scripts/package/install.sh 卸载的删除完整性与失败边界 | uninstall-deletion-tests | scripts/package/install.sh、测试自身 |

> 注：test_run_cli / test_l1norm / test_contract_warning 三行暂时注释，待 !473（run_cli/l1norm 测试文件）与 !478（lu-verify-utils-tests hook）合入后 uncomment。

| test_read_binary_strict.py | read_binary_strict 的字节对齐 + expected_count 校验 | read-binary-strict-tests | test/utils/verify_utils.py、测试自身 |
| test_compare_and_report.py | compare_and_report 的 eps 双条件整体比对（#612 向量化语义） | — | — |
| test_solve_lu.py | forward/backward/solve_lu 的 LU 参考解计算 | — | — |
| test_load_pivots.py | load_pivots 的 1-based 主元校验 | — | — |
| test_verify_inverse.py | verify_inverse 的 A·A⁻¹ ≈ I 判定 | — | — |
<!-- | test_run_cli.py | run_cli 的退出码语义（True→0 / False/异常→1） | lu-verify-utils-tests | test/utils/lu_verify_utils.py、测试自身 | -->
<!-- | test_l1norm.py | l1norm 列绝对值和最大值 | lu-verify-utils-tests | test/utils/lu_verify_utils.py、测试自身 | -->
| test_contract_warning.py | contract_warning 超限提示 + resolve_data_dir 双布局 | lu-verify-utils-tests | test/utils/lu_verify_utils.py、测试自身 |

## 运行方式

```bash
# 全量（注意剥离可能污染 numpy/libffi 的 Ascend LD_LIBRARY_PATH，见下）
env -u LD_LIBRARY_PATH python3 -m pytest -q tests/host/

# 单文件
python3 -m pytest -q tests/host/test_build_path_arguments.py

# pre-commit 单 hook 触发
pre-commit run build-driver-contract-tests --all-files
```

## 环境前置

`tests/host/` 默认不依赖 NPU 设备，但以下两个用例需要 Host 侧工具链。未显式设置环境变量时，测试会自动探测本机 CANN 与 `PATH`；探测不到则以 `skip` 结束，不会把环境缺失误报为功能失败。显式设置但路径无效时仍会失败，并提示缺少的文件。

| 依赖 | 使用方 | 显式配置 | 自动探测 |
|---|---|---|---|
| `acl/acl.h` | `test_complex_divide_range.py` 使用真实 ACL ABI 类型编译 Host 回归 | `ACL_INCLUDE_PATH` 指向包含 `acl/acl.h` 的目录 | `ASCEND_HOME_PATH` / `ASCEND_TOOLKIT_HOME` 下的 `aarch64-linux/include`、`x86_64-linux/include`、`include`、`runtime/include`、`runtime/include/external`，以及 `/usr/local/Ascend/ascend-toolkit/latest`、`~/Ascend/ascend-toolkit/latest` 和 `cann-*` 版本目录 |
| makeself 2.5.x | `test_cpack_run_registration.py` 通过仓库 `cmake/makeself.cmake` 生成 `.run` | `SOLVER_MAKESELF_PATH` 指向包含 `makeself.sh` 的目录 | 仓库 `build/third_party/makeself`，以及 CANN 自带的 `.../tikcpp/ascendc_kernel_cmake/fwk_modules/util/makeself`、`.../op_project_tmpl/cmake/util/makeself` |

```bash
# 以本机 CANN 为例，路径按实际安装替换
export ACL_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/include
export SOLVER_MAKESELF_PATH=/usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/tikcpp/ascendc_kernel_cmake/fwk_modules/util/makeself
```

`cmake` / `cpack` 优先使用 `requirements.txt` 安装的 `cmake` 包，缺失时回退到 `PATH` 中同一目录下的二进制。`cpack-run-registration-tests` hook 会下载固定版本的 makeself 并自动设置 `SOLVER_MAKESELF_PATH`，因此提交门禁不依赖本机 CANN。

## 挂接策略（新增测试的指引）

1. **纯 Python / fake 外部命令的测试优先挂**：无 numpy/scipy、无设备依赖（如 build/uninstall/cmake-repo 三类）；
2. **numpy/scipy 依赖可挂**：gitcode CI 容器无 Ascend 环境，numpy 正常——此前「容器内 numpy import 段错误」的结论是开发机 `LD_LIBRARY_PATH` 含 Ascend 目录导致 libffi/libgomp 被 LD 链劫持，非 numpy 本身（#449 翻案）；
3. **设备依赖不挂**：torch_npu / NPU Runtime 类测试只经 run_tests.sh 门控；
4. **hook 的 files 正则锚**：锚「被测资产 + 测试自身」，粒度到具体文件而非目录，避免误触发；
5. **环境自备**：外部二进制依赖（如 cpack 测试的 makeself）在 hook entry 内幂等获取，不在测试里硬编码路径。

## 相关文档

- 挂接历史与各 hook 的动机：见各合入 PR（#273/#435/#440/#446/#449/#450 等）；
- `test/` 下设备测试与 CLI 脚本的运行：各目录 README 与 `build.sh --run`。

## conftest.py（共享 fixture）

`tests/host/conftest.py` 提供 Host 测试的共享 pytest fixture：

| fixture | 功能 | 使用方 |
|---|---|---|
| `toolkit_roots` | 自动探测 ACL 头文件与 makeself 位置（环境变量优先，无则 fallback 到常用安装路径） | test_cpack_run_registration.py 等 |

新增需要探测外部工具路径的 Host 测试时优先复用 `toolkit_roots`，不要在各测试文件内重复探测逻辑。

