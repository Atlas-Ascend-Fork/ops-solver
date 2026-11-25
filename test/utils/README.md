# test/utils —— 共享测试工具

Host 侧共享 Python/C++ 工具，被各算子测试的 `data/gen_data.py` 与 `data/verify_result.py` 消费。

## Python 工具

| 文件 | 核心接口 | 消费方 |
|---|---|---|
| verify_utils.py | `read_binary_strict(path, dtype, expected_count=None)`——字节对齐 + 元素数双重校验 | sgetrf/cgetrf/sgetri/cgetri/cgetri_batched/cmatinv_batched/cheevj/cgeqrf/cholesky/spotrf/cpotrf |
| verify_utils.py | `verify_inverse(input, inv, order, dtype, mismatch)`——A·A⁻¹ ≈ I 判定 | sgetri/cgetri |
| lu_verify_utils.py | `solve_lu/forward_substitution/backward_substitution`——LU 参考解计算（#612 向量化） | sgetrs/cgetrs |
| lu_verify_utils.py | `load_pivots(path, n)`——1-based 主元校验+转换 | sgetrs/cgetrs |
| lu_verify_utils.py | `compare_and_report(x_ref, x, n, nrhs)`——eps 双条件整体比对 | sgetrs/cgetrs |

## C++ 工具

| 文件 | 功能 | 消费方 |
|---|---|---|
| test_utils.h | ReadFile/WriteFile 设备文件 IO | cgeqrf/cgetrs/sgetrs 等 <op>_test.cpp |
| qr_test_utils.h | QR 测试共用断言 | cheevj 等 |
| cholesky/fixture_runner.cpp | Cholesky 五接口共享调用与设备管理 | cholesky/cpotrf_950 等 |

## sys.path 约定

消费方（`test/<op>/data/*.py`）通过 `sys.path.insert` 注入 `test/utils` 目录后 `from verify_utils import ...`。新建 verify 脚本请遵循同款注入（参照 `test/sgetrf/data/verify_result.py:17-20`）。

## verify_utils 的 expected_count

`read_binary_strict` 的 `expected_count` 参数（issue #598）：传入期望元素数后，截断/追加文件在读取点报精确差异（`expected N elements, got M`）——主要 verify 脚本已接入（sgetrf/cgetrf/sgetri/cgetri/cgetri_batched/cmatinv_batched/cheevj/cgeqrf/cholesky/spotrf/cpotrf 及 fixtures），部分直接调用 `read_binary_strict` 的新脚本仍可省略该参数（向后兼容，默认不校验）。
