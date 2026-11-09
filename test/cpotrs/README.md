# cpotrs 样例（Ascend 950 C API）

## 算子描述

- 本样例驱动 [aclsolverCpotrs](../../docs/zh/cpotrs.md)（使用已有 Cholesky 因子求解 A·X = B，原地覆盖右端项），按 fixtures 清单逐用例完成因子与 RHS 装载、LOWER/UPPER 求解与逐位确定性检查。
- 输入须为 aclsolverCpotrf 输出的因子；矩阵为 COMPLEX64、Device 列主序；`n ∈ [0, 4096]`，`nrhs ∈ [0, 128]`，`lda/ldb >= max(1, n)`；`devInfo` 逐次写入。完整契约见 [cpotrs 接口说明](../../docs/zh/cpotrs.md)。

## 目录结构介绍

```text
cpotrs/
├── CMakeLists.txt  # 测试目标构建配置
├── cpotrs_test.cpp # CholeskyTest::Run 薄入口（fixture_runner 驱动全部用例）
└── data/
    ├── gen_data.py      # 生成 fixtures 用例（输入因子、RHS 与 golden）
    └── verify_result.py # fixtures.verify：LAPACK 残差两步判定
```

## 编译运行

```bash
bash build.sh --ops=cpotrs --soc=ascend950 --run
```

运行流程：`gen_data.py` 生成用例 → `cpotrs_test` 执行 NPU 调用与逐位确定性检查 → `verify_result.py` 按 LAPACK 残差口径校验。直接运行测试二进制前须先执行数据生成，并清理上一次的 input/output 产物。

- 完整数值/契约门禁与可配置维数、batch、RHS、padding 的通用测试见 [test/cholesky](../cholesky/README.md)。
