# cpotrs_batched 样例（Ascend 950 C API）

## 算子描述

- 本样例驱动 [aclsolverCpotrsBatched](../../docs/zh/cpotrs_batched.md)（Device 指针数组批量单 RHS 求解），按 fixtures 清单逐用例完成指针数组装载、LOWER/UPPER 求解与逐位确定性检查。
- 输入须为 aclsolverCpotrfBatched 输出的因子；矩阵为 COMPLEX64、Device 列主序；`n ∈ [0, 4096]`，`batchSize ∈ [0, 1000000]`；`infoArray` 逐 batch 写入状态。完整契约见 [cpotrs_batched 接口说明](../../docs/zh/cpotrs_batched.md)。

## 目录结构介绍

```text
cpotrs_batched/
├── CMakeLists.txt          # 测试目标构建配置
├── cpotrs_batched_test.cpp # CholeskyTest::Run 薄入口（fixture_runner 驱动全部用例）
└── data/
    ├── gen_data.py          # 生成 fixtures 用例（输入因子、RHS 与 golden）
    └── verify_result.py     # fixtures.verify：LAPACK 残差两步判定
```

## 编译运行

```bash
bash build.sh --ops=cpotrs_batched --soc=ascend950 --run
```

运行流程：`gen_data.py` 生成用例 → `cpotrs_batched_test` 执行 NPU 调用与逐位确定性检查 → `verify_result.py` 按 LAPACK 残差口径校验。直接运行测试二进制前须先执行数据生成，并清理上一次的 input/output 产物。

- 完整数值/契约门禁与可配置维数、batch、RHS、padding 的通用测试见 [test/cholesky](../cholesky/README.md)。
