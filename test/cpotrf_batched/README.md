# cpotrf_batched 样例（Ascend 950 C API）

## 算子描述

- 本样例驱动 [aclsolverCpotrfBatched](../../docs/zh/cpotrf_batched.md)（Device 指针数组逐矩阵 Cholesky 分解），按 fixtures 清单逐用例完成指针数组装载、LOWER/UPPER 分解与逐位确定性检查。
- 矩阵为 COMPLEX64、Device 列主序；`n ∈ [0, 4096]`，`batchSize ∈ [0, 1000000]`，`lda >= max(1, n)`；`infoArray` 逐 batch 写入状态。完整契约见 [cpotrf_batched 接口说明](../../docs/zh/cpotrf_batched.md)。

## 目录结构介绍

```text
cpotrf_batched/
├── CMakeLists.txt           # 测试目标构建配置
├── cpotrf_batched_test.cpp  # CholeskyTest::Run 薄入口（fixture_runner 驱动全部用例）
└── data/
    ├── gen_data.py          # 生成 fixtures 用例（input/manifest.tsv 与逐用例输入/golden）
    └── verify_result.py     # fixtures.verify：LAPACK 残差两步判定
```

## 编译运行

```bash
bash build.sh --ops=cpotrf_batched --soc=ascend950 --run
```

运行流程：`gen_data.py` 生成用例 → `cpotrf_batched_test` 执行 NPU 调用与逐位确定性检查 → `verify_result.py` 按 LAPACK 残差口径校验。直接运行测试二进制前须先执行数据生成，并清理上一次的 input/output 产物。

- 完整数值/契约门禁与可配置维数、batch、padding 的通用测试见 [test/cholesky](../cholesky/README.md)。
