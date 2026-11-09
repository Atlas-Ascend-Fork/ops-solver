# cpotrf_950 样例（Ascend 950 C API）

## 算子描述

- 本样例驱动 [aclsolverCpotrf](../../docs/zh/cpotrf.md)（Ascend 950 复数 Hermitian 正定矩阵的 Cholesky 分解，含 `aclsolverCpotrf_bufferSize` workspace 查询），按 fixtures 清单逐用例完成 bufferSize 查询、Device 列主序矩阵装载、LOWER/UPPER 双向分解与逐位确定性检查。
- 矩阵为 COMPLEX64、Device 列主序；`n ∈ [0, 4096]`，`lda >= max(1, n)`，padding 保留；`devInfo` 逐次写入（0 成功、负值参数非法、正值非正定位置）。完整契约见 [cpotrf 接口说明](../../docs/zh/cpotrf.md)。

## 目录结构介绍

```text
cpotrf_950/
├── CMakeLists.txt      # 测试目标构建配置
├── cpotrf_950_test.cpp # CholeskyTest::Run 薄入口（fixture_runner 驱动全部用例）
└── data/
    ├── gen_data.py      # 生成 fixtures 用例（input/manifest.tsv 与逐用例输入/golden）
    └── verify_result.py # fixtures.verify：LAPACK 残差两步判定
```

## 编译运行

```bash
bash build.sh --ops=cpotrf_950 --soc=ascend950 --run
```

运行流程：`gen_data.py` 生成用例 → `cpotrf_950_test` 执行 NPU 调用与逐位确定性检查 → `verify_result.py` 按 [LAPACK 残差口径](../../docs/zh/cpotrf.md) 校验。直接运行测试二进制前须先执行数据生成，并清理上一次的 input/output 产物。

- 完整数值/契约门禁与可配置维数、batch、padding 的通用测试见 [test/cholesky](../cholesky/README.md)。
- 非 950 SoC 上本测试按仓内构建约定跳过或报架构不支持。
