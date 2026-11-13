# Cgetrf

## 产品支持情况

本实现支持 Ascend950 系列，已在 Ascend950PR 验证。其他架构返回 `ACLSOLVER_STATUS_ARCH_MISMATCH`。

## 功能说明

对 `m × n` 复数矩阵执行原地 LU 分解。输出将 L 的严格下三角和 U 合并存储，L 的对角线隐含为 1。
主元记录按列顺序执行的行交换；依次应用这些交换得到 `P * A = L * U`。
部分主元选择使用 `abs(real) + abs(imag)`，相等时选择行号较小者。

## 函数原型

```cpp
aclsolverStatus_t aclsolverCgetrf_bufferSize(
    aclsolverHandle_t handle,
    int m,
    int n,
    aclFloatComplex* A,
    int lda,
    int* Lwork);

aclsolverStatus_t aclsolverCgetrf(
    aclsolverHandle_t handle,
    int m,
    int n,
    aclFloatComplex* A,
    int lda,
    aclFloatComplex* Workspace,
    int* devIpiv,
    int* devInfo);
```

## 参数说明

| 参数 | 位置与方向 | 说明 |
| --- | --- | --- |
| `handle` | Host，输入 | 已创建的 solver handle，使用其设置的 stream |
| `m`、`n` | Host，输入 | `0 ≤ m ≤ 4096`，`0 ≤ n ≤ 8192`；超限时两个入口均返回 `ACLSOLVER_STATUS_INVALID_VALUE`（执行入口另写 `devInfo` 置 -1/-2，查询入口不写状态缓冲） |
| `A` | Device，输入/输出 | 列主序矩阵，分配至少 `lda * n` 个复数元素 |
| `lda` | Host，输入 | `lda ≥ max(1, m)`，支持列尾 padding |
| `Lwork` | Host，查询输出 | 工作空间的复数元素数；分配字节数为 `Lwork * sizeof(aclFloatComplex)` |
| `Workspace` | Device，计算输入 | 按查询结果分配；查询结果为 0 时可传 `nullptr` |
| `devIpiv` | Device，输出 | 至少 `min(m, n)` 个 `int`，1-based 行交换索引；`nullptr` 表示无主元分解 |
| `devInfo` | Device，输出 | 1 个 `int`；非空且容量有效 |

## 数据与状态约定

- 数据类型为 `aclFloatComplex`，包含连续的 `float real`、`float imag`。
- 矩阵按列主序存储，元素 `(row, col)` 位于 `A[col * lda + row]`。`lda` 以元素为单位。
- 矩阵、工作空间、主元和 `info` 位于当前 Device。查询接口的 `Lwork` 是 Host 输出。
- 函数返回 `aclsolverStatus_t`，成功值为 `ACLSOLVER_STATUS_SUCCESS`；参数非法返回 `ACLSOLVER_STATUS_INVALID_VALUE`。
- 有效的 Device `info` 写入 `0` 表示成功，正数表示首个零对角元素的 1-based 索引，负数表示非法参数位置（不计 `handle`）。无效的 `info` 缓冲区无法接收状态。
- 数值结果和 Device `info` 应在调用方 stream 同步后读取。返回成功不代表矩阵非奇异。
- 旧 Host 接口的迁移步骤见[复数求解器接口迁移](complex_solver_migration.md)。

## 调用步骤

1. 分配 Device 矩阵、主元和 `devInfo`，将 Host 输入转换为列主序并复制到 Device。
2. 调用 `aclsolverCgetrf_bufferSize`，分配对应的 Device 工作空间。
3. 调用 `aclsolverCgetrf`，同步 stream 后读取 LU、主元和 `devInfo`。
4. `m == 0` 或 `n == 0` 时不计算矩阵，写 `devInfo = 0`；仍需满足 `lda` 和 `devInfo` 约束。

完整调用与结果校验见[正式测试](../../test/cgetrf/cgetrf_test.cpp)和[共享测试实现](../../test/utils/complex_solver_test.h)。
