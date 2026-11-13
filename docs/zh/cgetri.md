# Cgetri

## 产品支持情况

本实现支持 Ascend950 系列，已在 Ascend950PR 验证。其他架构返回 `ACLSOLVER_STATUS_ARCH_MISMATCH`。

## 功能说明

输入已完成 LU 分解的 `n × n` 复数矩阵及其主元，原地输出原矩阵的逆。
本接口消费 `aclsolverCgetrf` 的结果；输入原始矩阵时，应先调用 `aclsolverCgetrf`。

## 函数原型

```cpp
aclsolverStatus_t aclsolverCgetri_bufferSize(
    aclsolverHandle_t handle,
    int n,
    aclFloatComplex* A,
    int lda,
    int* Lwork);

aclsolverStatus_t aclsolverCgetri(
    aclsolverHandle_t handle,
    int n,
    aclFloatComplex* A,
    int lda,
    int* devIpiv,
    aclFloatComplex* Workspace,
    int Lwork,
    int* devInfo);
```

## 参数说明

| 参数 | 位置与方向 | 说明 |
| --- | --- | --- |
| `handle` | Host，输入 | 已创建的 solver handle，使用其设置的 stream |
| `n` | Host，输入 | `0 ≤ n ≤ 4096`；超限时两个入口均返回 `ACLSOLVER_STATUS_INVALID_VALUE`（执行入口另写 `devInfo` 置 -1，查询入口不写状态缓冲） |
| `A` | Device，输入/输出 | 输入为列主序 packed LU，输出为逆矩阵；至少 `lda * n` 个复数元素 |
| `lda` | Host，输入 | `lda ≥ max(1, n)`，支持列尾 padding |
| `devIpiv` | Device，输入 | `n` 个 1-based 行交换索引；`nullptr` 仅用于无主元 LU |
| `Lwork` | Host，查询输出/计算输入 | 复数元素数，查询结果为 `n * n`；计算时至少为该值 |
| `Workspace` | Device，计算输入 | 至少 `Lwork` 个复数元素，不与 `A` 重叠 |
| `devInfo` | Device，输出 | 1 个 `int`；非空且容量有效 |

## 数据与状态约定

- 数据类型为 `aclFloatComplex`，包含连续的 `float real`、`float imag`。
- 矩阵按列主序存储，元素 `(row, col)` 位于 `A[col * lda + row]`。`lda` 以元素为单位。
- 矩阵、工作空间、主元和 `info` 位于当前 Device。查询接口的 `Lwork` 是 Host 输出。
- 函数返回 `aclsolverStatus_t`，成功值为 `ACLSOLVER_STATUS_SUCCESS`；参数非法返回 `ACLSOLVER_STATUS_INVALID_VALUE`。
- 有效的 Device `info` 写入 `0` 表示成功，正数表示首个零对角元素的 1-based 索引，负数表示非法参数位置（不计 `handle`）。无效的 `info` 缓冲区无法接收状态。
- 数值结果和 Device `info` 应在调用方 stream 同步后读取。返回成功不代表矩阵非奇异。
- 旧 Host 接口的迁移步骤见[复数求解器接口迁移](complex_solver_migration.md)。

## 调用与边界

- 先完成 LU 分解，再查询并分配求逆工作空间，随后调用本接口。
- 有效主元在第 `k` 列满足 `k + 1 ≤ devIpiv[k] ≤ n`（`k` 从 0 开始）。非法主元写 `devInfo = -4`，函数返回 `ACLSOLVER_STATUS_INVALID_VALUE`。
- 主元非空时，当前实现会同步调用方 stream 并回读 `info`，以便通过返回值报告非法主元。
- `n == 0` 时不计算矩阵，写 `devInfo = 0`；仍需满足 `lda`、非负 `Lwork` 和 `devInfo` 约束。

完整调用与结果校验见[正式测试](../../test/cgetri/cgetri_test.cpp)；连续调用示例见[迁移说明](complex_solver_migration.md#单矩阵调用示例)。
