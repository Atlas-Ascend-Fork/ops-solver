# CmatinvBatched

## 产品支持情况

本实现支持 Ascend950 系列，已在 Ascend950PR 验证。其他架构返回 `ACLSOLVER_STATUS_ARCH_MISMATCH`。

## 功能说明

直接对一批原始复数小方阵求逆，内部完成带主元 LU 分解，并写入独立输出矩阵。
支持 `0 ≤ n ≤ 32`；`n > 32` 返回参数错误，不转调 `CgetriBatched`。

## 函数原型

```cpp
aclsolverStatus_t aclsolverCmatinvBatched(
    aclsolverHandle_t handle,
    int n,
    const aclFloatComplex* const A[],
    int lda,
    aclFloatComplex* const Ainv[],
    int lda_inv,
    int* info,
    int batchSize);
```

## 参数说明

| 参数 | 位置与方向 | 说明 |
| --- | --- | --- |
| `handle` | Host，输入 | 已创建的 solver handle，使用其设置的 stream |
| `n` | Host，输入 | `0 ≤ n ≤ 32` |
| `A` | Device，输入 | 长度为 `batchSize` 的 Device 指针数组，每项指向一个原始列主序矩阵 |
| `lda` | Host，输入 | `lda ≥ max(1, n)`；每个输入矩阵至少 `lda * n` 个复数元素 |
| `Ainv` | Device，输出地址 | 长度为 `batchSize` 的 Device 指针数组，每项指向独立输出矩阵 |
| `lda_inv` | Host，输入 | `lda_inv ≥ max(1, n)`；每个输出矩阵至少 `lda_inv * n` 个复数元素 |
| `info` | Device，输出 | 长度为 `batchSize` 的 `int` 数组，每个矩阵分别写状态 |
| `batchSize` | Host，输入 | `0 ≤ batchSize ≤ 10000000` |

## 数据与状态约定

- 复数类型为 `aclFloatComplex`；矩阵按列主序存储，支持 leading dimension padding。
- 外层指针数组和数组内的每个矩阵都位于当前 Device。连续存储也需要构建 Device 指针数组。
- 输入和输出矩阵之间禁止重叠；各输出矩阵互不重叠。接口无显式工作空间和主元参数。
- 函数返回 `aclsolverStatus_t`。`info[b]` 为 `0`、首个零对角元素的正索引，或非法参数位置的负索引（不计 `handle`）。
- 当前实现同步 stream 回读指针元数据以校验地址与重叠，计算内核使用 handle 的 stream。
- 同步 stream 后读取逆矩阵和 Device `info`。返回成功不代表所有矩阵非奇异。
- `batchSize == 0` 时在标量参数校验后直接成功；`n == 0` 且 batch 非空时清零 `info`。

## 调用步骤

1. 按列主序准备原始矩阵并复制到 Device，分配独立 Device 输出。
2. 在 Host 组织输入、输出的 Device 地址列表，将两个列表复制到 Device 指针数组。
3. 分配 `batchSize` 个 Device `int` 作为 `info`，调用本接口。
4. 同步 stream，逐矩阵读取结果与状态。

完整调用与结果校验见[正式测试](../../test/cmatinv_batched/cmatinv_batched_test.cpp)。旧接口迁移见[迁移说明](complex_solver_migration.md)。
