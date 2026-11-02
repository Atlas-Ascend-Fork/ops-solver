# CgetriBatched

## 产品支持情况

本实现支持 Ascend950 系列，已在 Ascend950PR 验证。其他架构返回 `ACLSOLVER_STATUS_ARCH_MISMATCH`。

## 功能说明

对一批已完成 LU 分解的复数方阵求逆，读取对应主元并写入独立输出矩阵。
本接口支持小矩阵和大矩阵，输入是 LU 因子，不是原始矩阵。

## 函数原型

```cpp
aclsolverStatus_t aclsolverCgetriBatched(
    aclsolverHandle_t handle,
    int n,
    const aclFloatComplex* const Aarray[],
    int lda,
    const int* PivotArray,
    aclFloatComplex* const Carray[],
    int ldc,
    int* infoArray,
    int batchSize);
```

## 参数说明

| 参数 | 位置与方向 | 说明 |
| --- | --- | --- |
| `handle` | Host，输入 | 已创建的 solver handle，使用其设置的 stream |
| `n` | Host，输入 | `0 ≤ n ≤ 4096` |
| `Aarray` | Device，输入 | 长度为 `batchSize` 的 Device 指针数组，每项指向列主序 LU 矩阵 |
| `lda` | Host，输入 | `lda ≥ max(1, n)`；每个输入矩阵至少 `lda * n` 个复数元素 |
| `PivotArray` | Device，输入 | 连续的 `n * batchSize` 个 `int`，第 `b` 组从 `PivotArray[b * n]` 开始；`nullptr` 表示所有矩阵均为无主元 LU |
| `Carray` | Device，输出地址 | 长度为 `batchSize` 的 Device 指针数组，每项指向独立输出矩阵 |
| `ldc` | Host，输入 | `ldc ≥ max(1, n)`；每个输出矩阵至少 `ldc * n` 个复数元素 |
| `infoArray` | Device，输出 | 长度为 `batchSize` 的 `int` 数组，每个矩阵分别写状态 |
| `batchSize` | Host，输入 | `0 ≤ batchSize ≤ 1000000` |

## 数据与状态约定

- 复数类型为 `aclFloatComplex`，矩阵按列主序存储，支持 leading dimension padding。
- 外层指针数组和数组内的每个矩阵都位于当前 Device；Host 指针数组需要先复制到 Device。
- 输入和输出矩阵之间禁止重叠；各输出矩阵互不重叠。接口无显式工作空间参数。
- 函数返回 `aclsolverStatus_t`。`infoArray[b]` 为 `0`、首个零对角元素的正索引，或非法参数位置的负索引（不计 `handle`）。
- 第 `k` 列主元满足 `k + 1 ≤ PivotArray[b * n + k] ≤ n`。非法主元对应 `infoArray[b] = -4`，函数返回 `ACLSOLVER_STATUS_INVALID_VALUE`；其他矩阵保留各自状态。
- 当前实现同步 stream 回读指针元数据以校验地址与重叠；主元非空时还会同步回读 `info`。
- 数值结果和 Device `infoArray` 在调用方 stream 同步后读取。返回成功不代表所有矩阵非奇异。
- `batchSize == 0` 时在标量参数校验后直接成功；`n == 0` 且 batch 非空时清零 `infoArray`。

## 调用步骤

1. 对每个原始矩阵执行 LU 分解，保存对应的主元。
2. 在 Host 组织输入、输出的 Device 地址列表，将两个列表复制到 Device 指针数组。
3. 将各矩阵的主元按 batch 顺序存入连续 Device `PivotArray`。
4. 调用本接口，同步 stream 后读取输出矩阵和逐矩阵状态。

完整调用与结果校验见[正式测试](../../test/cgetri_batched/cgetri_batched_test.cpp)。旧接口迁移见[迁移说明](complex_solver_migration.md)。
