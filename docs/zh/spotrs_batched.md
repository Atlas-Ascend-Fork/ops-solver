# SpotrsBatched

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Ascend 950PR | √ |
| Ascend 950DT | √ |

## 功能说明

`aclsolverSpotrsBatched` 使用批量 Cholesky 因子求解单 RHS 线性方程组：

$$
A_{0,b}x_b=b_b,\qquad A_{0,b}=L_b L_b^T\ \text{或}\ U_b^T U_b.
$$

`Aarray` 中的矩阵为 [SpotrfBatched](spotrf_batched.md) 等接口成功分解后的因子，保持只读；
`Barray` 中的右端项原地覆盖为解。矩阵和向量均可分别分配。

## 函数原型

```cpp
aclsolverStatus_t aclsolverSpotrsBatched(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, int nrhs, float *Aarray[], int lda,
    float *Barray[], int ldb, int *info, int batchSize);
```

## 参数说明

| 参数 | 输入/输出 | 内存位置及说明 |
|:---|:---:|:---|
| `handle` | 输入 | 由 `aclsolverCreate` 创建并关联计算流的句柄 |
| `uplo` | 输入 | 各因子所在三角，须与分解时一致 |
| `n` | 输入 | 矩阵阶数，`0 <= n <= 4096` |
| `nrhs` | 输入 | 有效计算仅支持 `nrhs=1` |
| `Aarray` | 输入 | Device 指针数组，每个 Device FLOAT32 因子按列主序存放至少 `lda*n` 个元素 |
| `lda` | 输入 | 因子列跨度（float 元素数），`lda >= max(1,n)` |
| `Barray` | 输入/输出 | Device 指针数组，每个槽位指向至少 `ldb` 个 float 的 Device 缓冲区；前 n 个元素从 RHS 覆盖为解 |
| `ldb` | 输入 | RHS 列跨度（float 元素数），`ldb >= max(1,n)` |
| `info` | 输出 | Device `int*`，一个标量；与 SpotrfBatched 的逐矩阵 infoArray 不同 |
| `batchSize` | 输入 | 非负 `int`，可执行规模受实际内存资源约束 |

## 返回值

接口返回值见[公共约定](cholesky.md#接口公共约定)。
`info=0` 表示参数通过检查；负值表示首个非法参数，按
`uplo,n,nrhs,Aarray,lda,Barray,ldb,info,batchSize` 从 1 开始编号。
Device 数组内任一 A 槽位为空时 info=-4，否则任一 B 槽位为空时 info=-6；
此时不修改 RHS。该异步检查不改变 info 的标量约定，不能据此返回矩阵索引。
本接口不重新判断因子正定性或检测零对角，应先逐个确认分解的 infoArray 均为 0。

## 算子约束

- 仅支持 FLOAT32，有效计算（`n>0 && batchSize>0`）仅支持单 RHS，`nrhs=0` 也非法。
- 空问题（`n=0` 或 `batchSize=0`）允许非负 nrhs、空 Aarray/Barray；
  仍须满足 `lda/ldb >= max(1,n)` 并提供 Device info，成功时写 0。
- 有效计算要求每个指针槽位有效，各输出向量相互独立且不与因子、指针数组和 info 重叠。
  因子全部存储保持不变，RHS 的 padding 保持不变。
  Aarray 保持任务指定及 cuSOLVER 的 `float *Aarray[]` 原型；其只读输入语义由契约和测试保证。
- 无 workspace 参数，也不申请内部 GM scratch；指针数组、因子、RHS、info 均须保持有效直到流完成。

## 调用示例

完整代码见 [`cholesky_test.cpp`](../../test/cholesky/cholesky_test.cpp)。
下面沿用样例的 `Check(int)`；`dAarray`、`dBarray` 均为 Device 指针数组，
`dInfo` 仅需一个 Device `int`。调用前应已检查各因子的分解结果。

```cpp
Check(aclsolverSpotrsBatched(handle, uplo, n, 1, dAarray, lda, dBarray, ldb, dInfo, batchSize));
Check(aclrtSynchronizeStream(stream));
int info = 0;
Check(aclrtMemcpy(&info, sizeof(info), dInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST));
// info == 0 且所有因子有效时，各 Barray 槽位保存对应解向量。
```

## 编译运行

在已完成[环境和依赖准备](../../test/cholesky/README.md#编译运行)的 solver 仓根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=cholesky --run
```

共享 `cholesky_test` 覆盖两个三角方向、真实 Device 指针数组和因子不变性。
