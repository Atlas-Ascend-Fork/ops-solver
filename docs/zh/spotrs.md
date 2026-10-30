# Spotrs

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Ascend 950PR | √ |
| Ascend 950DT | √ |

## 功能说明

`aclsolverSpotrs` 使用已成功分解的 Cholesky 因子求解实数线性方程组：

$$
A_0 X=B,\qquad A_0=L L^T\ \text{或}\ A_0=U^T U.
$$

参数 `A` 存放 L 或 U，必须是 [Spotrf](spotrf.md#ascend-950-接口) 等分解得到的因子；
原始矩阵记作 `A_0`。因子只读，B 原地覆盖为 X。

## 函数原型

```cpp
aclsolverStatus_t aclsolverSpotrs(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, int nrhs, const float *A, int lda,
    float *B, int ldb, int *devInfo);
```

## 参数说明

| 参数 | 输入/输出 | 内存位置及说明 |
|:---|:---:|:---|
| `handle` | 输入 | 由 `aclsolverCreate` 创建并关联计算流的句柄 |
| `uplo` | 输入 | 指定 A 中 L 或 U 所在三角，须与分解时一致 |
| `n` | 输入 | 矩阵阶数，`0 <= n <= 4096` |
| `nrhs` | 输入 | 右端项列数，`0 <= nrhs <= 128` |
| `A` | 输入 | Device FLOAT32 因子，列主序 `A[i+j*lda]`，至少 `lda*n` 个元素 |
| `lda` | 输入 | 因子的列跨度（float 元素数），`lda >= max(1,n)` |
| `B` | 输入/输出 | Device FLOAT32，列主序 `[n,nrhs]`，至少 `ldb*nrhs` 个元素；返回 X |
| `ldb` | 输入 | B 的列跨度（float 元素数），`ldb >= max(1,n)` |
| `devInfo` | 输出 | Device `int*`，一个状态值；同步流后读取 |

## 返回值

接口返回值见[公共约定](cholesky.md#接口公共约定)。
`devInfo=0` 表示参数通过检查；负值为首个非法参数编号，按
`uplo,n,nrhs,A,lda,B,ldb,devInfo` 从 1 开始。
本接口不重新判断因子的正定性或检测零对角，调用前须确认分解成功。

## 算子约束

- 仅支持 FLOAT32；因子、B 与 info 的有效存储不得重叠。
- 仅读取 A 的选定三角；A 的全部存储、另一三角和 padding 均保持不变。
  B 的 padding 保持不变。
- `n=0` 或 `nrhs=0` 时允许 A、B 为空；其他参数仍校验，`devInfo` 仍须提供并写 0。
- 无公开 workspace 查询或 workspace 参数；需要时使用句柄内可复用缓冲区。
- 返回成功表示已排队，读取 X、info 或释放缓冲区前须同步计算流。

## 调用示例

完整代码见 [`cholesky_test.cpp`](../../test/cholesky/cholesky_test.cpp)。
下面沿用样例的 `Check(int)`；`dFactor` 为成功分解后的 Device 因子，
`dB` 已存放右端项，`dInfo` 为 Device 标量，`handle` 已关联 `stream`。

```cpp
Check(aclsolverSpotrs(handle, uplo, n, nrhs, dFactor, lda, dB, ldb, dInfo));
Check(aclrtSynchronizeStream(stream));
int info = 0;
Check(aclrtMemcpy(&info, sizeof(info), dInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST));
// info == 0 且因子有效时，dB 保存列主序解矩阵 X；dFactor 保持不变。
```

## 编译运行

在已完成[环境和依赖准备](../../test/cholesky/README.md#编译运行)的 solver 仓根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=cholesky --run
```

本接口通过共享的 `cholesky_test` 运行；`--ops` 指定测试目录名 `cholesky`。
