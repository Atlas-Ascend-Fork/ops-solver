# Spotri

## 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Ascend 950PR | √ |
| Ascend 950DT | √ |

## 功能说明

`aclsolverSpotri` 根据 Cholesky 因子计算原始对称正定矩阵的逆矩阵：

$$
C=A_0^{-1},\qquad A_0=L L^T\ \text{或}\ A_0=U^T U.
$$

参数 A 输入为 [Spotrf](spotrf.md#ascend-950-接口) 等分解得到的 L 或 U，原地返回 C 的同侧三角。
另一三角和 padding 保持不变；需要完整逆矩阵时，调用者应将已输出三角镜像到另一侧。

## 函数原型

```cpp
aclsolverStatus_t aclsolverSpotri_bufferSize(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSpotri(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo);
```

## 参数说明

| 参数 | 输入/输出 | 内存位置及说明 |
|:---|:---:|:---|
| `handle` | 输入 | 由 `aclsolverCreate` 创建并关联计算流的句柄 |
| `uplo` | 输入 | 指定输入因子和输出逆矩阵所在三角，须与分解时一致 |
| `n` | 输入 | 矩阵阶数，`0 <= n <= 4096` |
| `A` | 输入/输出 | Device FLOAT32，列主序 `A[i+j*lda]`，至少 `lda*n` 个元素；查询接口不修改 A |
| `lda` | 输入 | 相邻两列起始地址之间的 float 元素数，`lda >= max(1,n)` |
| `Lwork`（查询） | 输出 | Host `int*`，返回所需 float 元素个数 |
| `Workspace` | 输入 | Device FLOAT32 临时区，至少 `Lwork` 个元素 |
| `Lwork`（计算） | 输入 | 实际提供的 float 元素数 |
| `devInfo` | 输出 | Device `int*`，一个状态值；同步流后读取 |

## 返回值

接口返回值见[公共约定](cholesky.md#接口公共约定)。
`devInfo=0` 表示成功；正值 `k` 表示输入因子首个非有限或非正对角的位置（从 1 开始），此时 A 保持不变。
负值按 `uplo,n,A,lda,Workspace,Lwork,devInfo` 编号，报告首个非法参数。
查询接口无 `devInfo`，参数错误仅通过返回值报告。

## 算子约束

- 输入须为有效的非奇异 Cholesky 因子，数据类型仅支持 FLOAT32。
- 非空计算始终需要 workspace，应独立调用本接口的 bufferSize，不能沿用 Spotrf 的查询值。
- `n=0` 时所需 workspace 为 0，允许 A 为空；仍须提供查询的 Host 输出指针或计算的 Device info。
- 非空查询要求 A 非空；workspace、A、info 的有效存储不得重叠。
- workspace 和输入输出须保持有效直到关联计算流完成。

## 调用示例

完整代码见 [`cholesky_test.cpp`](../../test/cholesky/cholesky_test.cpp)。
下面沿用样例的 `Check(int)`；`dFactor` 为成功分解后的 Device 因子，
`dInfo` 为 Device 标量，`handle` 已关联 `stream`。

```cpp
int lwork = 0;
Check(aclsolverSpotri_bufferSize(handle, uplo, n, dFactor, lda, &lwork));
float *workspace = nullptr;
if (lwork > 0) {
    Check(aclrtMalloc(reinterpret_cast<void **>(&workspace),
                      size_t(lwork) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
}
Check(aclsolverSpotri(handle, uplo, n, dFactor, lda, workspace, lwork, dInfo));
Check(aclrtSynchronizeStream(stream));
int info = 0;
Check(aclrtMemcpy(&info, sizeof(info), dInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST));
// info == 0 时，dFactor 的选定三角为原始矩阵逆矩阵的同侧三角。
if (workspace) Check(aclrtFree(workspace));
```

## 编译运行

在已完成[环境和依赖准备](../../test/cholesky/README.md#编译运行)的 solver 仓根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=cholesky --run
```

分解、求解、求逆与批量接口使用同一构建和标准调用样例。
