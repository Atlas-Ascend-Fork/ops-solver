# Spotrf

`aclsolverSpotrf` 有两套不同的接口原型，调用时须同时匹配参数数量、类型和内存约定：

| 接口 | 产品 | 参数与内存约定 |
|:---|:---|:---|
| [Ascend 950 接口](#ascend-950-接口) | Ascend 950PR | `aclsolverFillMode_t`、32 位维数、8 个参数，包含 Device workspace；矩阵与 info 在 Device，矩阵为列主序，返回 `aclsolverStatus_t` |
| [A2/A3 接口](#a2a3-接口) | Atlas A2/A3 | `char uplo`、64 位 `m,n`、7 个参数，不含 workspace；沿用 Host 矩阵输入及原有调用约定，返回 `aclError` |

950 接口保留 C 链接符号；A2/A3 接口保留已有 C++ 链接符号。
C++ 调用按完整原型选择对应重载，不根据设备隐式转换参数或数据布局。
两套接口的实现、参数校验、返回值和存储约定分别如下。

## Ascend 950 接口

### 产品支持情况

| 产品 | 是否支持 |
|:---|:---:|
| Atlas 200I/500 A2 推理产品、Atlas 推理系列产品、Atlas 训练系列产品 | × |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | × |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | × |
| Ascend 950PR | √ |
| Ascend 950DT | √ |

### 功能说明

`aclsolverSpotrf` 对单精度实数对称正定矩阵进行 Cholesky 分解：

$$
A=L L^T\quad(\mathrm{LOWER}),\qquad A=U^T U\quad(\mathrm{UPPER}).
$$

只读取和覆盖 `uplo` 指定的三角，另一三角和 leading-dimension padding 保持不变。
`aclsolverSpotrf_bufferSize` 查询调用者需要提供的 workspace 大小。

### 函数原型

```cpp
aclsolverStatus_t aclsolverSpotrf_bufferSize(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverSpotrf(
    aclsolverHandle_t handle, aclsolverFillMode_t uplo,
    int n, float *A, int lda, float *Workspace, int Lwork, int *devInfo);
```

### 参数说明

| 参数 | 输入/输出 | 内存位置及说明 |
|:---|:---:|:---|
| `handle` | 输入 | 由 `aclsolverCreate` 创建并通过 `aclsolverSetStream` 关联计算流的句柄 |
| `uplo` | 输入 | `ACLSOLVER_FILL_MODE_LOWER` 或 `ACLSOLVER_FILL_MODE_UPPER` |
| `n` | 输入 | 矩阵阶数，`0 <= n <= 4096` |
| `A` | 输入/输出 | Device FLOAT32，列主序；`A[i+j*lda]` 为第 i 行、第 j 列，存储至少 `lda*n` 个元素；查询接口不修改 A |
| `lda` | 输入 | 相邻两列起始地址之间的 float 元素数，`lda >= max(1,n)` |
| `Lwork`（查询） | 输出 | Host `int*`，返回所需 float 元素个数 |
| `Workspace` | 输入 | Device FLOAT32 临时缓冲区，至少 `Lwork` 个元素 |
| `Lwork`（计算） | 输入 | 实际提供的 workspace 元素个数，不能按字节传入 |
| `devInfo` | 输出 | Device `int*`，一个状态值；同步流后读取 |

### 返回值

接口返回 `aclsolverStatus_t`，具体取值见[公共约定](cholesky.md#接口公共约定)。
计算接口的 `devInfo=0` 表示分解成功；正值 `k` 表示首个非正定主元位置（从 1 开始），
保留此前完成的列；负值表示参数非法，编号按 `uplo,n,A,lda,Workspace,Lwork,devInfo`
从 1 开始，多个参数非法时报告首个。空句柄通过返回值报告，不写 `devInfo`。
查询接口无 `devInfo`，参数错误仅通过返回值报告。

### 算子约束

- 有效计算的输入须为对称正定矩阵，数据类型仅支持 FLOAT32。
- `n=0` 仍校验枚举、`lda`、workspace 参数和 `devInfo`，允许 `A=nullptr`；查询返回 0。
- 非空查询要求 `A` 为非空 Device 指针，`Lwork` 输出指针始终不能为空。
- `n<=64` 时所需 workspace 为 0，可传 `Workspace=nullptr,Lwork=0`；
  其余规模按查询结果分配。若主动传入正的 `Lwork`，`Workspace` 不能为空。
- A、workspace 与 info 的有效存储不得重叠，并须保持有效直到计算流完成。

### 调用示例

完整初始化、Device 分配、数据传输和资源释放见
[`cholesky_test.cpp` 的 `Run`](../../test/cholesky/cholesky_test.cpp)。
下面片段沿用样例的 `Check(int)` 错误检查函数；`handle` 已关联 `stream`，
`dA` 已存放输入，`dInfo` 已分配为 Device `int`。

```cpp
int lwork = 0;
Check(aclsolverSpotrf_bufferSize(handle, uplo, n, dA, lda, &lwork));
float *workspace = nullptr;
if (lwork > 0) {
    Check(aclrtMalloc(reinterpret_cast<void **>(&workspace),
                      size_t(lwork) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
}
Check(aclsolverSpotrf(handle, uplo, n, dA, lda, workspace, lwork, dInfo));
Check(aclrtSynchronizeStream(stream));
int info = 0;
Check(aclrtMemcpy(&info, sizeof(info), dInfo, sizeof(info), ACL_MEMCPY_DEVICE_TO_HOST));
// info == 0 时 dA 的选定三角可以继续传给 Spotrs 或 Spotri。
if (workspace) Check(aclrtFree(workspace));
```

### 编译运行

本接口与其余 Cholesky 接口共用 `src/cholesky/arch35/` 和 `test/cholesky/`。
在 solver 仓根目录运行：

```bash
bash build.sh --soc=ascend950 --ops=cholesky --run
```

环境、离线依赖和三段测试流程见[调用样例说明](../../test/cholesky/README.md#编译运行)。

## A2/A3 接口

### 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |


### 功能说明

- 接口功能\
    aclsolverSpotrf：对实数正定对称矩阵进行cholesky分解。

- 计算公式
    $$
    A = L \cdot L^T
    $$
    其中$A$为$n \times n$阶实数正定对称矩阵，$L$为单位下三角矩阵，$L^T$为上三角矩阵。

- 示例 \
    输入"A"为：
    ```
    [1.000000 0.109793 -2.551319 -1.269791
     0.109793 81.012054 7.577830 12.328438
     -2.551319 7.577830 32.271538 3.670894
     -1.269791 12.328438 3.670894 103.555702]
    ```
    输入"n"为： 4\
    调用"aclsolverSpotrf"算子后，\
    输出"A"为L^T存储：
    ```
    [1.000000 0.109793 -2.551319 -1.269791
     0.000000 9.000000 0.873105 1.385317
     0.000000 0.000000 5.000000 -0.155655
     0.000000 0.000000 0.000000 10.000000]
    ```


### 函数原型
- 函数定义
    ```
    aclError aclsolverSpotrf(
        aclsolverHandle_t handle,
        const char uplo,
        const int64_t m,
        const int64_t n,
        float *A,
        const int64_t lda,
        int32_t *info);
    ```
- 参数说明：
    <table>
    <tr>
        <td align="center">参数名</td>
        <td align="center">输入输出</td>
        <td align="center">描述</td>
    </tr>
    <tr>
        <td align="center">handle</td>
        <td align="center">输入</td>
        <td align="left">solver handle，通过aclsolverCreate创建</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的行数，当前约束为m == n（Cholesky要求方阵）</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的列数，取值范围[1, 8192]</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵A，输出为L^T矩阵<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A左右相邻元素间的内存地址偏移量（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">分解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - m必须等于n（Cholesky要求方阵）；n取值范围为[1, 8192]，超出返回ACL_ERROR_INVALID_PARAM。
  - lda >= n且lda <= 16384（kernel以uint16_t字节计数承载行间隙，'U'因子成员列数逐轮递减，最坏行间隙由lda本身决定而非lda - n）。
  - info须为有效指针（当前版本未解引用，预留扩展）。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

### 调用示例
- 完整代码示例：[aclsolverSpotrf矩阵cholesky分解示例](../../test/spotrf/spotrf_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    int32_t main(int32_t argc, char *argv[])
    {
        // 固定写法，acl初始化
        int32_t deviceId = 0;
        aclrtStream stream = nullptr;
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        aclrtCreateStream(&stream);

        // 创建solver handle并设置stream
        aclsolverHandle_t handle = nullptr;
        aclsolverCreate(&handle);
        aclsolverSetStream(handle, stream);

        // 构造输入数
        int64_t n = 4;
        size_t aMatrixFileSize = n * n * sizeof(float);
        float* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t *info;

        // 调用 aclsolverSpotrf
        auto ret = aclsolverSpotrf(handle, uplo, n, n, A, n, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSpotrf failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclrtFreeHost(A);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
