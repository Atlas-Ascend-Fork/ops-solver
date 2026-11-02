## Cgeqrf

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |


## 功能说明

- 接口功能\
    aclsolverCgeqrf：对单精度复数矩阵进行QR分解。

- 计算公式

    $$
    A = Q \cdot R
    $$

    其中$A$为$m \times n$阶单精度复数矩阵，$Q$为酉矩阵，$R$为上三角矩阵，满足$Q^H Q = I$。
    该接口使用Householder反射表示$Q$矩阵，输出的$R$矩阵和$Q$矩阵的Householder向量合并存储在$A$中，Householder反射系数存储在$tau$中。

- 示例

    以下示例中，输入矩阵为虚部均为0的单精度复数矩阵，复数以`实部+虚部i`的形式表示。

    输入"A"为：

    ```
    [ 1.0+1.0i,  -0.8-3.6i
    1.0-1.0i,   0.6+0.2i ]
    ```

    输入"m"为：2\
    输入"n"为：2\
    调用"aclsolverCgeqrf"算子后，\
    输出"A"为R矩阵和Q矩阵的Householder向量合并存储：

    ```
    [ -2.65+0.0i, 0.76-1.51i
    0.44-0.40i 2.04+0.0i]
    ```

    输出"tau"为Householder反射系数：

    ```
    [ 1.38+0.38i, 1.79+0.61i ]
    ```

## A2/A3 接口

以下原型和参数适用于 A2/A3；950DT/PR 使用本页后文的工作区接口。

## 函数原型

- 函数定义

    ```
    aclError aclsolverCgeqrf(
        aclsolverHandle_t handle,
        const int64_t m,
        const int64_t n,
        std::complex<float> *A,
        const int64_t lda,
        std::complex<float> *tau,
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
        <td align="left">矩阵A的行数</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的列数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为矩阵A，输出为R矩阵和Q矩阵的Householder向量，合并存储在A中<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[m, n]，矩阵按列主序存储</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A的主维度，列主序下为相邻列首元素之间的元素跨度</td>
    </tr>
    <tr>
        <td align="center">tau</td>
        <td align="center">输出</td>
        <td align="left">Q矩阵的Householder反射系数<br>数据类型支持COMPLEX64，数据格式支持ND，shape为[min(m,n)]</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">分解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - info参数在当前版本实际未启用。
  - 输入矩阵A按列主序存储。

- 调用实现

    使用内核调用符<<<>>>调用核函数。

## 调用示例

- 完整代码示例：[aclsolverCgeqrf单精度复数矩阵QR分解示例](../../test/cgeqrf/cgeqrf_test.cpp)

- 核心调用步骤：

    ```
    #include <vector>
    #include <complex>
    #include <algorithm>
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

        // 构造输入数据
        int64_t m = 4;
        int64_t n = 4;
        size_t aMatrixFileSize = m * n * sizeof(std::complex<float>);
        std::complex<float>* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        size_t tauFileSize = std::min(m, n) * sizeof(std::complex<float>);
        std::complex<float>* tau;
        aclrtMallocHost((void**)(&tau), tauFileSize);

        int32_t info = 0;

        // 调用 aclsolverCgeqrf
        auto ret = aclsolverCgeqrf(handle, m, n, A, m, tau, &info);
        CHECK_RET(ret == ACL_SUCCESS,
                  LOG_PRINT("aclsolverCgeqrf failed. ERROR: %d\n", ret);
                  return ret);

        // 释放资源
        aclrtFreeHost(tau);
        aclrtFreeHost(A);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```


## 950DT/PR 接口

以下带工作区的 C 接口适用于 Ascend 950DT/PR，不与前面的 A2/A3 七参数 C++ 接口混用。

## 函数原型

```c
aclsolverStatus_t aclsolverCgeqrf_bufferSize(
    aclsolverHandle_t handle, int m, int n, aclsolverFloatComplex *A, int lda, int *Lwork);

aclsolverStatus_t aclsolverCgeqrf(
    aclsolverHandle_t handle, int m, int n, aclsolverFloatComplex *A, int lda,
    aclsolverFloatComplex *TAU, aclsolverFloatComplex *Workspace, int Lwork, int *devInfo);
```

## 参数说明

| 参数 | 归属 | 说明 |
| --- | --- | --- |
| handle | Host | `aclsolverCreate` 创建的非空句柄；使用其 stream |
| m、n | Host 标量 | 非负行数、列数；支持方阵、瘦高和扁宽矩阵 |
| A | Device 输入/输出 | COMPLEX64 列主序，元素 `(r,c)` 位于 `A[r+c*lda]` |
| lda | Host 标量 | `lda >= max(1,m)`；padding 保持不变 |
| TAU | Device 输出 | 长度至少 `min(m,n)`，不得与 A、workspace 重叠 |
| Workspace | Device 工作区 | 按本次 bufferSize 查询结果分配，使用 `aclrtMalloc` |
| Lwork | Host | 查询时为输出指针；计算时为工作区的复数元素个数，单位为 8 字节 |
| devInfo | Device 输出 | 一个 INT32；成功为 0，参数非法为 `-i`，参数编号不计 handle |

## 约束与状态

- 数据类型为两个连续 FLOAT32 的 `aclsolverFloatComplex {real, imag}`；不支持广播或行主序。
- 任务保证范围为 `m,n` 从 1 到 4096 的合法组合，另外处理空问题；更大维数不作为本任务验收保证。
- `m=0` 或 `n=0` 时成功写入 info=0，允许 A、TAU 为空；lda 仍须合法，devInfo 必须可写。
- 非空问题的输入必须为有限复数。非有限输入、非法参数或不足的工作区返回
  `ACLSOLVER_STATUS_INVALID_VALUE`，并在 info 可写时写入对应负编号。
- 空 handle 返回已有的 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`。
- bufferSize 不执行分解、不读取 Device 数据；计算不分配隐藏的 Device 工作区。
- 输入有限性检查为满足同步返回状态要求，会同步调用方 stream；后续计算及 info 写入
  使用同一 stream。读取结果或释放相关 Device 内存前，调用方应同步该 stream。
- 核心运算在 AI Core 上完成，补偿 HF32 用于块更新；不使用 CPU 计算回退。

## 输入检查与同步开销

非空计算路径在下发主计算前，先由设备端扫描实际使用的输入，检查 NaN/Inf，
随后同步 handle 对应的 stream，并将 4 字节检查结果复制到 Host，以同步返回
`ACLSOLVER_STATUS_INVALID_VALUE`。该同步也会等待同一 stream 上此前排队的任务，
因此调用不是完全异步；主计算仍在同一 stream 上下发，API 返回不代表主计算完成。
空问题与参数拒绝路径不执行这次扫描。

扫描增加与有效输入元素数相关的读取开销；leading-dimension padding 及反射器的
隐式对角/上三角不属于有效输入。大矩阵应同时评估扫描 kernel 占比和完整调用耗时。
kernel 性能统计应包含检查 kernel；仅统计主计算会低估成本。kernel 耗时不包含
Host 等待和 D2H 回读的全部开销，不能据此宣称端到端延迟相同。
当前不提供跳过检查的开关，以保持非有限输入的错误返回契约。

## 调用与测试

先创建 handle 并设置 stream，再查询 Lwork、分配 `Lwork*sizeof(aclsolverFloatComplex)`
字节的 Device 工作区，调用计算接口并同步 stream，最后将 devInfo 和输出复制到 Host 检查。
公开头文件支持 C 调用；完整资源管理示例见
[profile_case.cpp](../../test/complex_qr/profile_case.cpp)，正确性和性能命令见
[测试说明](../../test/complex_qr/README.md)。

由反射器生成 Q 使用 [Cungqr](cungqr.md)，施加 Q 使用 [Cunmqr](cunmqr.md)。
