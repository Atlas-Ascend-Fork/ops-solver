## Sgeqrf

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |


## 功能说明

- 接口功能\
    aclsolverSgeqrf：对单精度浮点矩阵进行QR分解。

- 计算公式
    $$
    A = Q \cdot R
    $$
    其中$A$为$m \times n$阶单精度浮点矩阵，$Q$为正交矩阵，$R$为上三角矩阵。
    该接口使用Householder反射表示$Q$矩阵，输出的$R$矩阵和$Q$矩阵的Householder向量合并存储在$A$中，Householder反射系数存储在$tau$中。

- 示例 \
    输入"A"为：
    ```
    [1.0, 1.0, 1.0, 1.0
     1.0, 2.0, 3.0, 4.0
     1.0, 3.0, 6.0, 10.0
     1.0, 4.0, 10.0, 20.0]
    ```
    输入"m"为： 4\
    输入"n"为： 4\
    调用"aclsolverSgeqrf"算子后，\
    输出"A"为R矩阵和Q矩阵的Householder向量合并存储：
    ```
    [-2.0000, -5.0000, -10.0000, -17.5000
    0.3333, -2.2361, -6.7082, -14.0872
    0.3333, 0.4472, 1.0000, 3.5000
    0.3333, 0.8944, -0.6793, -0.2236]
    ```
    输出"tau"为Householder反射系数：
    ```
    [1.5 1 1.36852 0​]
    ```

## 函数原型

- 函数定义
    ```
    aclError aclsolverSgeqrf(
        aclsolverHandle_t handle,
        const int64_t m,
        const int64_t n,
        float *A,
        const int64_t lda,
        float *tau,
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
        <td align="left">输入为矩阵A，输出为R矩阵和Q矩阵的Householder向量，合并存储在A中<br>数据类型仅支持FLOAT32，数据格式支持ND，shape为[m, n]，矩阵按列主序存储</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A的主维度，列主序下为相邻列首元素之间的元素跨度，需满足lda >= max(1, m)</td>
    </tr>
    <tr>
        <td align="center">tau</td>
        <td align="center">输出</td>
        <td align="left">Q矩阵的Householder反射系数<br>数据类型支持FLOAT32，数据格式支持ND，shape为[min(m,n)]</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">分解结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - lda需满足lda >= max(1, m)。
  - info参数在当前版本实际未启用。
  - 输入矩阵A按列主序存储。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例

- 完整代码示例：[aclsolverSgeqrf单精度矩阵QR分解示例](../../test/sgeqrf/sgeqrf_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
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
        size_t aMatrixFileSize = m * n * sizeof(float);
        float* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        size_t tauFileSize = std::min(m, n) * sizeof(float);
        float* tau;
        aclrtMallocHost((void**)(&tau), tauFileSize);
        int32_t info = 0;

        // 调用 aclsolverSgeqrf
        auto ret = aclsolverSgeqrf(handle, m, n, A, m, tau, &info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverSgeqrf failed. ERROR: %d\n", ret); return ret);

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
