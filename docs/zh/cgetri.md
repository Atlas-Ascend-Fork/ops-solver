## Cgetri

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     √    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     √    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     ×    |


注：表中"Atlas 训练系列产品"指非A2代际的历史训练产品；Atlas A2训练/推理系列产品（含910B等）单独标注支持情况。

## 功能说明

- 接口功能\
    aclsolverCgetri：计算复数矩阵的逆矩阵。

- 计算公式
    $$
    A^{-1}A = I
    $$
    其中$A$为$n \times n$阶非奇异复数方阵，$I$为$n$阶单位矩阵。

- 示例 \
    输入"A"为：
    ```
    [4+0i, 3+0i, 2+0i, 1+0i
     3+0i, 4+0i, 3+0i, 2+0i
     2+0i, 3+0i, 4+0i, 3+0i
     1+0i, 2+0i, 3+0i, 4+0i]
    ```
    输入"n"为： 4\
    调用"aclsolverCgetri"算子后，\
    输出"A"为：
    ```
    [0.6+0i, -0.5+0i, 0+0i, 0.1+0i
     -0.5+0i, 1+0i, -0.5+0i, 0+0i
     0+0i, -0.5+0i, 1+0i, -0.5+0i
     0.1+0i, 0+0i, -0.5+0i, 0.6+0i]
    ```

## 函数原型
- 函数定义
    ```
    aclError aclsolverCgetri(
        aclsolverHandle_t handle,
        const int64_t n,
        std::complex<float> *A,
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
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">矩阵A的行数和列数</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为待求逆的原始矩阵A（算子内部完成LU分解与求解，不消费Cgetrf的输出LU因子；请勿传入Cgetrf返回的packed LU，否则结果会被主元置换且不报错），输出为逆矩阵<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">求逆结果信息<br>数据类型支持int32_t</td>
    </tr>
    </table>

- 算子约束：
  - n 的有效范围为 [1, 8192]（kernel 列暂存区设计上限，超出将返回参数错误）。
  - lda参数当前必须等于n（见下条）；info参数允许传入nullptr（不关心状态时可不接收）。
  - lda当前必须等于n（矩阵按m*n/n*n连续布局处理），传入其它值将直接返回参数错误。
  - info参数遵循LAPACK语义：成功时写0；矩阵奇异时上报首个奇异列的1-based索引（getrf族）；其余状态经返回值上报。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverCgetri复数矩阵求逆示例](../../test/cgetri/cgetri_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
    #include <complex>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    #include <cstdio>

    #define CHECK_RET(cond, return_expr) \
        do {                            \
            if (!(cond)) {              \
                return_expr;            \
            }                           \
        } while (0)

    #define LOG_PRINT(message, ...) \
        std::printf(message, ##__VA_ARGS__)

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
        int64_t n = 4;
        size_t aMatrixFileSize = n * n * sizeof(float) * 2;
        std::complex<float>* A;
        aclrtMallocHost((void**)(&A), aMatrixFileSize);
        // 填充A矩阵数据...

        int32_t infoVal = 0;
        int32_t *info = &infoVal;

        // 调用 aclsolverCgetri
        auto ret = aclsolverCgetri(handle, n, A, n, info);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCgetri failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclrtFreeHost(A);
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
