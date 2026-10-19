## CgetriBatched

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
    aclsolverCgetriBatched：计算批量复数矩阵的逆矩阵，适用于矩阵维度较大的场景（n ≥ 32）。

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

    [4+0i, 3+0i, 2+0i, 1+0i
     3+0i, 4+0i, 3+0i, 2+0i
     2+0i, 3+0i, 4+0i, 3+0i
     1+0i, 2+0i, 3+0i, 4+0i]
    ```
    输入"n"为： 4（仅为数据格式示意，实际调用需满足 32 ≤ n ≤ 256）\
    输入"batchSize"为：2\
    调用"aclsolverCgetriBatched"算子后，\
    输出"Ainv"为：
    ```
    [0.6+0i, -0.5+0i, 0+0i, 0.1+0i
     -0.5+0i, 1+0i, -0.5+0i, 0+0i
     0+0i, -0.5+0i, 1+0i, -0.5+0i
     0.1+0i, 0+0i, -0.5+0i, 0.6+0i]

    [0.6+0i, -0.5+0i, 0+0i, 0.1+0i
     -0.5+0i, 1+0i, -0.5+0i, 0+0i
     0+0i, -0.5+0i, 1+0i, -0.5+0i
     0.1+0i, 0+0i, -0.5+0i, 0.6+0i]
    ```

## 函数原型
- 函数定义
    ```
    aclError aclsolverCgetriBatched(
        aclsolverHandle_t handle,
        const int64_t n,
        std::complex<float> *A,
        const int64_t lda,
        std::complex<float> *Ainv,
        const int64_t lda_inv,
        int32_t *info,
        int64_t batchSize);
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
        <td align="left">单个矩阵A的行数和列数（方阵）</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入</td>
        <td align="left">公式中的矩阵A，行主序<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[batch, n, n]</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">Ainv</td>
        <td align="center">输出</td>
        <td align="left">输出的逆矩阵<br>数据类型仅支持COMPLEX64，数据格式支持ND，shape为[batch, n, n]</td>
    </tr>
    <tr>
        <td align="center">lda_inv</td>
        <td align="center">输入</td>
        <td align="left">输出的逆矩阵同一列中相邻两行元素间的内存地址偏移量（leading dimension）（当前约束为n）</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">输出</td>
        <td align="left">求逆结果状态信息（当前版本仅保证 info[0] 清零成功态，不逐batch上报）<br>数据类型支持int32_t，数据格式支持ND，单个int32_t（API签名为单指针，非[batch, 1]数组）</td>
    </tr>
    <tr>
        <td align="center">batchSize</td>
        <td align="center">输入</td>
        <td align="left">复数矩阵求逆中的矩阵数量</td>
    </tr>
    </table>

- 算子约束：
  - lda与lda_inv当前必须等于n（见下条）；info参数允许传入nullptr（不关心状态时可不接收）。
  - lda与lda_inv当前必须等于n（矩阵按连续布局处理，batch内每个矩阵均为n*n连续存储），传入其它值将直接返回参数错误。
  - info参数遵循LAPACK语义：成功时写0；矩阵奇异时上报首个奇异列的1-based索引（getrf族）；其余状态经返回值上报。
  - 入参n大于等于32且小于等于256。
  - 入参batchSize小于等于3000。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverCgetriBatched批量复数矩阵求逆示例](../../test/cgetri_batched/cgetri_batched_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
    #include <complex>
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
        int64_t batchSize = 2;
        int64_t n = 48;
        int64_t tensorASize = batchSize * n * n;
        std::vector<std::complex<float>> tensorInAData;
        std::vector<std::complex<float>> tensorInAinvData;
        std::vector<int32_t> tensorInInfoData;
        tensorInAData.resize(tensorASize);
        tensorInAinvData.resize(tensorASize);
        tensorInInfoData.resize(batchSize);
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int32_t i = 0; i < n; i++) {
                for (int32_t j = 0; j < n; j++) {
                    if (i == j) {
                        tensorInAData[n * n * batchIdx + n * i + j] = {4.0f, 0.0f};
                    } else {
                        tensorInAData[n * n * batchIdx + n * i + j] = {static_cast<float>(n - std::abs(i - j)), 0.0f};
                    }
                }
            }
        }
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            for (int32_t i = 0; i < n; i++) {
                for (int32_t j = 0; j < n; j++) {
                    tensorInAinvData[n * n * batchIdx + n * i + j] = {-1.0f, -1.0f};
                }
            }
        }
        for (int32_t batchIdx = 0; batchIdx < batchSize; batchIdx++) {
            tensorInInfoData[batchIdx] = 0;
        }

        // 调用 aclsolverCgetriBatched
        auto ret = aclsolverCgetriBatched(handle, n, tensorInAData.data(), n, tensorInAinvData.data(), n, tensorInInfoData.data(), batchSize);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCgetriBatched failed. ERROR: %d\n", ret); return ret);

        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
