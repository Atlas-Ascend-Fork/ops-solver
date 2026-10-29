## SgetriBatched

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   | 待验证 |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     | 待验证 |
|  <term>Ascend 950PR</term>   |     √    |
|  <term>Ascend 950DT</term>   |  待验证  |


注：表中"Atlas 训练系列产品"指非A2代际的历史训练产品；Atlas A2训练/推理系列产品（含910B等）单独标注支持情况。

本次实数Device接口在Ascend 950PR及仓库README列出的配套CANN 9.0.0环境完成自验证；其余标为“待验证”的产品尚需对应平台回归。

## 功能说明

- 接口功能\
    aclsolverSgetriBatched：计算批量单精度实数矩阵的逆矩阵，从每个矩阵已分解的LU与对应主元计算逆矩阵。

- 计算公式
    $$
    C_b A_{\mathrm{original},b} = I
    $$
    其中$A_{\mathrm{original},b}$为第$b$个$n \times n$阶原始实数矩阵，$C_b$为输出逆矩阵，$I$为$n$阶单位矩阵。

- 示例 \
    两个原始矩阵分别为：
    ```text
    [2, 0       [5,  0
     0, 4]       0, 10]
    ```
    输入"n"为：2，输入"batchSize"为：2。这两个对角矩阵的packed LU与原矩阵相同，各自主元为[1,2]；实际调用时传入对应分解产生的LU和主元。
    输出逆矩阵分别为：
    ```text
    [0.5, 0         [0.2, 0
     0,   0.25]      0,   0.1]
    ```

## 函数原型

- 函数定义

    公开头文件为 `cann_ops_solver.h`。

    ```cpp
    aclsolverStatus_t aclsolverSgetriBatched(
        aclsolverHandle_t handle, int n, const float *const Aarray[], int lda,
        const int *PivotArray, float *const Carray[], int ldc,
        int *infoArray, int batchSize);
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
        <td align="left">solver handle，通过aclsolverCreate创建，并通过aclsolverSetStream设置stream</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">每个方阵的行数和列数，0≤n≤4096</td>
    </tr>
    <tr>
        <td align="center">Aarray</td>
        <td align="center">输入</td>
        <td align="left">Device上的Device指针数组，长度为batchSize；每项指向列主序FLOAT32 packed LU，矩阵分配至少lda*n个元素</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">每个输入矩阵相邻两列起始元素的步长，lda≥max(1,n)</td>
    </tr>
    <tr>
        <td align="center">PivotArray</td>
        <td align="center">输入</td>
        <td align="left">Device连续INT32数组，长度为batchSize*n；第b组为PivotArray[b*n+k]，保存对应LU的1-based主元；nullptr表示不选主元</td>
    </tr>
    <tr>
        <td align="center">Carray</td>
        <td align="center">输出</td>
        <td align="left">Device上的Device指针数组，长度为batchSize；每项指向独立的列主序FLOAT32逆矩阵，分配至少ldc*n个元素</td>
    </tr>
    <tr>
        <td align="center">ldc</td>
        <td align="center">输入</td>
        <td align="left">每个输出矩阵相邻两列起始元素的步长，ldc≥max(1,n)</td>
    </tr>
    <tr>
        <td align="center">infoArray</td>
        <td align="center">输出</td>
        <td align="left">Device INT32数组，长度为batchSize；每个矩阵独立报告0、正奇异索引或负参数索引，编号不计handle</td>
    </tr>
    <tr>
        <td align="center">batchSize</td>
        <td align="center">输入</td>
        <td align="left">矩阵数量，Host int，batchSize≥0；实际容量受可用设备内存限制</td>
    </tr>
    </table>

- 算子约束：

  - 每个输入矩阵须为已分解LU，PivotArray须与分解结果对应；nullptr表示不选主元的LU。
  - 输入、输出矩阵均使用Device列主序，支持合法lda、ldc padding；各输出矩阵彼此独立，并与所有输入、主元和INFO分开分配。
  - 指针数组本身和所指向的矩阵均在Device；输入矩阵可位于独立分配中。
  - batchSize=0成功返回；n=0且batchSize>0时向全部INFO槽位写0。
  - 工作区由handle内部管理，计算接口没有调用方workspace参数。
  - 有效矩阵元素要求为有限FLOAT32数；A、主元和INFO须满足各自类型的对齐与容量要求。
  - INFO为Device输出；返回码报告参数、资源或执行状态。读取INFO及数值结果前同步handle关联的stream，并同时检查返回码和INFO。
  - 部分Device元数据和有限值校验会同步调用方stream；矩阵数值计算在AI Core执行。
  - 输入、输出、主元、指针数组和workspace在计算完成前保持有效；相同输入在同一stream串行重复调用时保持bit-wise一致。

- 调用实现

    Host完成参数校验并在handle关联的stream上启动Ascend C核函数；临时工作区随handle管理。

- 返回值

    成功提交返回 `ACLSOLVER_STATUS_SUCCESS`；handle为空返回 `ACLSOLVER_STATUS_HANDLE_IS_NULLPTR`，非法参数返回 `ACLSOLVER_STATUS_INVALID_VALUE`。资源或运行时失败返回相应 `aclsolverStatus_t`，状态定义见 [cann_ops_solver_common.h](../../include/cann_ops_solver_common.h)。

## 调用示例

- 完整代码示例：[SgetriBatched基础测试](../../test/sgetri_batched/sgetri_batched_test.cpp)；构建和测试说明见[test/sgetri_batched](../../test/sgetri_batched/README.md)。

- 核心调用步骤：

    调用方先完成ACL初始化、设备选择、handle与stream创建。以下函数设置handle关联的stream，分配Device资源、调用公开接口并回传结果；函数返回后由调用方销毁handle和stream。

    ```cpp
    #include <algorithm>
    #include <stdexcept>
    #include <vector>
    #include "acl/acl.h"
    #include "cann_ops_solver.h"

    static void Require(bool value)
    {
        if (!value)
        {
            throw std::runtime_error("solver example failed");
        }
    }

    // Synchronize before releasing Device memory used by the supplied stream.
    struct DeviceBuffer
    {
        void *data = nullptr;
        aclrtStream stream;
        DeviceBuffer(size_t bytes, aclrtStream s) : stream(s)
        {
            Require(aclrtMalloc(&data, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        }
        ~DeviceBuffer()
        {
            if (data != nullptr && aclrtSynchronizeStream(stream) == ACL_SUCCESS)
            {
                aclrtFree(data);
            }
        }
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;
        template <typename T> T *As() { return static_cast<T *>(data); }
    };

    // The caller initializes ACL, selects a device, creates handle/stream,
    // and destroys them after this function returns.
    std::vector<float> SgetriBatchedExample(aclsolverHandle_t handle, aclrtStream stream)
    {
        Require(aclsolverSetStream(handle, stream) == ACLSOLVER_STATUS_SUCCESS);
        const int n = 2, batchSize = 2;
        const float host0[] = {2, 0, 0, 4}, host1[] = {5, 0, 0, 10};
        const size_t bytes = sizeof(host0);
        DeviceBuffer a0(bytes, stream), a1(bytes, stream), c0(bytes, stream), c1(bytes, stream);
        Require(aclrtMemcpy(a0.data, bytes, host0, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
        Require(aclrtMemcpy(a1.data, bytes, host1, bytes, ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
        const float *hostInputs[] = {a0.As<float>(), a1.As<float>()};
        float *hostOutputs[] = {c0.As<float>(), c1.As<float>()};
        DeviceBuffer inputs(sizeof(hostInputs), stream), outputs(sizeof(hostOutputs), stream);
        DeviceBuffer info(batchSize * sizeof(int), stream);
        Require(aclrtMemcpy(inputs.data, sizeof(hostInputs), hostInputs, sizeof(hostInputs),
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
        Require(aclrtMemcpy(outputs.data, sizeof(hostOutputs), hostOutputs, sizeof(hostOutputs),
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
        const int hostPivots[] = {1, 2, 1, 2};
        DeviceBuffer piv(sizeof(hostPivots), stream);
        Require(aclrtMemcpy(piv.data, sizeof(hostPivots), hostPivots, sizeof(hostPivots),
                            ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
        Require(aclsolverSgetriBatched(handle, n, inputs.As<const float *>(), n,
                                      piv.As<int>(), outputs.As<float *>(), n, info.As<int>(), batchSize) == ACLSOLVER_STATUS_SUCCESS);
        Require(aclrtSynchronizeStream(stream) == ACL_SUCCESS);
        int hostInfo[batchSize] = {-1, -1};
        Require(aclrtMemcpy(hostInfo, sizeof(hostInfo), info.data, sizeof(hostInfo),
                            ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
        Require(hostInfo[0] == 0 && hostInfo[1] == 0);
        std::vector<float> result(batchSize * n * n);
        Require(aclrtMemcpy(result.data(), bytes, c0.data, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
        Require(aclrtMemcpy(result.data() + n * n, bytes, c1.data, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
        return result;
    }
    ```

    示例中的资源对象在释放Device内存前同步stream；同步失败时保留相应分配。完整环境初始化和测试异常处理见上述源码。
