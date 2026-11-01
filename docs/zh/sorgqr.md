## Sorgqr

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |


注：本算子的 kernel 位于`src/sorgqr/arch35/`，仅在 SOC 为 ascend950 时参与编译（见`src/CMakeLists.txt`
的架构目录机制），其他产品不提供该接口的实现。实现与验证在<term>Ascend 950PR</term>上完成，
同架构的<term>Ascend 950DT</term>复用同一份 arch35 代码。

## 功能说明

- 接口功能\
    aclsolverSorgqr：把 Sgeqrf 产生的 Householder 表示展开成显式的正交矩阵 Q 的前 n 列。

- 计算公式
    $$
    Q = H(1) H(2) \cdots H(k), \quad H(i) = I - \tau_i v_i v_i^T
    $$
    输出为 $Q$ 的前 $n$ 列（$m \times n$），就地覆盖 A。

- 示例 \
    输入"A"与"tau"为 aclsolverSgeqrf 对某个 $m \times n$ 矩阵的输出，\
    输入"m"、"n"、"k" 分别为 Q 的行数、要生成的列数、反射子个数，\
    调用"aclsolverSorgqr"算子后，"A"被覆盖为 $Q$ 的前 $n$ 列，满足 $Q^T Q = I$。\
    $k = 0$ 时结果是单位阵的前 $n$ 列，并<strong>显式写出</strong>（与 LAPACK sorgqr 一致）。

## 函数原型
- 函数定义
    ```
    aclsolverStatus_t aclsolverSorgqr_bufferSize(
        aclsolverHandle_t handle,
        int m,
        int n,
        int k,
        const float *A,
        int lda,
        const float *tau,
        int *lwork);

    aclsolverStatus_t aclsolverSorgqr(
        aclsolverHandle_t handle,
        int m,
        int n,
        int k,
        float *A,
        int lda,
        const float *tau,
        float *work,
        int lwork,
        int *devInfo);
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
        <td align="left">solver handle，通过 aclsolverCreate 创建；计算在 aclsolverSetStream 绑定的 stream 上下发</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">输入</td>
        <td align="left">Q 的行数，需满足 m >= 0 且 m <= 4096（超限返回 ACLSOLVER_STATUS_NOT_SUPPORTED）</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">Q 的列数，需满足 0 <= n <= m</td>
    </tr>
    <tr>
        <td align="center">k</td>
        <td align="center">输入</td>
        <td align="left">反射子个数，需满足 0 <= k <= n</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">输入/输出</td>
        <td align="left">输入为 Sgeqrf 的 Householder 表示，输出被覆盖为 Q 的前 n 列<br>数据类型仅支持 FLOAT32，<strong>列主序</strong>存放</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">输入</td>
        <td align="left">A 的 leading dimension，需满足 lda >= max(1, m)；允许 lda > m</td>
    </tr>
    <tr>
        <td align="center">tau</td>
        <td align="center">输入</td>
        <td align="left">Sgeqrf 输出的 k 个反射子标量<br>数据类型仅支持 FLOAT32</td>
    </tr>
    <tr>
        <td align="center">work</td>
        <td align="center">输入</td>
        <td align="left">device 侧 workspace，元素数不少于 lwork</td>
    </tr>
    <tr>
        <td align="center">lwork</td>
        <td align="center">输入/输出</td>
        <td align="left">_bufferSize 时为输出、主入口时为输入；workspace 的单位是 <strong>float 元素个数</strong>（不是字节），由对应的 _bufferSize 返回</td>
    </tr>
    <tr>
        <td align="center">devInfo</td>
        <td align="center">输出</td>
        <td align="left">device 侧 int32：成功写 0，第 i 个参数非法写 -i（handle 不计入）<br>允许传 nullptr</td>
    </tr>
    </table>

- 算子约束：
  - 输入必须是 Sgeqrf 的输出（Householder 表示），<strong>不是</strong>未分解的矩阵。
  - m、n 为 0 时 lwork = 0，主入口写 devInfo = 0 直接返回成功。
  - _bufferSize 不读 A 与 tau，允许传 nullptr。
  - 内部对"输出 tile 数少于 AIC 核数"的窄 GEMM 采用确定性分 K（各份写独立 partial、AIV 按固定次序归约，不使用原子加），因此同输入多次运行结果逐位一致。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverSorgqr示例](../../test/sorgqr/sorgqr_test.cpp)
- 核心调用步骤：

    ```
    #include <vector>
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
        // 固定写法，acl 初始化
        int32_t deviceId = 0;
        aclrtStream stream = nullptr;
        aclInit(nullptr);
        aclrtSetDevice(deviceId);
        aclrtCreateStream(&stream);

        // 创建 solver handle 并设置 stream
        aclsolverHandle_t handle = nullptr;
        aclsolverCreate(&handle);
        aclsolverSetStream(handle, stream);

        // A、tau 来自先前的 aclsolverSgeqrf 调用
        int m = 4, n = 4, k = 4;
        int lda = m;
        int lwork = 0;
        aclsolverSorgqr_bufferSize(handle, m, n, k, nullptr, lda, nullptr, &lwork);

        void *dA = nullptr, *dTau = nullptr, *dWork = nullptr, *dInfo = nullptr;
        aclrtMalloc(&dA, static_cast<size_t>(lda) * n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dTau, static_cast<size_t>(k) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dWork, static_cast<size_t>(lwork) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dInfo, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST);

        // 调用 aclsolverSorgqr
        auto ret = aclsolverSorgqr(handle, m, n, k, reinterpret_cast<float *>(dA), lda,
                                   reinterpret_cast<const float *>(dTau), reinterpret_cast<float *>(dWork), lwork,
                                   reinterpret_cast<int *>(dInfo));
        CHECK_RET(ret == ACLSOLVER_STATUS_SUCCESS, LOG_PRINT("aclsolverSorgqr failed. ERROR: %d\n", ret); return ret);
        aclrtSynchronizeStream(stream);

        aclrtFree(dA); aclrtFree(dTau); aclrtFree(dWork); aclrtFree(dInfo);


        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
