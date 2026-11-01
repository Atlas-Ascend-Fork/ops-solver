## Xlarft

## 产品支持情况

|产品             |  是否支持  |
|:-------------------------|:----------:|
|  <term>Atlas 200I/500 A2 推理产品</term>    |     ×    |
|  <term>Atlas 推理系列产品</term>    |     ×    |
|  <term>Atlas 训练系列产品</term>    |     ×    |
|  <term>Atlas A3 训练系列产品/Atlas A3 推理系列产品</term>   |     ×    |
|  <term>Atlas A2 训练系列产品/Atlas A2 推理系列产品</term>     |     ×    |
|  <term>Ascend 950PR/Ascend 950DT</term>   |     √    |


注：本算子的 kernel 位于`src/xlarft/arch35/`，仅在 SOC 为 ascend950 时参与编译（见`src/CMakeLists.txt`
的架构目录机制），其他产品不提供该接口的实现。实现与验证在<term>Ascend 950PR</term>上完成，
同架构的<term>Ascend 950DT</term>复用同一份 arch35 代码。

## 功能说明

- 接口功能\
    aclsolverXlarft：构造块反射子的三角因子 T。

- 计算公式
    $$
    H(1) H(2) \cdots H(k) = I - V T V^T \quad (direct = FORWARD,\ T\ \text{上三角})
    $$
    $$
    H(k) \cdots H(2) H(1) = I - V T V^T \quad (direct = BACKWARD,\ T\ \text{下三角})
    $$
    其中 $V$ 为 $n \times k$、列主序，<strong>单位对角隐含不存</strong>：第 $i$ 列在第 $i$ 行（FORWARD）
    或第 $n-k+i$ 行（BACKWARD）处为 1，该 1 另一侧的元素不被读取。

- 示例 \
    输入"V"、"tau"为某组反射子，"n"为块反射子阶数、"k"为反射子个数，\
    取 direct = ACLSOLVER_DIRECT_FORWARD、storev = ACLSOLVER_STOREV_COLUMNWISE，\
    调用"aclsolverXlarft"算子后，"T"的<strong>上三角</strong>被写为三角因子，
    下三角<strong>原样保留</strong>（与 LAPACK slarft 一致，只写被引用的那一半）。

## 函数原型
- 函数定义
    ```
    aclsolverStatus_t aclsolverXlarft_bufferSize(
        aclsolverHandle_t handle,
        aclsolverParams_t params,
        aclsolverDirectMode_t direct,
        aclsolverStorevMode_t storev,
        int64_t n,
        int64_t k,
        aclsolverDataType_t dataTypeV,
        const void *V,
        int64_t ldv,
        aclsolverDataType_t dataTypeTau,
        const void *tau,
        aclsolverDataType_t dataTypeT,
        void *T,
        int64_t ldt,
        aclsolverDataType_t computeType,
        size_t *workspaceInBytesOnDevice,
        size_t *workspaceInBytesOnHost);

    aclsolverStatus_t aclsolverXlarft(
        aclsolverHandle_t handle,
        aclsolverParams_t params,
        aclsolverDirectMode_t direct,
        aclsolverStorevMode_t storev,
        int64_t n,
        int64_t k,
        aclsolverDataType_t dataTypeV,
        const void *V,
        int64_t ldv,
        aclsolverDataType_t dataTypeTau,
        const void *tau,
        aclsolverDataType_t dataTypeT,
        void *T,
        int64_t ldt,
        aclsolverDataType_t computeType,
        void *bufferOnDevice,
        size_t workspaceInBytesOnDevice,
        void *bufferOnHost,
        size_t workspaceInBytesOnHost);
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
        <td align="center">params</td>
        <td align="center">输入</td>
        <td align="left">必须传 nullptr：本实现不提供参数对象，传非空返回参数错误</td>
    </tr>
    <tr>
        <td align="center">direct</td>
        <td align="center">输入</td>
        <td align="left">ACLSOLVER_DIRECT_FORWARD 或 ACLSOLVER_DIRECT_BACKWARD，<strong>两者均支持</strong></td>
    </tr>
    <tr>
        <td align="center">storev</td>
        <td align="center">输入</td>
        <td align="left"><strong>仅支持</strong> ACLSOLVER_STOREV_COLUMNWISE；传 ROWWISE 返回 ACLSOLVER_STATUS_NOT_SUPPORTED</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">输入</td>
        <td align="left">块反射子阶数，需满足 n >= k >= 0 且 n <= 4096</td>
    </tr>
    <tr>
        <td align="center">k</td>
        <td align="center">输入</td>
        <td align="left">反射子个数（也是 T 的阶），需满足 0 <= k <= min(n, 256)；超过 256 返回 ACLSOLVER_STATUS_NOT_SUPPORTED</td>
    </tr>
    <tr>
        <td align="center">dataTypeV / dataTypeTau / dataTypeT / computeType</td>
        <td align="center">输入</td>
        <td align="left"><strong>四者均仅支持</strong> ACLSOLVER_R_32F；任一不是则返回 ACLSOLVER_STATUS_NOT_SUPPORTED</td>
    </tr>
    <tr>
        <td align="center">V</td>
        <td align="center">输入</td>
        <td align="left">n x k 反射子矩阵，<strong>列主序</strong>，单位对角隐含<br>数据类型仅支持 FLOAT32</td>
    </tr>
    <tr>
        <td align="center">ldv</td>
        <td align="center">输入</td>
        <td align="left">V 的 leading dimension，需满足 ldv >= max(1, n)</td>
    </tr>
    <tr>
        <td align="center">tau</td>
        <td align="center">输入</td>
        <td align="left">k 个反射子标量<br>数据类型仅支持 FLOAT32</td>
    </tr>
    <tr>
        <td align="center">T</td>
        <td align="center">输出</td>
        <td align="left">k x k 三角因子，<strong>列主序</strong>；只写被引用的那一个三角，另一半不动</td>
    </tr>
    <tr>
        <td align="center">ldt</td>
        <td align="center">输入</td>
        <td align="left">T 的 leading dimension，需满足 ldt >= max(1, k)</td>
    </tr>
    <tr>
        <td align="center">bufferOnDevice</td>
        <td align="center">输入</td>
        <td align="left">device 侧 workspace，字节数不少于 workspaceInBytesOnDevice</td>
    </tr>
    <tr>
        <td align="center">workspaceInBytesOnDevice</td>
        <td align="center">输入/输出</td>
        <td align="left">_bufferSize 时为输出、主入口时为输入；单位是<strong>字节</strong>（与其他四个接口按元素计不同）</td>
    </tr>
    <tr>
        <td align="center">bufferOnHost / workspaceInBytesOnHost</td>
        <td align="center">输入</td>
        <td align="left">host 侧 workspace；本实现<strong>恒为 0 字节</strong>，指针可传 nullptr</td>
    </tr>
    </table>

- 算子约束：
  - <strong>本接口没有 devInfo 输出</strong>：参数错误只通过返回值上报（这与另外四个接口不同）。
  - n 或 k 为 0 时两个 workspace 尺寸均为 0。
  - _bufferSize 不读 V、tau、T，允许传 nullptr。
  - T 的未被引用的那一半<strong>保持调用前的内容</strong>，调用方不应假设它被清零。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 调用示例
- 完整代码示例：[aclsolverXlarft示例](../../test/xlarft/xlarft_test.cpp)
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

        // 构造输入数据（V 列主序、单位对角隐含）
        int64_t n = 8, k = 4;
        int64_t ldv = n, ldt = k;
        size_t wsDev = 0, wsHost = 0;
        aclsolverXlarft_bufferSize(handle, nullptr, ACLSOLVER_DIRECT_FORWARD, ACLSOLVER_STOREV_COLUMNWISE, n, k,
                                   ACLSOLVER_R_32F, nullptr, ldv, ACLSOLVER_R_32F, nullptr,
                                   ACLSOLVER_R_32F, nullptr, ldt, ACLSOLVER_R_32F, &wsDev, &wsHost);

        void *dV = nullptr, *dTau = nullptr, *dT = nullptr, *dWork = nullptr;
        aclrtMalloc(&dV, static_cast<size_t>(ldv) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dTau, static_cast<size_t>(k) * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        aclrtMalloc(&dT, static_cast<size_t>(ldt) * k * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
        if (wsDev > 0) {
            aclrtMalloc(&dWork, wsDev, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        // 调用 aclsolverXlarft
        auto ret = aclsolverXlarft(handle, nullptr, ACLSOLVER_DIRECT_FORWARD, ACLSOLVER_STOREV_COLUMNWISE, n, k,
                                   ACLSOLVER_R_32F, dV, ldv, ACLSOLVER_R_32F, dTau,
                                   ACLSOLVER_R_32F, dT, ldt, ACLSOLVER_R_32F, dWork, wsDev, nullptr, 0);
        CHECK_RET(ret == ACLSOLVER_STATUS_SUCCESS, LOG_PRINT("aclsolverXlarft failed. ERROR: %d\n", ret); return ret);
        aclrtSynchronizeStream(stream);

        aclrtFree(dV); aclrtFree(dTau); aclrtFree(dT); if (dWork != nullptr) { aclrtFree(dWork); }


        // 释放资源
        aclsolverDestroy(handle);
        aclrtDestroyStream(stream);
        aclrtResetDevice(deviceId);
        aclFinalize();

        return 0;
    }
    ```
