# 快速入门：基于ops-solver仓

## 使用须知

本指南旨在帮助您快速上手CANN和`ops-solver`算子仓的使用。为方便快速了解算子开发全流程，将以**cmatinv_batched**算子为实践对象，算子源码统一存放于 `src/<算子名>/` 目录（如 `src/cmatinv_batched/`，包含 host 与 kernel 源文件），以下以 Ascend 950 硬件平台为例。

1. **[环境准备](#环境准备)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐WebIDE或Docker环境**，安装操作简单。

   > **说明**：当前WebIDE或Docker环境默认最新商发版CANN包；如需体验master分支最新能力，可手动安装CANN包，注意软件与源码版本配套。

2. **[编译运行](#一编译运行)**：编译自定义算子包并安装，实现快速调用算子。

3. **[算子开发](#二算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。

4. **[算子调试](#三算子调试)**：掌握算子打印和性能采集方法。

## 环境准备

**[环境部署](zh/install/quick_install.md)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐WebIDE或Docker环境**，安装操作简单。

> **说明**：当前WebIDE或Docker环境默认最新商发版CANN包；如需体验master分支最新能力，可手动安装CANN包，注意软件与源码版本配套。

---

## 一、编译运行

本阶段目的是**快速体验项目标准流程**，验证环境能否成功进行算子源码编译、打包、安装和运行。

### 1. 编译cmatinv_batched算子

环境准备好后（注意软件与源码版本配套），进入环境并访问项目源码根目录，编译指定算子。

通用编译命令格式：`bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>`。以cmatinv_batched算子为例，编译命令如下：

```bash
bash build.sh --pkg --soc=ascend950 --ops=cmatinv_batched
```

若提示如下信息，说明编译成功。
```bash
Self-extractable archive "cann-950-ops-solver_9.0.0_linux-x86_64.run" successfully created.
```
编译成功后，run包存放于项目根目录的build_out目录下。

> **说明**：run包实际命名规则为 `cann-<SOC短名>-ops-solver_<版本>_linux-<架构>.run`，SOC短名由编译时指定的 `--soc` 决定（如 ascend950 对应 `950`、ascend910b 对应 `910b`），架构为当前系统 `uname -m` 输出。

### 2. 安装cmatinv_batched算子包

> **说明**：run包必须通过`--install`参数执行安装，不带参数仅显示帮助信息不会安装。可通过`--install-path=<路径>`指定安装目录，默认安装路径：root用户为`/usr/local/Ascend`，普通用户为`~/Ascend`。
```bash
./build_out/cann-950-ops-solver_9.0.0_linux-x86_64.run --install --quiet
```

### 3. 快速验证：运行算子样例

通用的运行命令格式：`bash build.sh --soc=${soc_version} --ops=<算子名> --run`。

以cmatinv_batched算子为例，运行该样例验证算子功能是否正常。

运行样例前，需先准备测试数据。测试程序从`./test/cmatinv_batched/data/input/A_gm.bin`读取输入数据，请在项目根目录执行以下命令生成（`<n>`为矩阵阶数，`<batchSize>`为批量大小，需与测试用例的调用参数保持一致）：

```bash
python3 test/cmatinv_batched/data/gen_data.py <n> <batchSize>
```

```bash
bash build.sh --soc=ascend950 --ops=cmatinv_batched --run
```
预期输出：打印算子`cmatinv_batched`的计算结果，表明算子已成功部署并正确执行。

> **提示**：若已完成编译且仅需重新验证功能，可直接运行已编译的二进制文件，无需再次编译。编译完成后，测试二进制文件位于`build/test/<算子名>/<算子名>_test`。
>
> 注意：`bash build.sh --run` 在测试结束后会清理生成的测试数据（`test/<算子名>/data/` 下的 input/output/golden），直接运行二进制前需先按上文重新生成（参数需与测试用例的调用参数一致）：
>
> ```bash
> python3 test/cmatinv_batched/data/gen_data.py <n> <batchSize>
> ./build/test/cmatinv_batched/cmatinv_batched_test
> ```

## 二、算子开发

以下为从零开发一个 Solver 算子的完整流程。

## 三、算子调试

### 1. 打印

在Kernel代码中使用`AscendC::PRINTF`进行调试打印：该接口支持打印Scalar类型数据（如整数、字符型、布尔型等），Vector/Cube Kernel 均可用；打印Tensor内容请使用`DumpTensor`，SIMT Kernel（`__simt_vf__`）请使用`AscendC::Simt::PRINTF`。编译运行后可在CANN日志（plog）中查看输出，详见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中"算子调测API"。

### 2. 性能采集

使用msprof工具进行性能采集，具体参考[CANN性能采集文档](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/aclcppdevg/aclcppdevg_0000.html)。

## 四、常见问题排查

### 1. 样例失败如何定位

运行算子样例（`bash build.sh --soc=<soc> --ops=<op> --run` 或直接执行 `build/test/<op>/<op>_test`）失败时，按以下顺序定位：

1. 先看退出时打印的错误码，对照下方"常见错误码"表；
2. 设备类错误先检查环境（见"设备检查清单"）；
3. 打开 CANN 日志重跑，获取设备侧详细报错（见"打开 CANN 日志"）；
4. 仍未定位时，将完整错误码与日志提交到 [Issues](https://gitcode.com/cann/ops-solver/issues)。

**常见错误码**：

| 错误码 | 宏定义 | 含义 | 定位建议 |
| --- | --- | --- | --- |
| 107001 | ACL_ERROR_RT_INVALID_DEVICEID | 设备号无效 | 见下方"设备检查清单"，通常是设备不可见或设备号不存在 |
| 107000 | ACL_ERROR_RT_PARAM_INVALID | 运行时参数无效 | 检查 API 入参 |
| 100000 | ACL_ERROR_INVALID_PARAM | 接口参数非法 | 对照算子文档的"约束"（如 getri 族要求 n ≤ 8192） |
| 507015 | ACL_ERROR_RT_AICORE_EXCEPTION | AI Core 执行异常（kernel 越界等） | 打开 plog 查看设备侧根因；确认输入满足算子约束 |
| 507057 | ACL_ERROR_RT_SUSPECT_REMOTE_ERROR | 疑似远端错误（通常是其他根因的伴随错误） | 在 plog 中找**最早出现**的根因错误（常见伴随 507015） |

**设备检查清单**（107001 等设备类错误）：

- 执行 `npu-smi info`：能否看到芯片？看不到说明驱动未加载或容器未映射设备；
- 容器场景：启动容器时需映射设备，可参考[环境部署](zh/install/quick_install.md)中 docker run 示例的 `--device=/dev/davinci*` 等参数；
- 确认设备号：样例默认使用 device 0，若实际设备号不是 0，可在测试命令后追加设备号（如 `./build/test/cmatinv_batched/cmatinv_batched_test 1`），或通过 `ASCEND_RT_VISIBLE_DEVICES` 指定可见设备；
- 确认驱动与 CANN 包版本配套（`cat /usr/local/Ascend/driver/version.info`）。

### 2. --soc 应该填什么

`--soc` 取值必须与硬件平台匹配，可用 `npu-smi info` 查看芯片型号后对照下表（未指定 `--soc` 时默认 `ascend910b`）：

| 硬件平台 | --soc 取值 |
| --- | --- |
| Atlas A3 训练系列/推理系列（Ascend 910C，型号如 Ascend910_93xx） | `ascend910_93` |
| Atlas A2 训练系列/推理系列（Ascend 910B，型号如 Ascend910Bxx） | `ascend910b` |
| Ascend 950 系列（型号如 Ascend950PR_/Ascend950DT_） | `ascend950` |
| Atlas 200I/500 A2 推理产品（Ascend 310P） | `ascend310p` |

> **说明**：初代 Ascend 910A（Ascend910A/PremiumA/ProA 等）不在当前支持列表中；完整支持列表见 `build.sh` 中 `SUPPORT_COMPUTE_UNIT_SHORT`。

### 3. 打开 CANN 日志

样例只打印错误码、无法定位时，打开日志后重跑：

```bash
export ASCEND_GLOBAL_LOG_LEVEL=1        # 0=DEBUG 更详细，1=INFO，2=WARN，3=ERROR
export ASCEND_SLOG_PRINT_TO_STDOUT=1    # 日志直接打印到终端
./build/test/<op>/<op>_test
```

> **说明**：设置 `ASCEND_SLOG_PRINT_TO_STDOUT=1` 后日志只输出到终端，不再写入 plog 文件；如需查看落盘的 plog 文件，请去掉该变量后重跑。

plog 文件按类别落盘在 `~/ascend/log` 下（root 用户为 `/root/ascend/log`），按时间排序查看最新文件，重点关注**第一条** ERROR 记录：

- `run/plog`：运行日志，默认记录；
- `debug/plog`：调试日志，记录等级由 `ASCEND_GLOBAL_LOG_LEVEL` 控制（如 1 记录 INFO 及以上，0 记录 DEBUG 及以上）。

---

## 算子开发实例

我们以CmatinvBatched算子（批量复数矩阵求逆算子）为例，说明基于Solver库开发一个算子的主要流程。


## 算子功能

计算批量复数矩阵的逆矩阵。

**计算公式：**
$
A^{-1}A = I
$

其中 $A$ 为 $n \times n$ 阶非奇异复数方阵，$I$ 为 $n$ 阶单位矩阵。

**参数说明：**
- `n`: 单个矩阵A的行数/列数（方阵）
- `A`: 输入的批量复数矩阵，shape为 [batchSize, n, n]
- `Ainv`: 输出的逆矩阵，shape为 [batchSize, n, n]
- `batchSize`: 批量矩阵的数量

**算子约束：**
- n ≤ 256
- batchSize ≤ 3000

**示例：**
输入"A"为：
```
[2-2i, 1-i, 1-i, 1-i
 1-i, 2-2i, 1-i, 1-i
 1-i, 1-i, 2-2i, 1-i
 1-i, 1-i, 1-i, 2-2i]

[3-3i, 1-i, 1-i, 1-i
 1-i, 3-3i, 1-i, 1-i
 1-i, 1-i, 3-3i, 1-i
 1-i, 1-i, 1-i, 3-3i]
```

输入"n"为：4
输入"batchSize"为：2

调用"aclsolverCmatinvBatched"算子后，输出"Ainv"为：
```
[0.4+0.4i, -0.1-0.1i, -0.1-0.1i, -0.1-0.1i
 -0.1-0.1i, 0.4+0.4i, -0.1-0.1i, -0.1-0.1i
 -0.1-0.1i, -0.1-0.1i, 0.4+0.4i, -0.1-0.1i
 -0.1-0.1i, -0.1-0.1i, -0.1-0.1i, 0.4+0.4i]

[0.208+0.208i, -0.0417-0.0417i, -0.0417-0.0417i, -0.0417-0.0417i
 -0.0417-0.0417i, 0.208+0.208i, -0.0417-0.0417i, -0.0417-0.0417i
 -0.0417-0.0417i, -0.0417-0.0417i, 0.208+0.208i, -0.0417-0.0417i
 -0.0417-0.0417i, -0.0417-0.0417i, -0.0417-0.0417i, 0.208+0.208i]
```

## 修改文件

- 在`include`目录下的`cann_ops_solver.h`文件中新增以下算子接口原型：

    ```c++
    aclError aclsolverCmatinvBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A,
                                     const int64_t lda, std::complex<float> *Ainv,
                                     const int64_t lda_inv, int32_t *info,
                                     int64_t batchSize);
    ```

## 新增文件

- 在`src`目录下新增目录`cmatinv_batched`，该目录下主要存放CmatinvBatched算子的源代码，目录结构如下：

    ```txt
    cmatinv_batched
    ├── cmatinv_batched_host.cpp
    └── cmatinv_batched_kernel.cpp
    ```

- 在`test`目录下新增目录`cmatinv_batched`，该目录下主要存放CmatinvBatched算子的测试代码，目录结构如下：

    ```txt
    cmatinv_batched
    ├── data
    |    ├── gen_data.py
    |    └── verify_result.py
    ├── cmatinv_batched_test.cpp
    ├── CMakeLists.txt
    └── README.md
    ```

- 在`docs/zh`目录下新增算子说明文档`cmatinv_batched.md`。

## Solver算子实现

CmatinvBatched算子实现主要包括：host侧实现和kernel侧实现。其中算子采用Kernel直调方式实现，Kernel直调算子开发流程可访问[Kernel直调算子开发-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_0056.html)查看。

### Host侧开发

host侧负责参数校验、数据准备、Tiling计算和内存管理。Tiling开发的核心概念：`TilingData`、`Workspace`等，可访问[术语表-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00013.html)查看。

#### cmatinv_batched_host.cpp

文件路径：`src/cmatinv_batched/cmatinv_batched_host.cpp`
主要功能：参数校验、数据准备、Tiling计算和内存管理。

```c++
#include <cstdint>
#include <iostream>
#include <complex>
#include <vector>
#include <algorithm>
#include <iterator>
#include <securec.h>
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include "cann_ops_solver.h"
#include "../utils/assert.h"

#define GM_ADDR uint8_t*

// 声明kernel函数
extern void cmatinv_batched_kernel_do(GM_ADDR dA, GM_ADDR dUniReal,
                                      GM_ADDR dUniImag, GM_ADDR dOffset,
                                      GM_ADDR dAinv, GM_ADDR workSpace,
                                      GM_ADDR tilingGm, uint32_t numBlocks,
                                      void *stream);

// 常量定义
static constexpr uint32_t COMPLEX_ELENUM = 2;
static constexpr uint32_t MAX_CORE_CNT = 40;
static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;
static constexpr int32_t MATRIX_SHAPE_LIMIT = 32;
static constexpr int32_t MAX_MATRIX_SHAPE = 256;
static constexpr int32_t MAX_MATRIX_BATCH = 3000;
static constexpr int64_t DTYPE_COMPLEX64 = 1;
static constexpr int ELEMENT_NUM_PER_BLOCK = 8;
static constexpr int ELEMENT_NUM_PER_REPEAT = 64;
static constexpr uint32_t BASE_COMPLEX64_NUM = 8192; // 64kb / 8b
static constexpr uint32_t MAX_COMPUTE_NUM = 128; // max compute num per repeat

// Tiling数据结构
struct CmatinvBatchedTilingData {
    uint32_t dtype;              // 数据类型，1表示COMPLEX64
    uint32_t n;                  // 矩阵维度
    uint32_t startOffset[40];    // 每个核心的起始偏移量
    uint32_t calNum[40];         // 每个核心需要计算的矩阵数量
};

// 计算Tiling数据，将批量矩阵均匀分配到多个AI核心上并行处理
CmatinvBatchedTilingData CalTilingData(uint32_t vecCoreNum, uint32_t dtype, uint32_t n, uint32_t batchSize)
{
    CmatinvBatchedTilingData tilingData;
    // 限制核心数最多为40
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;
    vecCoreNum = (vecCoreNum == 0) ? 1 : vecCoreNum;

    // 初始化tiling数据
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }

    // 计算每个核心处理的矩阵数量
    uint32_t matPerCore = batchSize / vecCoreNum;
    uint32_t remainMatNum = batchSize % vecCoreNum;

    if (matPerCore == 0) {
        // 如果矩阵数量少于核心数，每个矩阵分配给一个核心
        for (uint32_t i = 0; i < remainMatNum; i++) {
            tilingData.calNum[i] = 1;
            tilingData.startOffset[i] = i * n * n * COMPLEX_ELENUM;
        }
    } else {
        // 均匀分配矩阵到各个核心
        uint32_t currComputeNum;
        uint32_t currOffset = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainMatNum) {
                currComputeNum = matPerCore + 1;
            } else {
                currComputeNum = matPerCore;
            }
            tilingData.calNum[i] = currComputeNum;
            tilingData.startOffset[i] = currOffset;
            currOffset += currComputeNum * n * n * COMPLEX_ELENUM;
        }
    }
    tilingData.dtype = dtype;
    tilingData.n = n;

    return tilingData;
}

aclError aclsolverCmatinvBatched(aclsolverHandle_t handle, const int64_t n, std::complex<float> *A,
                                 const int64_t lda, std::complex<float> *Ainv,
                                 const int64_t lda_inv, int32_t *info,
                                 int64_t batchSize) {
    // 获取handle绑定的stream
    aclrtStream stream = nullptr;
    if (handle != nullptr) {
        aclsolverGetStream(handle, &stream);
    }

    if (n > MATRIX_SHAPE_LIMIT) {
        LOG_PRINT("CmatinvBatched only supports n ≤ 32. For n > 32, use CgetriBatched instead.\n");
        return aclsolverCgetriBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
    }

    // 获取设备核心数量，此处获取cube核，GetCoreNumAiv可以获取vector核
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    uint32_t numBlocks = 0;
    if (ascendcPlatform != nullptr) {
        numBlocks = ascendcPlatform->GetCoreNumAic();
    }

    // 参数校验
    SOLVER_ECHECK(n > 0 && batchSize > 0, "CmatinvBatched get n <= 0 || batchSize <= 0.", ACL_ERROR_INVALID_PARAM);
    SOLVER_ECHECK(n <= MAX_MATRIX_SHAPE && batchSize <= MAX_MATRIX_BATCH,
                  "CmatinvBatched get n <= 256 || batchSize <= 3000.",
                  ACL_ERROR_INVALID_PARAM);

    // 算子kernel计算所需辅助参数计算
    uint32_t N = static_cast<uint32_t>(n);
    uint32_t alignedN = (N + (ELEMENT_NUM_PER_BLOCK - 1)) / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    uint32_t computeNumPerRepeat = BASE_COMPLEX64_NUM / (N * alignedN);
    computeNumPerRepeat = std::min(computeNumPerRepeat, MAX_COMPUTE_NUM);
    computeNumPerRepeat = computeNumPerRepeat / ELEMENT_NUM_PER_BLOCK * ELEMENT_NUM_PER_BLOCK;

    // 准备单位矩阵数据
    std::vector<float> uniMatRealData(N * alignedN, 0.0f);
    std::vector<float> uniMatImagData(N * alignedN, 0.0f);
    for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
        for (uint32_t colIdx = 0; colIdx < alignedN; colIdx++) {
            if (rowIdx == colIdx) {
                uniMatRealData[alignedN * rowIdx + colIdx] = 1.0f;
            } else {
                uniMatRealData[alignedN * rowIdx + colIdx] = 0.0f;
            }
            uniMatImagData[alignedN * rowIdx + colIdx] = 0.0f;
        }
    }

    std::vector<float> uniBatchRealData(N * alignedN * computeNumPerRepeat, 0.0f);
    std::vector<float> uniBatchImagData(N * alignedN * computeNumPerRepeat, 0.0f);
    errno_t rc;
    for (uint32_t i = 0; i < computeNumPerRepeat; i++) {
        rc = memcpy_s(static_cast<float *>(uniBatchRealData.data() + N * alignedN * i),
            N * alignedN * sizeof(float), uniMatRealData.data(), N * alignedN * sizeof(float));
        SOLVER_ECHECK(rc == EOK, "Repeat uniMatRealData by memcpy_s failed.", ACL_ERROR_INTERNAL_ERROR);
        rc = memcpy_s(static_cast<float *>(uniBatchImagData.data() + N * alignedN * i),
            N * alignedN * sizeof(float), uniMatImagData.data(), N * alignedN * sizeof(float));
        SOLVER_ECHECK(rc == EOK, "Repeat uniMatImagData by memcpy_s failed.", ACL_ERROR_INTERNAL_ERROR);
    }

    uint32_t offsetSize = N * alignedN * computeNumPerRepeat * 2;
    offsetSize = (offsetSize + (ELEMENT_NUM_PER_REPEAT - 1)) / ELEMENT_NUM_PER_REPEAT * ELEMENT_NUM_PER_REPEAT;
    std::vector<uint32_t> offsetData(offsetSize, 0);
    uint32_t k = 0;
    uint32_t realBase = 0;
    uint32_t imagBase = 32 * 1024;
    for (uint32_t batchIdx = 0; batchIdx < computeNumPerRepeat; batchIdx++) {
        for (uint32_t rowIdx = 0; rowIdx < N; rowIdx++) {
            for (uint32_t colIdx = 0; colIdx < N; colIdx++) {
                offsetData[k++] = realBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
                offsetData[k++] = imagBase + sizeof(uint32_t) * (batchIdx * alignedN * N + alignedN * rowIdx + colIdx);
            }
        }
    }

    // 分配设备内存
    uint8_t *d_A = nullptr;
    uint8_t *d_Ainv = nullptr;
    uint8_t *d_info = nullptr;
    uint8_t *d_UniReal = nullptr;
    uint8_t *d_UniImag = nullptr;
    uint8_t *d_Offset = nullptr;

    auto sizeA = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeAinv = batchSize * n * n * sizeof(std::complex<float>);
    auto sizeinfo = batchSize * sizeof(int32_t);
    auto sizeUniReal = N * alignedN * computeNumPerRepeat * sizeof(float);
    auto sizeUniImag = N * alignedN * computeNumPerRepeat * sizeof(float);
    auto sizeOffset = offsetSize * sizeof(uint32_t);

    CHECK_ACLRT(aclrtMalloc((void **)&d_A, sizeA, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_Ainv, sizeAinv, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_info, sizeinfo, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_UniReal, sizeUniReal, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_UniImag, sizeUniImag, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMalloc((void **)&d_Offset, sizeOffset, ACL_MEM_MALLOC_HUGE_FIRST));

    // 拷贝数据到Device
    uint8_t *hostA = reinterpret_cast<uint8_t *>(A);
    uint8_t *hostAinv = reinterpret_cast<uint8_t *>(Ainv);
    uint8_t *hostinfo = reinterpret_cast<uint8_t *>(info);
    uint8_t *hostUniReal = reinterpret_cast<uint8_t *>(uniBatchRealData.data());
    uint8_t *hostUniImag = reinterpret_cast<uint8_t *>(uniBatchImagData.data());
    uint8_t *hostOffset = reinterpret_cast<uint8_t *>(offsetData.data());

    CHECK_ACLRT(aclrtMemcpy(d_A, sizeA, hostA, sizeA, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_Ainv, sizeAinv, hostAinv, sizeAinv, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_info, sizeinfo, hostinfo, sizeinfo, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_UniReal, sizeUniReal, hostUniReal, sizeUniReal, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_UniImag, sizeUniImag, hostUniImag, sizeUniImag, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACLRT(aclrtMemcpy(d_Offset, sizeOffset, hostOffset, sizeOffset, ACL_MEMCPY_HOST_TO_DEVICE));

    // 计算Tiling数据并拷贝到Device
    CmatinvBatchedTilingData tiling = CalTilingData(numBlocks, DTYPE_COMPLEX64, n, batchSize);
    uint8_t *tilingDevice = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&tilingDevice, sizeof(CmatinvBatchedTilingData), ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACLRT(aclrtMemcpy(tilingDevice, sizeof(CmatinvBatchedTilingData),
                            &tiling, sizeof(CmatinvBatchedTilingData),
                            ACL_MEMCPY_HOST_TO_DEVICE));

    // 分配workSpace空间
    uint8_t *workSpace = nullptr;
    CHECK_ACLRT(aclrtMalloc((void **)&workSpace, WORKSPACE_SIZE, ACL_MEM_MALLOC_HUGE_FIRST));

    // 固定写法，调用kernel
    cmatinv_batched_kernel_do(d_A, d_UniReal, d_UniImag, d_Offset, d_Ainv, workSpace, tilingDevice, numBlocks, stream);
    aclrtSynchronizeStream(stream);

    // 拷贝结果回host
    CHECK_ACLRT(aclrtMemcpy(hostAinv, sizeAinv, d_Ainv, sizeAinv, ACL_MEMCPY_DEVICE_TO_HOST));

    // 释放设备内存
    aclrtFree(d_A);
    aclrtFree(d_UniReal);
    aclrtFree(d_UniImag);
    aclrtFree(d_Offset);
    aclrtFree(d_Ainv);
    aclrtFree(workSpace);
    aclrtFree(tilingDevice);

    return ACL_SUCCESS;
}
```

### Kernel侧开发
kernel侧负责实际的矩阵求逆计算。Kernel相关的`Compute`、`CopyIn`、`CopyOut`等概念，可访问[术语表-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/opdevg/Ascendcopdevg/atlas_ascendc_10_00013.html)查看。

### cmatinv_batched_kernel.cpp

文件路径：`src/cmatinv_batched/cmatinv_batched_kernel.cpp`
主要功能：核函数入口，解析Tiling数据并调用算子类; 定义CmatinvBatchedAIV模板类，实现矩阵求逆的核心算法。

#### 核函数入口

```c++
__global__ __aicore__ void
cmatinv_batched_kernel(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
                       GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace,
                       GM_ADDR tilingGm) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    // 解析数据类型
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    uint32_t dtype = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf));

    // 根据数据类型选择不同的模板实例
    if (dtype > 0) {
        CmatinvBatched::CmatinvBatchedAIV<float> op;
        op.Init(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
        op.Process();
    } else {
        CmatinvBatched::CmatinvBatchedAIV<half> op;
        op.Init(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
        op.Process();
    }
}

// Kernel调用封装函数
void cmatinv_batched_kernel_do(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
                               GM_ADDR dOffset, GM_ADDR dAinv,
                               GM_ADDR workSpace, GM_ADDR tilingGm,
                               uint32_t numBlocks, void *stream) {
    cmatinv_batched_kernel<<<numBlocks, nullptr, stream>>>(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
}
```

#### 算子核函数实现

```c++
#include <cstdint>
#include "kernel_operator.h"

using namespace AscendC;

namespace CmatinvBatched {

constexpr uint32_t BYTENUM_BLOCK = 32;
constexpr uint32_t BYTENUM_REPEAT = 256;

constexpr uint32_t COMPLEX_ELENUM = 2;
constexpr uint32_t MAX_CORE_CNT = 40;

constexpr uint32_t ELENUM_BLOCK_FP32 = 8;
constexpr uint32_t ELENUM_REPEAT_FP32 = 64;
constexpr uint32_t ELENUM_BLOCK_FP16 = 16;
constexpr uint32_t ELENUM_REPEAT_FP16 = 128;

constexpr uint32_t BASE_SIZE_ONE_ITERATION = 64 * 1024; // 64 kb
constexpr uint32_t BASE_COMPLEX64_NUM = BASE_SIZE_ONE_ITERATION / sizeof(float) / COMPLEX_ELENUM; // 8192
constexpr uint32_t MAX_BATCH_NUM = 128;
constexpr uint32_t MAX_STRIDE_NUM = 255;
constexpr uint32_t MAX_REPEAT_NUM = 255;
constexpr uint32_t MAX_UBUF_SIZE = 192 * 1024; // 192kb

template <typename T>
class CmatinvBatchedAIV {
public:
    __aicore__ inline CmatinvBatchedAIV(){};
    __aicore__ inline void Init(GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
        GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace, GM_ADDR tilingGm);

    __aicore__ inline void InitUbuf();
    __aicore__ inline void InitTempUbuf(LocalTensor<float> ubufLocal, uint32_t allocOffset);
    __aicore__ inline void ParseTilingData(GM_ADDR tilingGm);

    __aicore__ inline void Process();
    __aicore__ inline void SingleProcess();

    __aicore__ inline void CopyCmatBatchGmToUb(LocalTensor<T> dst,
        GlobalTensor<T> src, uint32_t matColSize, uint32_t batchSize);
    __aicore__ inline void CopyUniMatBatchGmToUb(LocalTensor<float> dst, GlobalTensor<float> src,
        uint32_t matColSize, uint32_t batchSize, uint32_t isComplex);
    __aicore__ inline void CopyVecGmToUbUint32(LocalTensor<uint32_t> dst,
        GlobalTensor<uint32_t> src, uint32_t len);
    __aicore__ inline void CopyVecGmToUb(LocalTensor<float> dst,
        GlobalTensor<float> src, uint32_t len);
    __aicore__ inline void CopyVecUbToGm(GlobalTensor<T> dst,
        LocalTensor<T> src, uint32_t len);

    __aicore__ inline void ComputeComplexScalarOffset();
    __aicore__ inline void PrepareDataForHalf(uint32_t matColSize, uint32_t batchSize);
    __aicore__ inline void SeparateComplexMat();
    __aicore__ inline void CollectScalarSingle(LocalTensor<float> collectDst, uint32_t sOffset);
    __aicore__ inline void ComplexScalarDiv();

    __aicore__ inline void RealScalarVecBatchMul(LocalTensor<float> dst,
        LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset);
    __aicore__ inline void RealScalarVecBatchSub(LocalTensor<float> dst,
        LocalTensor<float> src, uint32_t oneMatOffset);
    __aicore__ inline void RealVecScalarBatchDiv(LocalTensor<float> dst,
        LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset);

    __aicore__ inline void ComplexScalarVecBatchMul(LocalTensor<float> matReal,
        LocalTensor<float> matImag, uint32_t oneMatOffset);
    __aicore__ inline void ComputeAndUpdateAj(LocalTensor<float> AjReal,
        LocalTensor<float> AjImag, uint32_t oneMatOffset);
    __aicore__ inline void ComplexVecScalarBatchDiv(LocalTensor<float> vecReal,
        LocalTensor<float> vecImag, uint32_t oneMatOffset);
    __aicore__ inline void ComplexCombinationAndCopyout(uint32_t outOffset, uint32_t batchSize);

private:
    static constexpr bool IS_FLOAT = IsSameType<T, float>::value;  // half or float

    TPipe pipe;

    GlobalTensor<T> inAGM;
    GlobalTensor<float> uniRealGM;
    GlobalTensor<float> uniImagGM;
    GlobalTensor<uint32_t> offsetGM;
    GlobalTensor<T> outGM;
    GlobalTensor<float> workGM;

    TBuf<QuePosition::VECCALC> uBuf;

    // main buf
    LocalTensor<float> invSepLocal;
    LocalTensor<float> invRealLocal;
    LocalTensor<float> invImagLocal;
    LocalTensor<float> matSepLocal;
    LocalTensor<float> matRealLocal;
    LocalTensor<float> matImagLocal;
    LocalTensor<float> matInLocal;

    // temp buf
    LocalTensor<float> preLocal;
    LocalTensor<float> preRealLocal;
    LocalTensor<float> preImagLocal;
    LocalTensor<float> afterLocal;
    LocalTensor<float> afterRealLocal;
    LocalTensor<float> afterImagLocal;
    LocalTensor<float> resLocal;
    LocalTensor<float> resRealLocal;
    LocalTensor<float> resImagLocal;

    LocalTensor<float> scalarTempLocal;
    LocalTensor<float> scalarBrcbLocal;
    LocalTensor<float> scalarBrcbRealLocal;
    LocalTensor<float> scalarBrcbImagLocal;
    LocalTensor<float> scalarBrcbAugLocal;

    LocalTensor<float> updateTempRealLocal;
    LocalTensor<float> updateTempImagLocal;
    LocalTensor<float> updateTempAugRealLocal;
    LocalTensor<float> updateTempAugImagLocal;

    LocalTensor<uint32_t> collectScalarOffsetLocal;
    LocalTensor<uint32_t> complexCombineOffsetLocal;

    uint32_t n = 0;
    uint32_t offset = 0;
    uint32_t calNum = 0;

    uint32_t alignedComplexMatSize = 0;
    uint32_t batchNumPerRepeat = 0;
    uint32_t oneMatUbOffsetFp32 = 0;

    int32_t coreIdx;
};

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::Init(
    GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag,
    GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    SetMaskNorm();
    SetAtomicNone();

    this->coreIdx = GetBlockIdx();

    ParseTilingData(tilingGm);

    this->inAGM.SetGlobalBuffer((__gm__ T *)dA);
    this->uniRealGM.SetGlobalBuffer((__gm__ float *)dUniReal);
    this->uniImagGM.SetGlobalBuffer((__gm__ float *)dUniImag);
    this->offsetGM.SetGlobalBuffer((__gm__ uint32_t *)dOffset);
    this->outGM.SetGlobalBuffer((__gm__ T *)dAinv);

    this->workGM.SetGlobalBuffer((__gm__ float *)workSpace);

    InitUbuf();
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::InitUbuf()
{
    /*
        can process 64kb one time,
        can be 8 * 32 * 32 ~ 128 * 8 * 8 ~ 512 * 2 * 8 (complex64 num)
        the batch size ranges [8, 512]
        the size of matrix ranges [2, 32]
    */
    this->alignedComplexMatSize = (this->n + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;

    this->batchNumPerRepeat = BASE_COMPLEX64_NUM / (this->n * this->alignedComplexMatSize); // [8, 512]
    if (this->batchNumPerRepeat > MAX_BATCH_NUM) {
        this->batchNumPerRepeat = MAX_BATCH_NUM;
    }
    this->batchNumPerRepeat = this->batchNumPerRepeat / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;

    // offset of one matrix in ub, mat_size x matAlignedSize, 8 * 8 = 64
    this->oneMatUbOffsetFp32 = this->n * this->alignedComplexMatSize;

    this->pipe.InitBuffer(this->uBuf, MAX_UBUF_SIZE);
    LocalTensor<float> ubufLocal = uBuf.Get<float>();

    // init main buf
    uint32_t allocOffset = 0;
    this->invSepLocal = ubufLocal[allocOffset]; // 64kb
    this->invRealLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->invImagLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->matSepLocal = ubufLocal[allocOffset]; // 64kb
    this->matRealLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->matImagLocal = ubufLocal[allocOffset]; // 32kb
    allocOffset += BASE_COMPLEX64_NUM;
    this->matInLocal = ubufLocal[allocOffset]; // 64kb

    // reuse mat sep addr for saving complex combination offset
    this->complexCombineOffsetLocal = this->matSepLocal.template ReinterpretCast<uint32_t>();

    // init temp buf, reuse mat in buf
    InitTempUbuf(ubufLocal, allocOffset);
}

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::InitTempUbuf(LocalTensor<float> ubufLocal, uint32_t allocOffset)
{
    uint32_t alignedBatchSize = (this->batchNumPerRepeat +
        ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32; // 256 bytes

    this->collectScalarOffsetLocal = ubufLocal[allocOffset].template ReinterpretCast<uint32_t>();
    allocOffset += alignedBatchSize * COMPLEX_ELENUM; // 512 bytes
    this->preLocal = ubufLocal[allocOffset]; // 1kb ~ 4kb, 1kb
    this->preRealLocal = ubufLocal[allocOffset]; // num of alignedBatchSize scalar
    allocOffset += alignedBatchSize; // 256 bytes
    this->preImagLocal = ubufLocal[allocOffset]; // num of alignedBatchSize scalar
    allocOffset += alignedBatchSize; // 256 bytes

    this->afterLocal = ubufLocal[allocOffset]; // 1kb ~ 4kb, 1kb
    this->afterRealLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes
    this->afterImagLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes

    this->resLocal = ubufLocal[allocOffset]; // 1kb ~ 4kb, 1kb
    this->resRealLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes
    this->resImagLocal = ubufLocal[allocOffset];
    allocOffset += alignedBatchSize; // 256 bytes
    this->scalarTempLocal = ubufLocal[allocOffset]; // 512bytes ~ 2kb, 512 bytes
    allocOffset += alignedBatchSize; // 256 bytes

    uint32_t brcbEleNum = alignedBatchSize * ELENUM_BLOCK_FP32 * COMPLEX_ELENUM; // 64 * 32 * 2 = 4kb
    this->scalarBrcbLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    this->scalarBrcbRealLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    allocOffset += brcbEleNum; // 4kb
    this->scalarBrcbImagLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    allocOffset += brcbEleNum; // 4kb
    this->scalarBrcbAugLocal = ubufLocal[allocOffset]; // 4 ~ 32 kb, 4kb
    allocOffset += brcbEleNum; // 4kb

    uint32_t updateTempEleNum = this->alignedComplexMatSize * this->batchNumPerRepeat; // 8kb
    this->updateTempRealLocal = ubufLocal[allocOffset]; // 4kb
    allocOffset += updateTempEleNum; // 8kb
    this->updateTempImagLocal = ubufLocal[allocOffset]; // 4kb
    allocOffset += updateTempEleNum; // 8kb
    this->updateTempAugRealLocal = ubufLocal[allocOffset]; // 4kb
    this->updateTempAugImagLocal = ubufLocal[allocOffset]; // 4kb
}

template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ParseTilingData(GM_ADDR tilingGm)
{
    // skip dtype
    uint32_t locId = 1;
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    this->n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t))); // num of complex elements
    locId++;
    this->offset = (*(
        __gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t) + sizeof(uint32_t) * this->coreIdx));
    locId += MAX_CORE_CNT;
    this->calNum = (*(
        __gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + locId * sizeof(uint32_t) + sizeof(uint32_t) * this->coreIdx));
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::Process()
{
    auto eventId = EVENT_ID0;
    uint32_t overallRepeatTime = this->calNum / this->batchNumPerRepeat;
    uint32_t remainBatchNum = this->calNum % this->batchNumPerRepeat;
    if (remainBatchNum > 0) {
        overallRepeatTime++;
    }

    uint32_t batchOffset = this->batchNumPerRepeat * this->n * this->n * COMPLEX_ELENUM;
    uint32_t currentBatchSize = this->batchNumPerRepeat;
    uint32_t currOffset = this->offset;
    for (uint32_t i = 0; i < overallRepeatTime; i++) {
        SetFlag<HardEvent::MTE3_MTE2>(eventId);
        WaitFlag<HardEvent::MTE3_MTE2>(eventId);

        if (remainBatchNum > 0 && i == overallRepeatTime - 1) {
            currentBatchSize = remainBatchNum;
        }

        // copy A batch
        if constexpr (IS_FLOAT) {
            LocalTensor<T> inLocal = this->matInLocal.template ReinterpretCast<T>();
            CopyCmatBatchGmToUb(inLocal, this->inAGM[currOffset], this->n * COMPLEX_ELENUM, currentBatchSize);
        } else {
            LocalTensor<T> inLocal = this->matSepLocal.template ReinterpretCast<T>();
            CopyCmatBatchGmToUb(inLocal, this->inAGM[currOffset], this->n * COMPLEX_ELENUM, currentBatchSize);

            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);

            PrepareDataForHalf(this->n * COMPLEX_ELENUM, currentBatchSize);
        }

        // uni real part, uni imag part
        CopyUniMatBatchGmToUb(this->invRealLocal, this->uniRealGM, this->alignedComplexMatSize, currentBatchSize, 0);
        CopyUniMatBatchGmToUb(this->invImagLocal, this->uniImagGM, this->alignedComplexMatSize, currentBatchSize, 0);

        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        SingleProcess();

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        // copy out
        ComplexCombinationAndCopyout(currOffset, currentBatchSize);
        currOffset += batchOffset;
    }
    PipeBarrier<PIPE_ALL>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::SingleProcess()
{
    // complex separation
    SeparateComplexMat();
    PipeBarrier<PIPE_ALL>();

    ComputeComplexScalarOffset();
    PipeBarrier<PIPE_ALL>();
    for (uint16_t elementIdx = 0; elementIdx < this->n; elementIdx++) {
        // iterate diagonal elements
        // collect diagonal element, e.g., aii
        uint32_t preOffset = elementIdx * this->alignedComplexMatSize + elementIdx;
        CollectScalarSingle(this->preLocal, preOffset);

        uint32_t eleOffset = elementIdx * this->alignedComplexMatSize;
        LocalTensor<float> tmpAiReal = this->matRealLocal[eleOffset];
        LocalTensor<float> tmpAiImag = this->matImagLocal[eleOffset];
        LocalTensor<float> tmpAiInvReal = this->invRealLocal[eleOffset];
        LocalTensor<float> tmpAiInvImag = this->invImagLocal[eleOffset];

        for (uint16_t rowIdx = 0; rowIdx < this->n; rowIdx++) {
            // iterate elements in i-th column except aii
            if (rowIdx == elementIdx) {
                continue;
            }

            uint32_t rowOffset = rowIdx * this->alignedComplexMatSize;
            LocalTensor<float> tmpAjReal = this->matRealLocal[rowOffset];
            LocalTensor<float> tmpAjImag = this->matImagLocal[rowOffset];
            LocalTensor<float> tmpAjInvReal = this->invRealLocal[rowOffset];
            LocalTensor<float> tmpAjInvImag = this->invImagLocal[rowOffset];

            // suppose the collected element is aji
            uint32_t afterOffset = rowIdx * this->alignedComplexMatSize + elementIdx;
            CollectScalarSingle(this->afterLocal, afterOffset);
            PipeBarrier<PIPE_V>();

            ComplexScalarDiv();
            PipeBarrier<PIPE_V>();

            // compute and update A[j,:] = A[j,:] - (aji / aii) * A[i,:]
            // (aji / aii) * A[i,:]
            ComplexScalarVecBatchMul(tmpAiReal, tmpAiImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            ComputeAndUpdateAj(tmpAjReal, tmpAjImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            // update inv
            ComplexScalarVecBatchMul(tmpAiInvReal, tmpAiInvImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();

            ComputeAndUpdateAj(tmpAjInvReal, tmpAjInvImag, this->oneMatUbOffsetFp32);
            PipeBarrier<PIPE_V>();
        }

        // update A[i,:] = A[i,:] / aii
        ComplexVecScalarBatchDiv(tmpAiReal, tmpAiImag, this->oneMatUbOffsetFp32);
        PipeBarrier<PIPE_V>();

        ComplexVecScalarBatchDiv(tmpAiInvReal, tmpAiInvImag, this->oneMatUbOffsetFp32);
        PipeBarrier<PIPE_V>();
    }
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyCmatBatchGmToUb(LocalTensor<T> dst,
    GlobalTensor<T> src, uint32_t matColSize, uint32_t batchSize)
{
    uint32_t matRowSize = this->n;
    uint16_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(T);
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    if constexpr (IS_FLOAT) {
        uint32_t paddingBlockNum = (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
        if (paddingBlockNum % COMPLEX_ELENUM != 0) {
            dstStride = 1;
        }
    }

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyUniMatBatchGmToUb(LocalTensor<float> dst, GlobalTensor<float> src,
    uint32_t matColSize, uint32_t batchSize, uint32_t isComplex)
{
    /*
        GM matrix shape:
            | a11, a12, ..., a1n|
            | a21, a22, ..., a2n|
            | ..., ..., ..., ...|
            | an1, ..., ..., ann|
        UB matrix shape:
            | a11, a12, ..., a1n, 0, 0, ..., 0| // padding
            | a21, a22, ..., a2n, 0, 0, ..., 0|
            | ..., ..., ..., ..., 0, 0, ..., 0|
            | an1, ..., ..., ann, 0, 0, ..., 0|
    */
    // copy matrix row by row

    uint32_t matRowSize = this->n;
    uint16_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(float);
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    uint32_t paddingBlockNum = (matColSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;
    if (isComplex && (paddingBlockNum % COMPLEX_ELENUM != 0)) {
        dstStride = 1;
    }

    DataCopyExtParams copyParams{blockCnt, blockLen, srcStride, dstStride, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyVecGmToUbUint32(LocalTensor<uint32_t> dst,
    GlobalTensor<uint32_t> src, uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(uint32_t);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPadExtParams<uint32_t> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyVecGmToUb(LocalTensor<float> dst,
    GlobalTensor<float> src, uint32_t len)
{
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(float);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(dst, src, copyParams, padParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CopyVecUbToGm(GlobalTensor<T> dst,
    LocalTensor<T> src, uint32_t len) {
    uint16_t blockCnt = 1;
    uint32_t blockLen = len * sizeof(T);

    DataCopyExtParams copyParams{blockCnt, blockLen, 0, 0, 0};
    DataCopyPad(dst, src, copyParams);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComputeComplexScalarOffset()
{
    LocalTensor<uint32_t> offsetLocal = this->collectScalarOffsetLocal;
    uint32_t oneMatOffset = this->oneMatUbOffsetFp32; // offset of one matrix in ub, matSize x matAlignedSize
    uint32_t batchSize = this->batchNumPerRepeat;
    uint32_t imagOffset = BASE_SIZE_ONE_ITERATION / COMPLEX_ELENUM; // 32kb

    uint32_t alignedBatchSize = (batchSize + ELENUM_REPEAT_FP32 - 1)
        / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32; // 128

    // scalar real part
    for (uint32_t j = 0; j < alignedBatchSize; j++) {
        uint32_t offsetVal = 0;
        if (j < batchSize) {
            offsetVal = oneMatOffset * j * sizeof(float); // real part
        } else {
            offsetVal = 0; // useless num, set to 0
        }
        offsetLocal.SetValue(j, offsetVal);
    }
    // scalar imag part
    for (uint32_t i = 0; i < alignedBatchSize; i++) {
        uint32_t offsetVal = 0;
        if (i < batchSize) {
            offsetVal = sizeof(float) * (i * oneMatOffset) + imagOffset; // imag part
        } else {
            offsetVal = 0; // useless num, set to 0
        }
        offsetLocal.SetValue(alignedBatchSize + i, offsetVal);
    }
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::PrepareDataForHalf(uint32_t matColSize, uint32_t batchSize)
{
    LocalTensor<half> src = this->matSepLocal.template ReinterpretCast<half>();
    LocalTensor<float> dst = this->matInLocal;

    uint32_t matRowSize = this->n;
    uint32_t blockCnt = matRowSize * batchSize;
    uint32_t blockLen = matColSize * sizeof(half);
    uint32_t paddingBlockNum = (blockLen + BYTENUM_BLOCK - 1) / BYTENUM_BLOCK;
    uint32_t vecCastRepeatNum = (blockCnt * paddingBlockNum * BYTENUM_BLOCK - 1) / BYTENUM_REPEAT + 1;

    Cast(dst, src, RoundMode::CAST_NONE, vecCastRepeatNum * ELENUM_REPEAT_FP16);
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::SeparateComplexMat()
{
    LocalTensor<float> matInner = this->matInLocal;
    LocalTensor<float> realInner = this->matRealLocal;
    LocalTensor<float> imagInner = this->matImagLocal;
    uint32_t matSize = this->n;
    uint32_t batchSize = this->batchNumPerRepeat;

    uint32_t alignedFpMatSize = (matSize * COMPLEX_ELENUM + ELENUM_BLOCK_FP32 - 1)
        / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;
    uint32_t dstGap = 0;
    uint32_t paddingBlockNum = (matSize * COMPLEX_ELENUM + ELENUM_BLOCK_FP32 - 1)
        / ELENUM_BLOCK_FP32 * ELENUM_BLOCK_FP32;
    if ((paddingBlockNum / ELENUM_BLOCK_FP32) % COMPLEX_ELENUM != 0) {
        dstGap = 1;
    }

    uint32_t matAlignedSize = alignedFpMatSize + ELENUM_BLOCK_FP32 * dstGap;
    uint16_t repeatTime = (matSize * batchSize * matAlignedSize +
        ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    uint64_t rsvdCnt = 0;
    GatherMask(realInner, matInner, 1, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
    GatherMask(imagInner, matInner, 2, false, 0, {1, repeatTime, 8, 8}, rsvdCnt);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::CollectScalarSingle(LocalTensor<float> collectDst,
    uint32_t sOffset)
{
    /*
        collect scalar from each matrix
    */
    LocalTensor<float> matInner = this->matSepLocal;
    LocalTensor<uint32_t> offsetInner = this->collectScalarOffsetLocal;
    uint32_t batchSize = this->batchNumPerRepeat;

    uint8_t repeatTime = (batchSize + ELENUM_REPEAT_FP32 - 1) /
        ELENUM_REPEAT_FP32 * COMPLEX_ELENUM;
    int32_t calCount = repeatTime * ELENUM_REPEAT_FP32;

    Gather(collectDst, matInner, offsetInner, sOffset * sizeof(float), calCount);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexScalarDiv()
{
    /*
        given (a + bi) / (c + di)
        result real part: (ac + bd) / (c^2 + d^2)
        result imag part: (bc - ad) / (c^2 + d^2)
    */
    uint32_t len = this->batchNumPerRepeat;
    uint32_t repeatTime = (len + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    // compute c^2 + d^2
    Mul(this->scalarTempLocal, this->preRealLocal, this->preRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    Mul(this->resRealLocal, this->preImagLocal, this->preImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    Add(this->scalarTempLocal, this->scalarTempLocal, this->resRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // c^2 + d^2
    PipeBarrier<PIPE_V>();

    // compute ac, ad, bd
    Mul(this->resRealLocal, this->afterRealLocal, this->preRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // ac
    PipeBarrier<PIPE_V>();
    Mul(this->afterRealLocal, this->afterRealLocal, this->preImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // ad
    PipeBarrier<PIPE_V>();
    Mul(this->resImagLocal, this->afterImagLocal, this->preImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // bd
    PipeBarrier<PIPE_V>();

    // compute ac + bd
    Add(this->resRealLocal, this->resRealLocal, this->resImagLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // compute (ac + bd) / (c^2 + d^2)
    Div(this->resRealLocal, this->resRealLocal, this->scalarTempLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // compute bc
    Mul(this->resImagLocal, this->afterImagLocal, this->preRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8}); // bc
    PipeBarrier<PIPE_V>();
    // compute bc - ad
    Sub(this->resImagLocal, this->resImagLocal, this->afterRealLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // compute (bc - ad) / (c^2 + d^2)
    Div(this->resImagLocal, this->resImagLocal, this->scalarTempLocal,
        ELENUM_REPEAT_FP32, repeatTime, {1, 1, 1, 8, 8, 8});
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::RealScalarVecBatchMul(LocalTensor<float> dst,
    LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat; // 64 aligned, 32
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Mul(dst[ELENUM_BLOCK_FP32 * matAlignedSize * j], src0[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src1[ELENUM_REPEAT_FP32 * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneRowBlockStride, oneMatBlockStride, 1, 1, 1, 0});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Mul(dst[ELENUM_BLOCK_FP32 * i], src0[ELENUM_BLOCK_FP32 * i],
                src1, ELENUM_REPEAT_FP32, (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneRowBlockStride, oneMatBlockStride, 1,
                static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneRowBlockStride),
                static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneMatBlockStride), 8}); // vec.R x scalar.R
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexScalarVecBatchMul(LocalTensor<float> matReal,
    LocalTensor<float> matImag, uint32_t oneMatOffset)
{
    LocalTensor<float> scalarRealInner = this->resRealLocal;
    LocalTensor<float> scalarImagInner = this->resImagLocal;
    LocalTensor<float> scalarRealBrcbInner = this->scalarBrcbRealLocal;
    LocalTensor<float> scalarImagBrcbInner = this->scalarBrcbImagLocal;
    LocalTensor<float> resRealInner = this->updateTempRealLocal;
    LocalTensor<float> resImagInner = this->updateTempImagLocal;
    LocalTensor<float> resTempRealInner = this->updateTempAugRealLocal;
    LocalTensor<float> resTempImagInner = this->updateTempAugImagLocal;

    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat; // 64 aligned, 32
    /*
        given complex scalar and vec, compute scalar x vec
    */
    // ub_scalar_brcb stores: | R1, R1, ..., R1 | R2, .., R2 | ... | I1, ..., I1 | ... In |
    // real part: 64 * 8 nums, imag part 64 * 8 nums

    uint8_t brcbRepeatTime = (batchSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32; // 4
    Brcb(scalarRealBrcbInner, scalarRealInner, brcbRepeatTime, {1, 8});
    Brcb(scalarImagBrcbInner, scalarImagInner, brcbRepeatTime, {1, 8});
    PipeBarrier<PIPE_V>();

    /*
        need level-0 api to compute
            real part: vec.R x scalar.R - vec.I x scalar.I
            imag part: vec.I x scalar.R + vec.R x scalar.I
    */
    uint32_t resComputeRepeat = (matAlignedSize * batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32; // 2

    // vec.R x scalar.R
    RealScalarVecBatchMul(resRealInner, matReal, scalarRealBrcbInner, oneMatOffset);
    // vec.I x scalar.I
    RealScalarVecBatchMul(resTempRealInner, matImag, scalarImagBrcbInner, oneMatOffset);

    // vec.R x scalar.R - vec.I x scalar.I
    Sub(resRealInner, resRealInner, resTempRealInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();

    // vec.I x scalar.R
    RealScalarVecBatchMul(resImagInner, matImag, scalarRealBrcbInner, oneMatOffset);
    // vec.I x scalar.R
    RealScalarVecBatchMul(resTempRealInner, matReal, scalarImagBrcbInner, oneMatOffset);

    // vec.I x scalar.R + vec.R x scalar.I
    Add(resImagInner, resImagInner, resTempRealInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
}


// dst = dst - src
template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::RealScalarVecBatchSub(LocalTensor<float> dst,
    LocalTensor<float> src, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32; // 8
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32; // 64 / 8 = 8, 256 / 8 = 64
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Sub(dst[ELENUM_BLOCK_FP32 * oneMatOffset * j], dst[ELENUM_BLOCK_FP32 * oneMatOffset * j],
                src[ELENUM_BLOCK_FP32 * matAlignedSize * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneMatBlockStride, oneMatBlockStride, oneRowBlockStride, 1, 1, 1});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Sub(dst[ELENUM_BLOCK_FP32 * i], dst[ELENUM_BLOCK_FP32 * i],
                src[ELENUM_BLOCK_FP32 * i], ELENUM_REPEAT_FP32, (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneMatBlockStride, oneMatBlockStride, oneRowBlockStride,
                static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                static_cast<uint8_t>(oneMatBlockStride * ELENUM_BLOCK_FP32),
                static_cast<uint8_t>(oneRowBlockStride * ELENUM_BLOCK_FP32)});
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComputeAndUpdateAj(LocalTensor<float> AjReal,
    LocalTensor<float> AjImag, uint32_t oneMatOffset)
{
    /*
        compute and update A[j,:] = A[j,:] - scalar * A[i,:]
    */
    LocalTensor<float> tempRealInner = this->updateTempRealLocal; // size: matAlignedSize * batchSize
    LocalTensor<float> tempImagInner = this->updateTempImagLocal; // size: matAlignedSize * batchSize

    // compute and update A[j,:] real part
    RealScalarVecBatchSub(AjReal, tempRealInner, oneMatOffset);

    // compute and update A[j,:] imag part
    RealScalarVecBatchSub(AjImag, tempImagInner, oneMatOffset);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::RealVecScalarBatchDiv(LocalTensor<float> dst,
    LocalTensor<float> src0, LocalTensor<float> src1, uint32_t oneMatOffset)
{
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat; // 64 aligned, 32
    uint8_t oneRowBlockStride = matAlignedSize / ELENUM_BLOCK_FP32;
    uint8_t oneMatBlockStride = oneMatOffset / ELENUM_BLOCK_FP32;
    uintptr_t oneMatBlockStrideFlag = oneMatOffset / ELENUM_BLOCK_FP32;
    uint32_t singleInstrRepeatTime = (matAlignedSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    if (oneMatBlockStrideFlag * ELENUM_BLOCK_FP32 > MAX_STRIDE_NUM) {
        for (uint32_t j = 0; j < batchSize / ELENUM_BLOCK_FP32; j++) {
            Div(dst[ELENUM_BLOCK_FP32 * oneMatOffset * j], src0[ELENUM_BLOCK_FP32 * matAlignedSize * j],
                src1[ELENUM_REPEAT_FP32 * j], ELENUM_REPEAT_FP32, (uint8_t)(singleInstrRepeatTime),
                {oneMatBlockStride, oneRowBlockStride, 1, 1, 1, 0});
        }
    } else {
        for (uint32_t i = 0; i < singleInstrRepeatTime; i++) {
            Div(dst[ELENUM_BLOCK_FP32 * i], src0[ELENUM_BLOCK_FP32 * i],
                src1, ELENUM_REPEAT_FP32, (uint8_t)(batchSize / ELENUM_BLOCK_FP32),
                {oneMatBlockStride, oneRowBlockStride, 1, static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneMatBlockStride),
                static_cast<uint8_t>(ELENUM_BLOCK_FP32 * oneRowBlockStride), 8}); // vec.R / scalar.R
        }
    }
    PipeBarrier<PIPE_V>();
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexVecScalarBatchDiv(LocalTensor<float> vecReal,
    LocalTensor<float> vecImag, uint32_t oneMatOffset)
{
    /*
        update A[i,:] = A[i,:] / aii
        given (vec.R + vec.I) / (scalar.R + scalar.I)
        result real part: (vec.R * scalar.R + vec.I * scalar.I) / (scalar.R^2 + scalar.I^2)
        result imag part: (vec.I * scalar.R - vec.R * scalar.I) / (scalar.R^2 + scalar.I^2)
    */
    LocalTensor<float> scalarRealInner = this->preRealLocal; // should be ub_scalar + 0, read only
    LocalTensor<float> scalarImagInner = this->preImagLocal; // should be ub_scalar + 64, read only
    LocalTensor<float> scalarTempRealInner = this->afterRealLocal;
    LocalTensor<float> scalarTempImagInner = this->afterImagLocal;
    LocalTensor<float> scalarRealBrcbInner = this->scalarBrcbRealLocal;
    LocalTensor<float> scalarImagBrcbInner = this->scalarBrcbImagLocal;
    LocalTensor<float> scalarTempBrcbInner = this->scalarBrcbAugLocal;
    LocalTensor<float> tempRealInner = this->updateTempRealLocal; // size: matAlignedSize * batchSize
    LocalTensor<float> tempImagInner = this->updateTempImagLocal; // size: matAlignedSize * batchSize
    uint32_t matAlignedSize = this->alignedComplexMatSize;
    uint32_t batchSize = this->batchNumPerRepeat;
    uint32_t squareSumRepeat = (batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;
    uint32_t resComputeRepeat = (matAlignedSize * batchSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;
    uint8_t brcbRepeatTime = (batchSize + ELENUM_BLOCK_FP32 - 1) / ELENUM_BLOCK_FP32;

    // compute scalar.R^2 + scalar.I^2
    // scalar.R^2
    Mul(scalarTempRealInner, scalarRealInner, scalarRealInner,
        ELENUM_REPEAT_FP32, squareSumRepeat, {1, 1, 1, 8, 8, 8});
    // scalar.I^2
    Mul(scalarTempImagInner, scalarImagInner, scalarImagInner,
        ELENUM_REPEAT_FP32, squareSumRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // scalar.R^2 + scalar.I^2
    Add(scalarTempRealInner, scalarTempRealInner, scalarTempImagInner,
        ELENUM_REPEAT_FP32, squareSumRepeat, {1, 1, 1, 8, 8, 8});
    // broadcast scalar.R
    Brcb(scalarRealBrcbInner, scalarRealInner, brcbRepeatTime, {1, 8});
    // broadcast scalar.I
    Brcb(scalarImagBrcbInner, scalarImagInner, brcbRepeatTime, {1, 8});
    PipeBarrier<PIPE_V>();
    // broadcast scalar.R^2 + scalar.I^2
    Brcb(scalarTempBrcbInner, scalarTempRealInner, brcbRepeatTime, {1, 8});

    // compute vec.R * scalar.R
    RealScalarVecBatchMul(tempRealInner, vecReal, scalarRealBrcbInner, oneMatOffset);
    // compute vec.I * scalar.I
    RealScalarVecBatchMul(tempImagInner, vecImag, scalarImagBrcbInner, oneMatOffset);
    // compute vec.R * scalar.R + vec.I * scalar.I
    Add(tempRealInner, tempRealInner, tempImagInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // compute vec.R * scalar.I
    RealScalarVecBatchMul(tempImagInner, vecReal, scalarImagBrcbInner, oneMatOffset);
    // (vec.R * scalar.R + vec.I * scalar.I) / (scalar.R^2 + scalar.I^2)
    RealVecScalarBatchDiv(vecReal, tempRealInner, scalarTempBrcbInner, oneMatOffset);
    // vec.I * scalar.R
    RealScalarVecBatchMul(tempRealInner, vecImag, scalarRealBrcbInner, oneMatOffset);
    // vec.I * scalar.R - vec.R * scalar.I
    Sub(tempRealInner, tempRealInner, tempImagInner, ELENUM_REPEAT_FP32, resComputeRepeat, {1, 1, 1, 8, 8, 8});
    PipeBarrier<PIPE_V>();
    // (vec.I * scalar.R - vec.R * scalar.I) / (scalar.R^2 + scalar.I^2)
    RealVecScalarBatchDiv(vecImag, tempRealInner, scalarTempBrcbInner, oneMatOffset);
}


template <typename T>
__aicore__ inline void CmatinvBatchedAIV<T>::ComplexCombinationAndCopyout(uint32_t outOffset, uint32_t batchSize)
{
    GlobalTensor<uint32_t> gmOffsetInner = this->offsetGM;
    GlobalTensor<T> gmMatOutInner = this->outGM[outOffset];
    LocalTensor<float> matInInner = this->invSepLocal;
    LocalTensor<uint32_t> offsetInner = this->complexCombineOffsetLocal;
    LocalTensor<float> matOutInner = this->matInLocal;

    uint32_t matRowSize = this->n; // complex
    uint32_t matColSize = this->n; // complex
    uint32_t matColFpAlignedSize = this->alignedComplexMatSize; // fp32, single mat, 8
    auto eventId = EVENT_ID0;

    uint32_t repeatTime = (matRowSize * matColFpAlignedSize * batchSize * COMPLEX_ELENUM +
        ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32;

    PipeBarrier<PIPE_ALL>();

    // copy offset in
    uint32_t copyinSize = matRowSize * matColFpAlignedSize * batchSize * COMPLEX_ELENUM;
    copyinSize = (copyinSize + ELENUM_REPEAT_FP32 - 1) / ELENUM_REPEAT_FP32 * ELENUM_REPEAT_FP32;
    CopyVecGmToUbUint32(offsetInner, gmOffsetInner, copyinSize);

    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);

    uint32_t cmdRepeats = repeatTime / MAX_REPEAT_NUM; // 255 repeats per time
    uint32_t remainRepeats = repeatTime % MAX_REPEAT_NUM;
    uint32_t currOffset = 0;
    for (uint32_t i = 0; i < cmdRepeats; i++) {
        Gather(matOutInner[currOffset], matInInner, offsetInner[currOffset], 0, MAX_REPEAT_NUM * ELENUM_REPEAT_FP32);
        currOffset += MAX_REPEAT_NUM * ELENUM_REPEAT_FP32;
        PipeBarrier<PIPE_ALL>();
    }
    if (remainRepeats > 0) {
        Gather(matOutInner[currOffset], matInInner, offsetInner[currOffset], 0, remainRepeats * ELENUM_REPEAT_FP32);
        PipeBarrier<PIPE_ALL>(); // necessary
    }

    if constexpr (IS_FLOAT) {
        LocalTensor<T> outLocal = matOutInner.template ReinterpretCast<T>();
        CopyVecUbToGm(gmMatOutInner, outLocal, matRowSize * matColSize * COMPLEX_ELENUM * batchSize);
    } else {
        LocalTensor<half> castDstLocal = matInInner.template ReinterpretCast<half>();
        uint32_t castRepeatNum = (matRowSize * matColSize * COMPLEX_ELENUM * batchSize - 1) / ELENUM_REPEAT_FP32 + 1;

        Cast(castDstLocal, matOutInner, RoundMode::CAST_NONE, castRepeatNum * ELENUM_REPEAT_FP32);

        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);

        LocalTensor<T> outLocal = matInInner.template ReinterpretCast<T>();
        CopyVecUbToGm(gmMatOutInner, outLocal, matRowSize * matColSize * COMPLEX_ELENUM * batchSize);
    }
}
}  // namespace CmatinvBatched
```

## 算子测试用例补充

算子实现后，在`test`目录下新增目录`cmatinv_batched`，补充算子测试用例，测试算子的正确性。

### gen_data.py
文件路径：`test/cmatinv_batched/data/gen_data.py`
主要功能：生成测试用例的输入数据和期望输出数据。

```python
import sys, os
import numpy as np

np.random.seed(42)

# 从命令行读取矩阵维度 n 和 batchSize
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
batchSize = int(sys.argv[2]) if len(sys.argv) > 2 else 2

# 生成批量随机复数矩阵 (complex64)
a_real = np.random.randn(batchSize, n, n).astype(np.float32)
a_imag = np.random.randn(batchSize, n, n).astype(np.float32)
a = a_real + 1j * a_imag

# 确保每个矩阵对角占优（复数情况）
for b in range(batchSize):
    for i in range(n):
        # 计算非对角线元素的绝对值之和（复数取模）
        row_sum = np.sum(np.abs(a[b, i, :])) - np.abs(a[b, i, i])
        # 调整对角线元素，使其模大于行其他元素模之和
        a[b, i, i] = (row_sum + np.random.rand() * 10) * (1 + 1j)

# 计算每个batch的逆矩阵
inv_a = np.zeros((batchSize, n, n), dtype=np.complex64)
for b in range(batchSize):
    inv_a[b] = np.linalg.inv(a[b]).astype(np.complex64)

os.makedirs(os.path.dirname("./test/cmatinv_batched/data/input/A_gm.bin"), exist_ok=True)
os.makedirs(os.path.dirname("./test/cmatinv_batched/data/golden/Ainv_gm.bin"), exist_ok=True)

# 写入输入数据 (原始矩阵)
with open("./test/cmatinv_batched/data/input/A_gm.bin", "wb") as f:
    f.write(a.tobytes())

# 写入输出数据 (逆矩阵)
with open("./test/cmatinv_batched/data/golden/Ainv_gm.bin", "wb") as f:
    f.write(inv_a.tobytes())

print(f"Generated batch {batchSize} of {n}x{n} complex matrices and their inverses")


# # 可选：打印矩阵形状和部分数据用于验证
# print("Input matrix batch 0 (first 3x3):\n", a[0, :3, :3])
# print("Inverse matrix batch 0 (first 3x3):\n", inv_a[0, :3, :3])

# # 验证第一个batch
# product = np.matmul(a[0], inv_a[0])
# identity = np.eye(n, dtype=np.complex64)

# # 计算误差
# abs_error = np.abs(product - identity)
# max_error = np.max(abs_error)
# mean_error = np.mean(abs_error)
# print("Verification results for batch 0:")
# print(f"Max absolute error: {max_error:.3e}")
# print(f"Mean absolute error: {mean_error:.3e}")
# print("Product of A and inv(A) (first 3x3):\n", product[:3, :3])
```

### verify_result.py
文件路径：`test/cmatinv_batched/data/verify_result.py`
主要功能：验证算子输出结果是否正确。

```python
import sys
import numpy as np

# for float32
relative_tol = 5e-3
absolute_tol = 5e-3
error_tol = 1e-4


def verify_result(A, invA, batchSize, N):
    A = np.fromfile(A, dtype=np.complex64).reshape(batchSize, N, N)
    invA = np.fromfile(invA, dtype=np.complex64).reshape(batchSize, N, N)

    # 计算每个batch的误差
    total_elements = 0
    total_different = 0

    for b in range(batchSize):
        output = np.dot(A[b], invA[b]).reshape(-1)
        golden = np.eye(N).astype(np.complex64).reshape(-1)
        different_element_results = np.isclose(
            output, golden, rtol=relative_tol, atol=absolute_tol, equal_nan=True
        )
        different_element_indexes = np.where(different_element_results == False)[0]

        # 打印前32个不匹配的元素
        for index in range(len(different_element_indexes)):
            if total_different >= 32:
                break
            real_index = different_element_indexes[index]
            golden_data = golden[real_index]
            output_data = output[real_index]
            print(
                "batch: %d, data index: %08d, expected: %.9f%+.9fj, actual: %.9f%+.9fj, rdiff: %.6f"
                % (
                    b,
                    real_index,
                    golden_data.real,
                    golden_data.imag,
                    output_data.real,
                    output_data.imag,
                    abs(output_data - golden_data) / abs(golden_data),
                )
            )

        total_elements += golden.size
        total_different += different_element_indexes.size

    error_ratio = float(total_different) / total_elements
    print("error ratio: %.4f, tolerance: %.4f" % (error_ratio, error_tol))
    return error_ratio <= error_tol


if __name__ == "__main__":
    try:
        batchSize = int(sys.argv[1]) if len(sys.argv) > 1 else 2
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 3

        res = verify_result(
            "./test/cmatinv_batched/data/input/A_gm.bin",
            "./test/cmatinv_batched/data/output/Ainv_gm.bin",
            batchSize,
            n,
        )

        if not res:
            print("[Failed] Case accuracy verification failed.")
            sys.exit(1)
        else:
            print("[Success] Case accuracy verification passed.")
            sys.exit(0)
    except Exception as e:
        print(e)
        sys.exit(1)
```

### cmatinv_batched_test.cpp
文件路径：`test/cmatinv_batched/cmatinv_batched_test.cpp`
主要功能：运行算子测试用例。

```c++
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <vector>

#include "../utils/test_utils.h"
#include "acl/acl.h"
#include "cann_ops_solver.h"

int32_t main(int32_t argc, char* argv[])
{
    int deviceId, batchSize, n;
    deviceId = (argc > 1) ? std::atoi(argv[1]) : 0;
    batchSize = (argc > 2) ? std::atoi(argv[2]) : 2;
    n = (argc > 3) ? std::atoi(argv[3]) : 3;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    aclsolverHandle_t handle = nullptr;
    CHECK_ACL(aclsolverCreate(&handle));
    CHECK_ACL(aclsolverSetStream(handle, stream));

    size_t aMatrixFileSize = batchSize * n * n * sizeof(std::complex<float>);
    std::complex<float>* A = nullptr;
    auto cleanup = [&]() -> aclError
    {
        if (A != nullptr)
        {
            CHECK_ACL(aclrtFreeHost(A));
        }
        CHECK_ACL(aclrtDestroyStream(stream));
        if (handle != nullptr)
        {
            CHECK_ACL(aclsolverDestroy(handle));
        }
        CHECK_ACL(aclrtResetDevice(deviceId));
        CHECK_ACL(aclFinalize());
        return ACL_SUCCESS;
    };

    CHECK_ACL(aclrtMallocHost((void**)(&A), aMatrixFileSize));
    CHECK_RET(ReadFile("./test/cmatinv_batched/data/input/A_gm.bin", aMatrixFileSize, A, aMatrixFileSize),
              LOG_PRINT("ReadFile A_gm.bin failed.\n");
              cleanup(); return ACL_ERROR_INVALID_PARAM);
    std::vector<std::complex<float>> Ainv(batchSize * n * n, {-1.0f, -1.0f});
    std::vector<int32_t> info(batchSize, 0);

    std::cout << "[Input] A:" << std::endl;
    printTensor(A, batchSize, n, n);
    std::cout << "[Input] Ainv:" << std::endl;
    printTensor(Ainv.data(), batchSize, n, n);

    auto ret = aclsolverCmatinvBatched(handle, n, A, n, Ainv.data(), n, info.data(), batchSize);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclsolverCmatinvBatched failed. ERROR: %d\n", ret); cleanup(); return ret);

    std::cout << "[Output] Ainv:" << std::endl;
    printTensor(Ainv.data(), batchSize, n, n);

    WriteFile("./test/cmatinv_batched/data/output/Ainv_gm.bin", Ainv.data(), aMatrixFileSize);

    auto cleanupRet = cleanup();
    CHECK_RET(cleanupRet == ACL_SUCCESS, return cleanupRet);

    return 0;
}
```

### CMakeLists.txt
文件路径：`test/cmatinv_batched/CMakeLists.txt`
主要功能：配置算子测试用例的编译规则。

```cmake
add_executable(cmatinv_batched_test
    cmatinv_batched_test.cpp
)

target_include_directories(cmatinv_batched_test PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    $ENV{LINUX_INCLUDE_PATH}
)

target_link_libraries(cmatinv_batched_test PRIVATE
    ${OPS_SOLVER}
    $ENV{EAGER_LIBRARY_PATH}/libascendcl.so
)
```

### 运行结果示例

```cpp
========== Running cmatinv_batched_test ==========
step 1: generate test data...
Generated batch 2 of 3x3 complex matrices and their inverses

step 2: run cmatinv_batched_test...
[Input] A:
(6.13855,6.13855) (-0.138264,-1.4123) (0.647689,1.46565)
(1.52303,-0.225776) (8.18421,8.18421) (-0.234137,-1.42475)
(1.57921,-0.544383) (0.767435,0.110923) (7.91292,7.91292)

(3.15671,3.15671) (-0.463418,-0.600639) (-0.46573,-0.291694)
(0.241962,-0.601707) (12.0694,12.0694) (-1.72492,-0.0134972)
(-0.562288,-1.05771) (-1.01283,0.822545) (10.254,10.254)

[Input] Ainv:
(-1,-1) (-1,-1) (-1,-1)
(-1,-1) (-1,-1) (-1,-1)
(-1,-1) (-1,-1) (-1,-1)

(-1,-1) (-1,-1) (-1,-1)
(-1,-1) (-1,-1) (-1,-1)
(-1,-1) (-1,-1) (-1,-1)

[Output] Ainv:
(0.0800811,-0.0820767) (0.0143785,-0.00248317) (-0.0132419,0.00764272)
(0.00181733,0.0169676) (0.0595009,-0.0586753) (0.0112855,-0.00403378)
(0.00497235,0.0152462) (-0.00147531,0.00781663) (0.0625691,-0.0647724)

(0.161557,-0.15957) (0.00742409,-0.00653919) (0.00469627,-0.0084383)
(0.00852605,0.00139513) (0.0412678,-0.0414004) (0.000369989,-0.00697411)
(0.0168208,-0.00945229) (-0.00257783,-0.00441464) (0.0487157,-0.0494125)


step 3: verify result...
error ratio: 0.0000, tolerance: 0.0001
[Success] Case accuracy verification passed.
[PASS] cmatinv_batched_test
Test for cmatinv_batched completed.

```

## 编译说明

CmatinvBatched算子的编译步骤如下：

1. 确保已安装CANN开发环境和必要的依赖
2. 在项目根目录下执行编译命令：

```bash
    bash build.sh --ops=cmatinv_batched
```

3. 编译成功后，会在输出目录生成相应的库文件和可执行文件

**注意事项：**
- 确保CANN环境变量已正确设置
- 确保编译器版本与CANN版本兼容
- 如遇到编译错误，请检查头文件路径和库文件路径是否正确
