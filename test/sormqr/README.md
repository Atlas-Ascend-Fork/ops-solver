## Sormqr算子实现

## 概述

Solver Sormqr 算子实现（`aclsolverSormqr / aclsolverSormqr_bufferSize`）。把 Sgeqrf 得到的 op(Q) 左乘或右乘到矩阵 C 上（不显式构造 Q）。

## 支持的产品

- Ascend 950PR / Ascend 950DT

## 目录结构介绍

```
├── sormqr
│   ├── CMakeLists.txt       // 编译工程文件
│   ├── README.md            // 说明文档（本文件）
│   ├── sormqr_test.cpp   // 算子调用样例与自测
│   └── data                 // 测试数据目录
│       ├── sormqr_cases.txt      // 精度/性能用例 189 条（取自任务包）
│       └── sormqr_cases_pow2.txt // 2^k 与 2^k-1 形状 46 条，六种 (side,trans) 轮转
```

## 算子描述

- 算子功能：
    $$C \leftarrow \mathrm{op}(Q) \cdot C \;(\text{side} = \mathrm{LEFT}), \qquad C \leftarrow C \cdot \mathrm{op}(Q) \;(\text{side} = \mathrm{RIGHT})$$
    `trans = N` 时 $\mathrm{op}(Q) = Q$；`trans = T` 或 `C` 时 $\mathrm{op}(Q) = Q^{T}$（实数下两者等价）。

- 算子规格：

    | 输入/输出 | name | shape | data type | 说明 |
    |---|---|---|---|---|
    | 输入 | side | [1] | aclsolverSideMode_t | LEFT / RIGHT |
    | 输入 | trans | [1] | aclsolverOperation_t | N / T / C |
    | 输入 | m, n | [1] | int | C 的行数与列数，均 >= 0 |
    | 输入 | k | [1] | int | 反射子个数，0 <= k <= nq（nq = side==LEFT ? m : n） |
    | 输入 | A | [lda, k] | FLOAT32 | 列主序 ND，Sgeqrf 的输出；lda >= max(1, nq) |
    | 输入 | tau | [k] | FLOAT32 | 反射子系数 |
    | 输入/输出 | C | [ldc, n] | FLOAT32 | 列主序 ND，原地更新；ldc >= max(1, m) |
    | 输入 | work / lwork | [lwork] | FLOAT32 | device workspace，单位是元素个数 |
    | 输出 | devInfo | [1] | INT32 | 0 成功；-i 第 i 个参数非法 |

    核函数名：`sormqr_mix_kernel`

- 算子约束：nq 上限 4096；`k = 0` 时 op(Q) = I，C 原样返回。

- 调用实现：Host 侧做参数校验与 workspace 布局，使用内核调用符 `<<<>>>` 下发核函数。

## 精度判据

精度判据取任务书 §3.2.1-3（LAPACK **SQRT03**）：`ratio = ‖op2 − op1‖₁ / (nq·‖C‖₁·ε)`，阈值 `max(5·ratio_cpu, 0.1)`。其中 op1 是本接口的输出，op2 是**参考路径** —— 同一份 (v, tau) 先经 `aclsolverSorgqr` 生成显式 nq×nq 的 Q 再做矩阵乘（任务书也因此要求 Sorgqr 先过验收）。

## 编译运行

### 1. 前置依赖

| 依赖 | 说明 |
|---|---|
| 硬件 | Ascend 950PR（`npu-smi info` 的 Name 应为 `Ascend950PR`） |
| CANN | 9.0.0 及以上。本算子的自测在 **CANN 9.1.0**（`V100R001C11SPC001B243`）、驱动 **25.7.rc1.6** 上完成 |
| CATLASS | QR 系列的 GEMM 层基于 CATLASS（<CATLASS 仓库（cann/catlass）>），**不随 CANN 发布、也不在本仓内**，需自行 clone。任务书 §2.2 明文允许"基于 AscendC/CATLASS 实现" |

### 2. 配置环境变量

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export CATLASS_HOME=/path/to/catlass        # CATLASS 仓根（含 include/catlass）
```

### 3. 编译

两种方式都可以，推荐第一种（仓自带脚本）：

```bash
# 方式一：仓自带 build.sh
CATLASS_HOME=$CATLASS_HOME bash build.sh --ops=sormqr --soc=Ascend950PR_9579 --run

# 方式二：直接调 cmake（五个算子一起编）
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LINUX_INCLUDE_PATH=$ASCEND_HOME_PATH/$(uname -m)-linux/include
export EAGER_LIBRARY_PATH=$ASCEND_HOME_PATH/lib64
mkdir -p build && cd build
cmake .. -DSOC_VERSION=Ascend950PR_9579 -DCATLASS_HOME=$CATLASS_HOME \
         -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON \
         "-DTEST_NAMES=sgeqrf;sorgqr;sormqr;xlarft;sgels"
make -j16
```

> `CATLASS_HOME` 不传时整个 CATLASS 块会被跳过，本仓其它算子照常编译；但 QR 系列的 kernel 需要它。

### 4. 运行

```bash
export LD_LIBRARY_PATH=<build 目录>:$LD_LIBRARY_PATH
cd build/test/sormqr
./sormqr_test                                       # 功能 + 参数校验 + 12 形状 x 6 种 (side,trans)
./sormqr_test sweep     data/sormqr_cases.txt 32 0     # 全量精度 189 条（参数：抽样列数上限、额外 padding）
./sormqr_test perfsweep data/sormqr_cases.txt          # 全量性能 189 条
./sormqr_test sweep     data/sormqr_cases_pow2.txt 32 0    # 2^k 与 2^k-1 形状 46 条
./sormqr_test sweep     data/sormqr_cases_pow2.txt 32 8    # 同上，padding +8
./sormqr_test sweep     data/sormqr_cases_pow2.txt 32 32   # 同上，padding +32
./sormqr_test floor                                 # 启动地板分解（空同步 / memsetAsync / 单次调用）
./sormqr_test chain <m> <n>                         # 连续下发探测（中间不同步 vs 中间同步，逐位比对）
```


