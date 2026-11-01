## Sorgqr算子实现

## 概述

Solver Sorgqr 算子实现（`aclsolverSorgqr / aclsolverSorgqr_bufferSize`）。由 Sgeqrf 输出的 Householder 表示生成显式正交矩阵 Q 的前 n 列。

## 支持的产品

- Ascend 950PR / Ascend 950DT

## 目录结构介绍

```
├── sorgqr
│   ├── CMakeLists.txt       // 编译工程文件
│   ├── README.md            // 说明文档（本文件）
│   ├── sorgqr_test.cpp   // 算子调用样例与自测
│   └── data                 // 测试数据目录
│       ├── sorgqr_cases.txt      // 精度/性能用例 165 条（取自任务包）
│       └── sorgqr_cases_pow2.txt // 2^k 与 2^k-1 形状 44 条（无 GPU 基线，只做精度）
```

## 算子描述

- 算子功能：
    $$Q = H_1 H_2 \cdots H_k, \quad H_i = I - \tau_i v_i v_i^{T}$$
    输出 $Q$ 的前 $n$ 列，原地覆盖 $A$。

- 算子规格：

    | 输入/输出 | name | shape | data type | 说明 |
    |---|---|---|---|---|
    | 输入 | m | [1] | int | Q 的行数，m >= 0 |
    | 输入 | n | [1] | int | Q 的列数，0 <= n <= m |
    | 输入 | k | [1] | int | 反射子个数，0 <= k <= n |
    | 输入/输出 | A | [lda, n] | FLOAT32 | 列主序 ND；输入为 Sgeqrf 的输出，返回时被 Q 覆盖 |
    | 输入 | lda | [1] | int | lda >= max(1, m) |
    | 输入 | tau | [k] | FLOAT32 | Sgeqrf 输出的反射子系数 |
    | 输入 | work / lwork | [lwork] | FLOAT32 | device workspace，单位是元素个数 |
    | 输出 | devInfo | [1] | INT32 | 0 成功；-i 第 i 个参数非法 |

    核函数名：`sorgqr_mix_kernel`

- 算子约束：m 上限 4096；**k = 0 不是空问题** —— 此时 Q 为单位阵的前 n 列，仍须真实写出（对齐 LAPACK dorgqr）。

- 调用实现：Host 侧做参数校验与 workspace 布局，使用内核调用符 `<<<>>>` 下发核函数。

## 精度判据

精度判据取任务书 §3.2.1-2（LAPACK **SQRT02**）：`ratio = ‖I − QᵀQ‖₁ / (m·ε)`，阈值 `max(5·ratio_cpu, 1.5)`。分母固定取**行数 m**，不随 n 变化。

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
CATLASS_HOME=$CATLASS_HOME bash build.sh --ops=sorgqr --soc=Ascend950PR_9579 --run

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
cd build/test/sorgqr
./sorgqr_test                                      # 功能 + 参数校验 + 14 个形状
./sorgqr_test sweep     data/sorgqr_cases.txt 48 0    # 全量精度 165 条（参数：抽样列数上限、额外 padding）
./sorgqr_test perfsweep data/sorgqr_cases.txt 0       # 全量性能 165 条
./sorgqr_test sweep     data/sorgqr_cases_pow2.txt 48 0   # 2^k 与 2^k-1 形状 44 条
./sorgqr_test sweep     data/sorgqr_cases_pow2.txt 48 8   # 同上，padding +8
./sorgqr_test sweep     data/sorgqr_cases_pow2.txt 48 32  # 同上，padding +32
```


