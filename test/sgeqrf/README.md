## Sgeqrf算子实现

## 概述

Solver Sgeqrf 算子实现（`aclsolverSgeqrf / aclsolverSgeqrf_bufferSize`）。对单精度实数矩阵做 QR 分解（Householder 形式）。

## 支持的产品

- Ascend 950PR / Ascend 950DT

## 目录结构介绍

```
├── sgeqrf
│   ├── CMakeLists.txt       // 编译工程文件
│   ├── README.md            // 说明文档（本文件）
│   ├── sgeqrf_test.cpp   // 算子调用样例与自测
│   └── data                 // 测试数据目录
│       ├── sgeqrf_cases.txt        // 精度/性能用例 174 条（取自任务包 cases.json + bench_result.json）
│       ├── sgeqrf_cases_pow2.txt   // 2^k 与 2^k-1 形状 50 条（无 GPU 基线，只做精度）
│       └── sgeqrf_cases_subset.txt // padding 两档用的子集 30 条
```

## 算子描述

- 算子功能：
    $$A = Q \cdot R$$
    其中 $A$ 为 $m \times n$ 实数矩阵，$Q$ 为 $m \times m$ 正交矩阵，$R$ 为上三角矩阵。
    返回时 $A$ 的上三角存 $R$、对角以下存 Householder 向量，`TAU` 存各反射子的系数。

- 算子规格：

    | 输入/输出 | name | shape | data type | 说明 |
    |---|---|---|---|---|
    | 输入 | m | [1] | int | A 的行数，m >= 0 |
    | 输入 | n | [1] | int | A 的列数，n >= 0 |
    | 输入/输出 | A | [lda, n] | FLOAT32 | 列主序 ND；返回时上三角为 R、对角以下为 V |
    | 输入 | lda | [1] | int | A 的 leading dimension，lda >= max(1, m) |
    | 输出 | TAU | [min(m,n)] | FLOAT32 | 各 Householder 反射子的系数 |
    | 输入 | Workspace / Lwork | [Lwork] | FLOAT32 | device workspace，单位是**元素个数**，由 _bufferSize 查询 |
    | 输出 | devInfo | [1] | INT32 | 0 成功；-i 表示第 i 个参数非法（不计 handle） |

    核函数名：`sgeqrf_mix_kernel`

- 算子约束：m / n 上限 8192（kernel 的 UB 预算，超出返回 `ACLSOLVER_STATUS_NOT_SUPPORTED`）；`lda` 可大于 m（不要求紧凑排布）。

- 调用实现：Host 侧做参数校验与 workspace 布局，使用内核调用符 `<<<>>>` 下发核函数。

## 精度判据

精度判据取任务书 §3.2.1-1（LAPACK **SQRT01**）：`ratio = ‖A − Q·R‖₁ / (m·‖A‖₁·ε)`，阈值 `max(5·ratio_cpu, 0.1)`。`ratio_cpu` 为同 case 同式的 CPU fp32 参考链路实测值。

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
CATLASS_HOME=$CATLASS_HOME bash build.sh --ops=sgeqrf --soc=Ascend950PR_9579 --run

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
cd build/test/sgeqrf
./sgeqrf_test                                       # 功能 + 参数校验 + 28 个形状（含 m<n）
./sgeqrf_test sweep     ./data/sgeqrf_cases.txt 48 0   # 全量精度 174 条（第 3 个参数是抽样列数上限，第 4 个是额外 padding）
./sgeqrf_test perfsweep ./data/sgeqrf_cases.txt 0      # 全量性能 174 条（判据 >= 0.35x GPU）
./sgeqrf_test sweep     ./data/sgeqrf_cases_pow2.txt 48 0     # 2^k 与 2^k-1 形状 50 条
./sgeqrf_test sweep     ./data/sgeqrf_cases_subset.txt 48 8   # padding +8
./sgeqrf_test sweep     ./data/sgeqrf_cases_subset.txt 48 32  # padding +32
./sgeqrf_test perf <m> <n> <lda> <iters>            # 单形状计时
```


