## Xlarft算子实现

## 概述

Solver Xlarft 算子实现（`aclsolverXlarft / aclsolverXlarft_bufferSize`）。由 k 个 Householder 反射子构造块反射子的三角因子 T。

## 支持的产品

- Ascend 950PR / Ascend 950DT

## 目录结构介绍

```
├── xlarft
│   ├── CMakeLists.txt       // 编译工程文件
│   ├── README.md            // 说明文档（本文件）
│   ├── xlarft_test.cpp   // 算子调用样例与自测
│   └── data                 // 测试数据目录
│       └── data/README.md     // 说明：任务包里没有 xlarft 目录（无 cases.json / bench_result.json），
│                             //       形状由测试自己列，性能只记录绝对耗时
```

## 算子描述

- 算子功能：
    $$H_1 H_2 \cdots H_k = I - V T V^{T} \;(\text{direct} = \mathrm{FORWARD}),\qquad H_k \cdots H_1 = I - V T V^{T} \;(\text{BACKWARD})$$
    FORWARD 时 $T$ 为上三角，BACKWARD 时为下三角；**只写被引用的那半个三角**，另一半原样不动（对齐 LAPACK slarft）。

- 算子规格：

    | 输入/输出 | name | shape | data type | 说明 |
    |---|---|---|---|---|
    | 输入 | params | [1] | aclsolverParams_t | 必须为 nullptr（本实现不提供 params 对象） |
    | 输入 | direct | [1] | aclsolverDirectMode_t | FORWARD / BACKWARD |
    | 输入 | storev | [1] | aclsolverStorevMode_t | 仅 COLUMNWISE，其它值报错 |
    | 输入 | n, k | [1] | int64_t | 块反射子阶数与反射子个数，n >= k >= 0 |
    | 输入 | V | [ldv, k] | FLOAT32 | 列主序 ND，隐含对角 1；ldv >= max(1, n) |
    | 输入 | tau | [k] | FLOAT32 | 反射子系数 |
    | 输出 | T | [ldt, k] | FLOAT32 | 列主序 ND，k x k 三角因子；ldt >= max(1, k) |
    | 输入 | dataTypeV / Tau / T / computeType | [1] | aclsolverDataType_t | 四者必须全为 `ACLSOLVER_R_32F`，否则返回 `ACLSOLVER_STATUS_NOT_SUPPORTED` |
    | 输入 | bufferOnDevice / Host + 字节数 | - | void*/size_t | workspace 以**字节**计，device 与 host 两个查询值；host 侧恒为 0 |

    核函数名：`xlarft_mix_kernel`

- 算子约束：n 上限 4096，k 上限 256（任务书 §2.6 的 `k ∈ [1, min(n,256)]`）；**本接口没有 devInfo**，参数错误只通过返回码报告。

- 调用实现：Host 侧做参数校验与 workspace 布局，使用内核调用符 `<<<>>>` 下发核函数。

## 精度判据

⚠ 任务书 §3.2.1 **没有给 Xlarft 的精度判据**，本测试的判据是自定的，口径写在 `xlarft_test.cpp` 头部：对 float64 的 LAPACK slarft 等价实现逐元素比，`ratio = err / (k·max(1, max|T_ref|)·ε)`，阈值 `max(5·ratio_cpu, 1.0)`；另外两条结构性要求也进判据 —— 被引用三角必须全部写到、另一半必须原样不动（T 事先投毒 −7 检查）。

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
CATLASS_HOME=$CATLASS_HOME bash build.sh --ops=xlarft --soc=Ascend950PR_9579 --run

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
cd build/test/xlarft
./xlarft_test        # 功能 + 23 条参数校验 + 3 空问题 + 33 形状 x FORWARD/BACKWARD（共 70 条）
./xlarft_test perf   # 绝对耗时（任务书未提供 GPU 基线，**不做达标判定**），含 P-08 / P-09 两条
```


