## SSgels算子实现

## 概述

Solver SSgels 算子实现（`aclsolverSSgels / aclsolverSSgels_bufferSize`）。解超定/方阵最小二乘问题（QR 路线）。

## 支持的产品

- Ascend 950PR / Ascend 950DT

## 目录结构介绍

```
├── sgels
│   ├── CMakeLists.txt       // 编译工程文件
│   ├── README.md            // 说明文档（本文件）
│   ├── sgels_test.cpp    // 算子调用样例与自测
│   └── data                 // 测试数据目录
│       ├── sgels_cases.txt      // 精度/性能用例 166 条（其中 108 条 cuSolver 自己失败、无 GPU 基线）
│       └── sgels_cases_pow2.txt // 2^k 与 2^k-1 形状 46 条，nrhs 轮转 1/8/32/128
```

## 算子描述

- 算子功能：
    $$\min_X \lVert A X - B \rVert, \qquad A \in \mathbb{R}^{m \times n},\; m \ge n$$
    实现为三段：`Sgeqrf(A)` → `Sormqr(LEFT, OP_T)` 把 $Q^{T}$ 作用到 B 的副本 → 上三角回代 $R X = (Q^{T}B)(1{:}n,:)$。

- 算子规格：

    | 输入/输出 | name | shape | data type | 说明 |
    |---|---|---|---|---|
    | 输入 | m, n, nrhs | [1] | int | m >= n >= 0，nrhs >= 0 |
    | 输入/输出 | dA | [ldda, n] | FLOAT32 | 列主序 ND；**求解后内容被破坏**（任务书允许）；ldda >= max(1, m) |
    | 输入 | dB | [lddb, nrhs] | FLOAT32 | 列主序 ND，**只读、不被修改**；lddb >= max(1, m) |
    | 输出 | dX | [lddx, nrhs] | FLOAT32 | 列主序 ND，n x nrhs 的解；**不得与 dB 同址**；lddx >= max(1, n) |
    | 输入 | dWorkspace / lwork_bytes | - | void*/size_t | device workspace，以**字节**计 |
    | 输出 | niter | [1] | int* | **Host 标量**。本实现是直接 QR 解、不做迭代精化，恒为 0 |
    | 输出 | dinfo | [1] | INT32 | 0 成功；-i 第 i 个参数非法 |

    核函数名：`strsm_mix_kernel（前两段复用 sgeqrf_mix_kernel / sormqr_mix_kernel）`

- 算子约束：m / n 上限 4096；要求 m >= n（m < n 返回 info = -1）；`dX` 与 `dB` 同址返回 info = -8。

- 调用实现：Host 侧做参数校验与 workspace 布局，使用内核调用符 `<<<>>>` 下发核函数。

## 精度判据

精度判据取任务书 §3.2.1-4，按形状分流：
    - 方阵 m = n：LAPACK **SQRT16**，`resid_j = ‖B_j − A·X_j‖₁ / (max(m,n)·‖A‖₁·‖X_j‖₁·ε)` 逐列取最大，阈值 `max(5·ratio_cpu, 0.015)`；
    - 超定 m > n：LAPACK **SQRT17**，`ratio_opt = ‖RᵀA‖₁ / (‖A‖₁·‖B‖₁·max(m,n,nrhs)·ε)`（R = B − A·X），阈值 `10·ratio_opt_cpu`。
      ⚠ 纯相对判据在 `ratio_opt_cpu → 0` 时失去判别力（最小二乘解让正规方程残差本来就接近 0，实测 CPU 侧低到 1e-6~1e-8），
      所以阈值补了 **1.0 的绝对地板**（量纲已被 ε 归一化，1.0 即"1 个 ulp 的量级"）。这是补充而非放宽。

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
CATLASS_HOME=$CATLASS_HOME bash build.sh --ops=sgels --soc=Ascend950PR_9579 --run

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
cd build/test/sgels
./sgels_test                                     # 功能 + 16 条参数校验 + 3 空问题 + 14 形状（nrhs 覆盖 1/8/32/128）
./sgels_test scenario                            # 任务书 §3.5 的跨接口必测场景 15 条（见下）
./sgels_test sweep     data/sgels_cases.txt 0    # 全量精度 166 条（第 3 个参数是额外 padding）
./sgels_test perfsweep data/sgels_cases.txt      # 全量性能（有 GPU 基线的 58 条逐条判定，其余 108 条只记录）
./sgels_test sweep     data/sgels_cases_pow2.txt 0    # 2^k 与 2^k-1 形状 46 条
./sgels_test sweep     data/sgels_cases_pow2.txt 8    # 同上，padding +8
./sgels_test sweep     data/sgels_cases_pow2.txt 32   # 同上，padding +32
```

### 跨接口必测场景（`./sgels_test scenario`）

任务书 §3.5 要求的跨接口场景放在本目录的 `scenario.inc` 里（SSgels 本身就是
"geqrf + ormqr + 回代"的组合算子，它的测试已经链接了全部五个接口；而仓里 `test/<op>`
与 `src/<op>` 一一对应，多一个无主目录会破坏这个约定）：

| 场景 | 内容 |
|---|---|
| 1 | `geqrf + orgqr` 还原 Q：验 `‖I − QᵀQ‖` 与 `‖Q·R − A‖` |
| 2 | `geqrf + ormqr` 往返：Qᵀ 再 Q，回到原矩阵 |
| 3 | Xlarft 的 T 语义：块公式 `I − VTVᵀ` vs 逐个施加反射子（double 下比） |
| 4 | SSgels 端到端（两类判据） |
| 5 | **五个接口连续下发、中间完全不同步**，3 形状 × 3 趟，逐位一致（早期怀疑会挂死，实测不复现，保留为回归守卫）|
| 6 | 非默认 stream，且中途换一条流 |
| 7 | 确定性：A / tau / info **全部**逐位比对，×3 趟 |

