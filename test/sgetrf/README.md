## Sgetrf算子实现

## 概述

Solver Sgetrf算子实现。

本目录包含公开Device接口的基础测试和官方固定v2协议adapter。基础测试在程序内准备小规模输入，并检查实际Device输出、INFO、padding输入、空问题、非法维数和三次串行调用的bit-wise一致性。

## 支持的产品

- Ascend 950PR

使用仓库README列出的配套CANN环境；其他产品的当前实数Device接口验证状态见接口文档。

## 目录结构介绍

```text
├── sgetrf
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── README.md                   // 说明文档
│   ├── sgetrf_test.cpp              // 基础功能与边界测试
│   ├── sgetrf_adapter.cpp           // 固定协议与公开Device接口之间的适配
│   ├── solver_adapter.h            // 官方固定v2协议定义
│   ├── sgetrf_subnormal_pivot_test.cpp // 次正规主元专项回归
│   └── data
│       ├── gen_data.py             // 官方样例输入生成
│       └── verify_result.py        // 官方LU重构与非有限值校验
```

构建产物为 `sgetrf_test`、`sgetrf_subnormal_pivot_test` 和 `libsgetrf_adapter.so`。

`sgetrf_test` 默认执行基础回归，`--files` 模式接入官方文件样例；次正规测试覆盖2×2次正规数、正常小量对照和513×1次正规长列，按输入自身尺度要求相对残差≤1e-5。

## 算子描述

- 算子功能：

    Sgetrf算子对实数矩阵进行LU分解，对应的数学表达式为：
    $$P \cdot A = L \cdot U$$
    P表示按返回主元顺序执行的行交换，L为单位下三角部分，U为上三角部分。

- 算子规格：

    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">Sgetrf</td>
    </tr>
    <tr>
        <td rowspan="6" align="center">算子输入</td>
        <td align="center">name</td>
        <td align="center">shape</td>
        <td align="center">data type</td>
        <td align="center">Description</td>
        <td align="center">format</td>
    </tr>
    <tr>
        <td align="center">m</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">矩阵A的行数，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">矩阵A的列数，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[m,n]</td>
        <td align="center">FLOAT32</td>
        <td align="left">Device列主序原始矩阵，分配至少lda*n个元素</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">相邻两列起始元素的步长，lda≥max(1,m)，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">Workspace</td>
        <td align="center">[Lwork]</td>
        <td align="center">FLOAT32</td>
        <td align="left">调用方分配的Device临时工作区，Lwork由查询接口返回</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td rowspan="4" align="center">算子输出</td>
        <td align="center">name</td>
        <td align="center">shape</td>
        <td align="center">data type</td>
        <td align="center">Description</td>
        <td align="center">format</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[m,n]</td>
        <td align="center">FLOAT32</td>
        <td align="left">Device packed LU，L的单位对角不存储</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">devIpiv</td>
        <td align="center">[min(m,n)]</td>
        <td align="center">INT32</td>
        <td align="left">Device 1-based主元，nullptr表示不选主元</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">devInfo</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">Device实际状态；0成功、正值奇异、负值参数错误</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">real_lu_factor_kernel、real_lu_thin_panel_kernel及分块分解核</td>
    </tr>
    </table>

- 算子约束：

  - m、n可为0；空问题成功返回并写devInfo=0，devInfo仍须为有效Device INT32。
  - devIpiv为nullptr时不执行主元交换；非空时返回min(m,n)个1-based行交换索引。
  - 先调用aclsolverSgetrf_bufferSize，按返回的FLOAT32元素数分配Device workspace；查询时A可为nullptr，Lwork须指向Host int。
  - m、n、lda和分配容量须满足上表约束；workspace查询拒绝超过int可表示范围的元素数。

- 调用实现

    adapter把紧凑Host行主序传输缓冲转换为Device列主序，通过 `aclsolverSgetrf` 调用实际公开接口；同步后读取实际输出和INFO。布局转换、数据搬运和输入恢复在目标性能采集区间外完成。

    完整函数原型、内存容量和返回码见[Sgetrf接口说明](../../docs/zh/sgetrf.md)。

## 编译运行

在仓库根目录下执行如下步骤，编译并执行算子。
- 配置环境变量
  请根据当前环境上CANN开发套件包的安装方式，选择对应配置环境变量的命令。
  - 默认路径，root用户安装CANN软件包
    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

  - 默认路径，非root用户安装CANN软件包
    ```bash
    source $HOME/Ascend/cann/set_env.sh
    ```

  - 指定路径install_path，安装CANN软件包
    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行

  ```bash
  bash build.sh --soc=ascend950 --ops=sgetrf --run # --ops=<算子名> --run可选参数，执行测试样例
  ```

  命令构建算子库、本接口adapter和基础测试，并运行 `sgetrf_test`。成功时程序打印包含
  `sgetrf basic tests PASS` 的摘要，构建脚本打印 `[PASS] sgetrf_test (basic functionality and boundaries)`。

- 已构建后直接运行，参数为设备号：

  ```bash
  LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH" ./build/test/sgetrf/sgetrf_test 0
  ```

- 独立构建本目录，链接已固定的算子库：

  ```bash
  cmake -S test/sgetrf -B build-sgetrf \
    -DOPS_SOLVER_LIBRARY="$PWD/build/libops_solver.so"
  cmake --build build-sgetrf -j8
  LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH" ./build-sgetrf/sgetrf_test 0
  ```

  如需链接其他位置的固定算子库，替换 `OPS_SOLVER_LIBRARY`；CANN头文件及ascendcl由当前环境查找。

### 官方文件样例

在仓库根目录运行，以下命令使用32×32输入和设备0：

```bash
python3 test/sgetrf/data/gen_data.py 32 32
LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH" ./build/test/sgetrf/sgetrf_test --files 32 32 0
python3 test/sgetrf/data/verify_result.py 32 32
```

文件模式参数为 `--files m n [device_id]`。生成脚本提供紧凑行主序输入，测试程序转换为Device列主序，
查询并分配workspace，执行Sgetrf后同步并回读LU、1-based主元及INFO。
输出分别写入 `data/output/A_gm.bin`、`W_gm.bin`、`INFO_gm.bin`；普通样例要求实际INFO=0。
校验脚本读取原始输入、LU及主元，执行重构检查并拒绝NaN/Inf。

### 次正规主元回归

`bash build.sh --soc=ascend950 --ops=sgetrf --run` 在基础测试成功后自动执行该专项；也可单独运行：

```bash
LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH" ./build/test/sgetrf/sgetrf_subnormal_pivot_test 0
```

专项检查实际Device主元及INFO、有限的LU输出和原有相对残差。任一用例失败会令测试入口返回失败。


## 官方测试协议接入

`solver_adapter.h` 保持官方固定v2协议定义。descriptor声明 `sgetrf`、`float32`、
`device` 和 `column_major`，并给出实际公开函数地址；传输缓冲按紧凑Host行主序存放，
adapter负责Device分配和列主序转换，保留有效leading dimension padding。

调用方拥有handle和stream。`create_v2` 检查容量并准备资源；`reset_v2` 恢复输入、主元和哨兵；
`execute` 调用实际公开算子；调用方同步stream后，`readback_v2` 读取数值输出和实际INFO，Sgetrf还读取输出主元；
`destroy` 清理资源，并支持空state及部分初始化状态。

adapter返回码为0成功、1协议参数错误、2分配失败、3运行时错误、4workspace查询失败。
`dut_status` 单独返回原始算子状态码，数值结果和INFO由测试调用方判定。

## 完整自测

官方任务包全量用例由随自验证材料提供的独立 `lu950-selftest/run_package.py` 调用本adapter执行。
通过 `--package-root` 指定对应任务包，使用包内原始 `gen_data.py`、`verify_accuracy.py`、
`verify_perf.py`、冻结CPU参考和GPU性能基准；各参数和命令见该独立工程README。

精度和INFO按任务包脚本判定。性能使用 `msprof op` 采集目标接口调用的完整kernel明细，
汇总后按 `T_NPU≤T_GPU/0.35` 判定；输入生成、布局转换、搬运、输入恢复及求逆所需的前置分解
均在目标计算采集区间外完成。采集工具版本和环境兼容处理记录在自验证材料中。

基础测试的重构检查用于快速回归，全量自测的输入、参考与判据来自对应官方任务包。
逐用例精度、INFO、性能结果及原始日志随固定版本的自验证材料交付。
