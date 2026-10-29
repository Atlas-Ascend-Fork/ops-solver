## SmatinvBatched算子实现

## 概述

Solver SmatinvBatched算子实现。

本目录包含公开Device接口的基础测试和官方固定v2协议adapter。基础测试在程序内准备小规模输入，并检查实际Device输出、INFO、padding输入、空问题、非法维数和三次串行调用的bit-wise一致性。

## 支持的产品

- Ascend 950PR

使用仓库README列出的配套CANN环境；其他产品的当前实数Device接口验证状态见接口文档。

## 目录结构介绍

```text
├── smatinv_batched
│   ├── CMakeLists.txt              // 编译工程文件
│   ├── README.md                   // 说明文档
│   ├── smatinv_batched_test.cpp              // 基础功能与边界测试
│   ├── smatinv_batched_adapter.cpp           // 固定协议与公开Device接口之间的适配
│   └── solver_adapter.h            // 官方固定v2协议定义
```

构建产物为 `smatinv_batched_test` 和 `libsmatinvbatched_adapter.so`。

## 算子描述

- 算子功能：

    SmatinvBatched算子计算实数矩阵的逆矩阵，对应的数学表达式为：
    $$C A_{\mathrm{original}} = I$$
    其中C为输出逆矩阵。输入为原始小矩阵，接口内部完成LU分解与求逆。

- 算子规格：

    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">SmatinvBatched</td>
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
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">每个方阵的行数和列数，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[batchSize]</td>
        <td align="center">Device指针</td>
        <td align="left">Device指针数组，所指矩阵为Device列主序原始矩阵</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">每个输入矩阵的列步长，lda≥max(1,n)，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">lda_inv</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">每个输出矩阵的列步长，至少为max(1,n)，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">batchSize</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">矩阵数量，Host标量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">算子输出</td>
        <td align="center">name</td>
        <td align="center">shape</td>
        <td align="center">data type</td>
        <td align="center">Description</td>
        <td align="center">format</td>
    </tr>
    <tr>
        <td align="center">Ainv</td>
        <td align="center">[batchSize]</td>
        <td align="center">Device指针</td>
        <td align="left">Device指针数组，指向独立的Device列主序逆矩阵</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[batchSize]</td>
        <td align="center">INT32</td>
        <td align="left">Device实际状态；0成功、正值奇异、负值参数错误</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">real_lu_small_kernel、real_lu_inverse_kernel</td>
    </tr>
    </table>

- 算子约束：

  - 本接口输入原始矩阵，内部完成带主元的LU分解与求逆，没有公开主元参数。
  - 输入、输出矩阵均使用Device列主序，支持合法lda、lda_inv padding；各输出矩阵彼此独立，并与所有输入和INFO分开分配。
  - 指针数组本身和所指向的矩阵均在Device；输入矩阵可位于独立分配中。
  - n大于32返回ACLSOLVER_STATUS_INVALID_VALUE；batchSize=0成功返回，n=0且batchSize>0时向全部INFO槽位写0。
  - 工作区由handle内部管理，计算接口没有调用方workspace参数。

- 调用实现

    adapter把紧凑Host行主序传输缓冲转换为Device列主序，通过 `aclsolverSmatinvBatched` 调用实际公开接口；同步后读取实际输出和INFO。布局转换、数据搬运和输入恢复在目标性能采集区间外完成。

    完整函数原型、内存容量和返回码见[SmatinvBatched接口说明](../../docs/zh/smatinv_batched.md)。

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
  bash build.sh --soc=ascend950 --ops=smatinv_batched --run
  ```

  命令构建算子库、本接口adapter和基础测试，并运行 `smatinv_batched_test`。成功时程序打印包含
  `smatinv_batched basic tests PASS` 的摘要，构建脚本打印 `[PASS] smatinv_batched_test (basic functionality and boundaries)`。

- 已构建后直接运行，参数为设备号：

  ```bash
  LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH" ./build/test/smatinv_batched/smatinv_batched_test 0
  ```

- 独立构建本目录，链接已固定的算子库：

  ```bash
  cmake -S test/smatinv_batched -B build-smatinv_batched \
    -DOPS_SOLVER_LIBRARY="$PWD/build/libops_solver.so"
  cmake --build build-smatinv_batched -j8
  LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH" ./build-smatinv_batched/smatinv_batched_test 0
  ```

  如需链接其他位置的固定算子库，替换 `OPS_SOLVER_LIBRARY`；CANN头文件及ascendcl由当前环境查找。


## 官方测试协议接入

`solver_adapter.h` 保持官方固定v2协议定义。descriptor声明 `smatinvbatched`、`float32`、
`device` 和 `column_major`，并给出实际公开函数地址；传输缓冲按紧凑Host行主序存放，
adapter负责Device分配和列主序转换，保留有效leading dimension padding。

调用方拥有handle和stream。`create_v2` 检查容量并准备资源；`reset_v2` 恢复输入、主元和哨兵；
`execute` 调用实际公开算子；调用方同步stream后，`readback_v2` 读取数值输出和实际INFO；
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
