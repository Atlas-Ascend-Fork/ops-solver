## Cgetri算子实现

## 概述

Solver Cgetri算子实现。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── cgetri
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   ├── cgetri_test.cpp     // 算子调用样例
│   └── data                // 测试数据目录
│       ├── gen_data.py     // 生成测试数据脚本
│       └── verify_result.py // 验证结果脚本
```

## 算子描述

- 算子功能：
    Cgetri算子计算复数矩阵的逆矩阵，对应的数学表达式为：
    $$A^{-1}A = I$$
    其中$A$为$n \times n$阶非奇异复数方阵，$I$为$n$阶单位矩阵。

- 算子规格：
    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">Cgetri</td>
    </tr>
    <tr>
        <td rowspan="5" align="center">算子输入</td>
        <td align="center">name</td>
        <td align="center">shape</td>
        <td align="center">data type</td>
        <td align="left">Description</td>
        <td align="center">format</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">矩阵A的行数和列数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[n, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输入为待求逆矩阵经 aclsolverCgetrf 分解后的 packed LU 因子（列主序）；本算子消费该因子与 devIpiv 完成求逆，返回时 A 原地覆写为逆矩阵（与 docs/zh/cgetri.md 一致：输入原始矩阵时应先调用 aclsolverCgetrf）</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">devIpiv</td>
        <td align="center">[n]</td>
        <td align="center">INT32</td>
        <td align="left">aclsolverCgetrf 输出的 1-based 行交换索引，Device 内存；nullptr 仅用于无主元 LU；非法取值（不在对应行范围内）时 devInfo 置 -4 并返回 ACLSOLVER_STATUS_INVALID_VALUE</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">A同一列中相邻两行元素间的内存地址偏移量（leading dimension），须满足 lda >= max(1, n)</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">求逆结果信息</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="2" align="center">算子输出</td>
        <td align="center">A</td>
        <td align="center">[n, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输出的逆矩阵</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">求逆结果信息</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">cgetri_kernel</td>
    </tr>
    </table>


- 算子约束：
  - n 的有效范围为 [0, 4096]（docs/zh/cgetri.md 与 src/cgetri/cgetri_host.cpp InvalidShape 现行口径）。
  - lda >= max(1, n)，支持合法 padding（不要求连续布局）。
  - A、Workspace、devIpiv、devInfo 均为 Device 内存；输入须为 aclsolverCgetrf 输出的 LU 因子，devIpiv 为其 1-based 主元（nullptr 表示无主元 LU）。

- 调用实现
    使用内核调用符<<<>>>调用核函数。

## 编译运行

在本样例根目录下执行如下步骤，编译并执行算子。
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
  bash build.sh --soc=ascend950 --ops=cgetri --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy verification passed.
  ```
