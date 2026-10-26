## Cgeqrf算子实现

## 概述

Solver Cgeqrf算子实现。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```text
├── cgeqrf
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   ├── cgeqrf_test.cpp     // 算子调用样例
│   └── data                // 测试数据目录
│       ├── gen_data.py     // 生成测试数据脚本
│       └── verify_result.py // 验证结果脚本
```

## 算子描述

- 算子功能：

    Cgeqrf算子对单精度复数矩阵进行QR分解，对应的数学表达式为：

    $$
    A = Q \cdot R
    $$

    其中$A$为$m \times n$阶单精度复数矩阵，$Q$为酉矩阵，$R$为上三角矩阵，满足：

    $$
    Q^H Q = I
    $$

    其中，$Q^H$表示矩阵$Q$的共轭转置。该算子使用Householder反射表示$Q$矩阵，输出的$R$矩阵和$Q$矩阵的Householder向量合并存储在$A$中，Householder反射系数存储在$tau$中。

    复数矩阵的实部和虚部均为单精度浮点数，数据类型为`COMPLEX64`。

- 算子规格：

    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">Cgeqrf</td>
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
        <td align="center">m</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">矩阵A的行数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">矩阵A的列数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[m, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输入矩阵A，列主序</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">A的主维度，表示相邻列首元素之间的内存地址偏移量</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="3" align="center">算子输出</td>
        <td align="center">A</td>
        <td align="center">[m, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">输出的R矩阵和Q矩阵的Householder向量，合并存储在A中</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">tau</td>
        <td align="center">[min(m,n)]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">Q矩阵的Householder反射系数</td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">分解结果信息</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">cgeqrf_kernel</td>
    </tr>
    </table>

- 算子约束：

  - `info`参数在当前版本实际未启用。
  - 输入矩阵A按列主序存储。
  - A和tau的数据类型仅支持`COMPLEX64`。
  - `COMPLEX64`数据类型的实部和虚部均为`FLOAT32`。

- 调用实现

    使用内核调用符`<<<>>>`调用核函数。

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

  - 指定路径`install_path`，安装CANN软件包

    ```bash
    source ${install_path}/cann/set_env.sh
    ```

- 样例执行

  ```bash
  bash build.sh --ops=cgeqrf --run # --ops=<算子名> --run可选参数，执行测试样例
  ```

  执行结果如下，说明精度对比成功。

  ```text
  [Success] Case accuracy is verification passed.
  ```