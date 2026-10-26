## Cpotrf算子实现

## 概述

Solver Cpotrf算子实现。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```text
├── cpotrf
│   ├── CMakeLists.txt        // 编译工程文件
│   ├── README.md             // 说明文档
│   ├── cpotrf_test.cpp       // 算子调用样例
│   └── data                  // 测试数据目录
│       ├── gen_data.py       // 生成测试数据脚本
│       └── verify_result.py  // 验证结果脚本
```

## 算子描述

- 算子功能：

    Cpotrf算子对Hermitian正定复数矩阵进行Cholesky分解。

    当`uplo`为`U`时，计算上三角Cholesky因子$U$，对应的数学表达式为：

    $$A = U^H \cdot U$$

    当`uplo`为`L`时，计算下三角Cholesky因子$L$，对应的数学表达式为：

    $$A = L \cdot L^H$$

    其中，$A$为$n \times n$阶Hermitian正定复数矩阵，$U$为上三角矩阵，$L$为下三角矩阵，$U^H$和$L^H$表示矩阵的共轭转置。

- 算子规格：

    <table>
    <tr>
        <td rowspan="1" align="center">算子类型(OpType)</td>
        <td colspan="5" align="center">Cpotrf</td>
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
        <td align="center">uplo</td>
        <td align="center">[1]</td>
        <td align="center">CHAR</td>
        <td align="left">
            指定输出在矩阵A的上三角部分或下三角部分。<br>
            U：输出上三角Cholesky因子。<br>
            L：输出下三角Cholesky因子。
        </td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">n</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">矩阵A的阶数，即矩阵A的行数和列数</td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td align="center">A</td>
        <td align="center">[n, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">
            输入的n阶Hermitian正定复数矩阵A，采用行主序存储。<br>
        </td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">lda</td>
        <td align="center">[1]</td>
        <td align="center">INT64</td>
        <td align="left">
            矩阵A中相邻两行起始地址之间的元素跨度，以COMPLEX64元素为单位，要求lda不小于n。<br>
            当lda大于n时，每行前n个元素为有效矩阵数据，其余元素为填充区域。
        </td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="2" align="center">算子输出</td>
        <td align="center">A</td>
        <td align="center">[n, n]</td>
        <td align="center">COMPLEX64</td>
        <td align="left">
            Cholesky分解结果，采用行主序存储。<br>
            当uplo为U时，矩阵A的上三角部分存储上三角因子U；当uplo为L时，矩阵A的下三角部分存储下三角因子L。
        </td>
        <td align="center">ND</td>
    </tr>
    <tr>
        <td align="center">info</td>
        <td align="center">[1]</td>
        <td align="center">INT32</td>
        <td align="left">
            分解结果信息。<br>
            info等于0表示分解成功；info大于0表示矩阵不是Hermitian正定矩阵，分解未成功。
        </td>
        <td align="center">\</td>
    </tr>
    <tr>
        <td rowspan="1" align="center">核函数名</td>
        <td colspan="5" align="center">cpotrf_kernel</td>
    </tr>
    </table>

- 算子约束：

    - 输入矩阵`A`必须为$n \times n$阶Hermitian正定复数矩阵，上三角必须有值。
    - `uplo`仅支持`U`、`L`。
    - `n`必须大于0。
    - `lda`必须满足`lda >= n`。
    - `lda`的单位为`COMPLEX64`元素，不是字节数或`float`元素数。
    - 矩阵`A`采用行主序存储，第$i$行第$j$列元素的地址偏移为`i * lda + j`。
    - 当`lda > n`时，每行索引范围`[n, lda)`内的元素属于填充区域，不参与Cholesky分解。
    - 当`uplo`为`U`时，只使用输入矩阵的上三角部分，输出矩阵的上三角部分有效。
    - 当`uplo`为`L`时，只使用输入矩阵的下三角部分，输出矩阵的下三角部分有效。
    - 输入矩阵的对角元素必须为实数。对于Hermitian矩阵，其对角元素的虚部应为0。
    - 输入矩阵`A`会被Cholesky分解结果原地覆盖。

- 调用实现：
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
  bash build.sh --ops=cpotrf --run # --ops=<算子名> --run为可选参数，用于执行测试样例
  ```

  执行结果如下，说明精度对比成功。

  ```bash
  [Success] Case accuracy is verification passed.
  ```