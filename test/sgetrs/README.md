## Sgetrs算子实现

## 概述

Solver Sgetrs算子实现。

## 支持的产品

- Atlas A3 训练系列产品/Atlas A3 推理系列产品
- Atlas A2 训练系列产品/Atlas A2 推理系列产品

## 目录结构介绍

```
├── sgetrs
│   ├── CMakeLists.txt      // 编译工程文件
│   ├── README.md           // 说明文档
│   ├── sgetrs_test.cpp     // 算子调用样例
│   └── data                // 测试数据目录
│       ├── gen_data.py     // 生成测试数据脚本
│       └── verify_result.py // 验证结果脚本
```

## 算子描述

- 算子功能：
    Sgetrs算子利用Sgetrf算子计算得到的LU分解结果求解线性方程组，对应的数学表达式为：
    $$A \cdot X = B$$
    其中$A$为方阵（$n \times n$阶），$B$为右手边矩阵（$n \times nrhs$阶），$X$为解矩阵。

- 算子规格：
    <table>
        <tr>
            <td rowspan="1" align="center">算子类型(OpType)</td>
            <td colspan="5" align="center">Sgetrs</td>
        </tr>
        <tr>
            <td rowspan="9" align="center">算子输入</td>
            <td align="center">name</td>
            <td align="center">shape</td>
            <td align="center">data type</td>
            <td align="left">Description</td>
            <td align="center">format</td>
        </tr>
        <tr>
            <td align="center">trans</td>
            <td align="center">[1]</td>
            <td align="center">INT32</td>
            <td align="left">指定求解类型（0: AX=B, 1: AᵀX=B）</td>
            <td align="center">\</td>
        </tr>
        <tr>
            <td align="center">n</td>
            <td align="center">[1]</td>
            <td align="center">INT64</td>
            <td align="left">矩阵A的阶数</td>
            <td align="center">\</td>
        </tr>
        <tr>
            <td align="center">nrhs</td>
            <td align="center">[1]</td>
            <td align="center">INT64</td>
            <td align="left">右端项矩阵B的列数</td>
            <td align="center">\</td>
        </tr>
        <tr>
            <td align="center">A</td>
            <td align="center">[n, n]</td>
            <td align="center">FLOAT32</td>
            <td align="left">LU分解后的矩阵（由getrf输出）</td>
            <td align="center">ND</td>
        </tr>
        <tr>
            <td align="center">lda</td>
            <td align="center">[1]</td>
            <td align="center">INT64</td>
            <td align="left">A的leading dimension（通常为n）</td>
            <td align="center">\</td>
        </tr>
        <tr>
            <td align="center">ipiv</td>
            <td align="center">[n]</td>
            <td align="center">INT32</td>
            <td align="left">LU分解得到的pivot信息</td>
            <td align="center">ND</td>
        </tr>
        <tr>
            <td align="center">B</td>
            <td align="center">[n, nrhs]</td>
            <td align="center">FLOAT32</td>
            <td align="left">右端项矩阵B（输入时为B，输出时被覆盖为X）</td>
            <td align="center">ND</td>
        </tr>
        <tr>
            <td align="center">ldb</td>
            <td align="center">[1]</td>
            <td align="center">INT64</td>
            <td align="left">B的leading dimension（通常为nrhs）</td>
            <td align="center">\</td>
        </tr>
        <tr>
            <td rowspan="2" align="center">算子输出</td>
            <td align="center">X</td>
            <td align="center">[n, nrhs]</td>
            <td align="center">FLOAT32</td>
            <td align="left">方程AX=B的解（结果覆盖B）</td>
            <td align="center">ND</td>
        </tr>
        <tr>
            <td align="center">info</td>
            <td align="center">[1]</td>
            <td align="center">INT32</td>
            <td align="left">执行状态信息（0表示成功）</td>
            <td align="center">\</td>
        </tr>
        <tr>
            <td rowspan="1" align="center">核函数名</td>
            <td colspan="5" align="center">sgetrs_kernel</td>
        </tr>
    </table>


- 算子约束：无。

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
  bash build.sh --ops=sgetrs --run # --ops=<算子名> --run可选参数，执行测试样例
  ```
  执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy is verification passed.
  ```
