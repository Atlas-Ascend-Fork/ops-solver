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
            <td align="left">按行主序存储的packed LU因子，逻辑元素A(r, c)位于A[r * lda + c]；getrf输出为列主序，复用时需重排</td>
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


- 算子约束：
    - 输入矩阵A必须为aclsolverSgetrf分解得到的packed LU因子，并按行主序传入：A(r, c)位于A[r * lda + c]。aclsolverSgetrf输出Device列主序因子；复用该结果时，须先复制到Host，再按逻辑坐标重排为行主序（A_sgetrs[r * lda + c] = A_sgetrf[c * lda_rf + r]），不能直接传入列主序缓冲区。
    - ipiv必须为aclsolverSgetrf输出的pivot信息（1-based行号，取值须在[1, n]，非法值返回ACL_ERROR_INVALID_PARAM且info置-4）。
    - A、ipiv、B、info 均为 Host 内存；接口内部搬运至 Device 计算，返回前将解 X 写回 B（详见 docs/zh/sgetrs.md）。
    - n、nrhs取值范围均为[1, 8192]；lda >= n且lda - n <= 16383；ldb >= nrhs且ldb - nrhs <= 16383（kernel以uint16_t字节计数承载行间隙）。
    - trans仅支持0（AX=B）与1（AᵀX=B），其他取值返回ACL_ERROR_INVALID_PARAM。
    - 当前仅支持FLOAT32数据类型。

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
  bash build.sh --soc=ascend910b --ops=sgetrs --run # 请按实际机型替换 --soc：Atlas A2 用 ascend910b，Atlas A3 用 ascend910_93；--run 可选，用于执行测试样例
  ```

> **运行前提**：测试二进制以 `./test/<op>/data/...` 相对路径读写数据文件，须在**仓库根目录**下执行（`gen_data.py` 已改为按脚本所在目录产出；`verify_result.py` 仍以 cwd 相对路径读取，与二进制同依赖仓库根目录）。

> **精度判定口径**：`verify_result.py` 以 `error_tol = 1e-4` 为判据（参考解与实际解逐元素比较，绝对误差和相对误差同时超 eps（eps = 1e-3）的元素占比 ≤ 0.01% 判通过，即万分之一）。

> **确定性**：`gen_data.py` 固定 `np.random.seed(149)`（与 sgetrf/cgetrf 同种子保证 LU 因子一致性）——产物逐字节可复现。

> **trans 多档覆盖（issue #637）**：测试二进制按文档承诺的 trans 取值逐档执行（实数 0：AX=B、1：AᵀX=B），每档执行前从备份恢复右端项 B；输出按档落盘 `output/X_gm.bin`（trans=0）与 `output/X_gm_trans1.bin`（trans=1）。`verify_result.py` 逐档读取对应输出并与转置链参考解比对（共享工具 `lu_verify_utils.solve_lu(..., trans=...)`），`gen_data.py` 同样按档生成 `golden.bin` / `golden_trans1.bin`。仅提供 `X_gm.bin` 的历史 fixture（如 Host 单测）自动只回归 trans=0 档。

执行结果如下，说明精度对比成功。
  ```bash
  [Success] Case accuracy verification passed.
  ```

`--run` 结束后会清理测试数据。直接运行二进制前需重新生成（参数需一致）：

```bash
python3 test/sgetrs/data/gen_data.py 32 32   # 参数：矩阵阶数 n、右端项列数 nrhs
./build/test/sgetrs/sgetrs_test 0 32 32        # 参数：设备号、n、nrhs
python3 test/sgetrs/data/verify_result.py 32 32 # 参数：n、nrhs
```

> **路径对称性**：`gen_data.py` 与 `verify_result.py` 已按脚本所在目录解析路径（从任意 cwd 可正确产出/读取）；测试二进制仍以 `./test/sgetrs/data/...` cwd 相对路径读取——单步运行 binary 须在仓库根目录执行（`build.sh --run` 主流程不受影响）。
