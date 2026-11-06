# 算子列表

> 说明：
>
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](./zh/install/dir_structure.md)。
> - **算子执行硬件单元**：执行单元以表格和各算子说明为准；Cheevj包含AI Core与Host CPU数值计算分支。关于AI Core详细介绍参见[《Ascend C算子开发》](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)中"概念原理和术语 > 硬件架构与数据处理原理"。
> - **算子接口列表**：为方便调用算子，CANN提供一套C API执行算子，以aclsolver为前缀，接口风格参考cuSolver。

项目提供的所有算子分类和算子列表如下：

<table><thead>
  <tr>
    <th rowspan="2">算子分类</th>
    <th rowspan="2">算子目录</th>
    <th colspan="2">算子实现</th>
    <th rowspan="2">算子执行硬件单元</th>
    <th rowspan="2">说明</th>
  </tr>
  <tr>
    <th>op_kernel</th>
    <th>op_host</th>
  </tr></thead>
<tbody>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgetrf.md">cgetrf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度复数矩阵LU分解，P * A = L * U；使用Ascend950 Device列主序、显式工作空间与Device主元/状态。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgetri.md">cgetri</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度复数矩阵求逆，输入LU及主元，原地输出逆矩阵；使用Ascend950 Device列主序和显式工作空间。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sgetrf.md">sgetrf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度实数矩阵LU分解，P * A = L * U；支持950PR Device列主序、workspace查询和Device主元/INFO。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/spotrf.md">spotrf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度矩阵cholesky分解，计算单精度矩阵A的LL^T分解，A = L * L^T，L为下三角矩阵，L^T为上三角矩阵。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sgeqrf.md">sgeqrf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度矩阵QR分解，计算单精度矩阵A的QR分解，A = Q * R，其中Q为正交矩阵，R为上三角矩阵。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgeqrf.md">cgeqrf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>复数矩阵QR分解，计算复数矩阵A的QR分解，A = Q * R，其中Q为酉矩阵，R为上三角矩阵。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sgetri.md">sgetri</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度实数矩阵求逆，输入已分解LU及主元，原地输出逆矩阵；支持950PR Device接口和workspace查询。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cmatinv_batched.md">cmatinv_batched</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>原始复数小矩阵直接批量求逆，n≤32；Device指针数组与逐矩阵状态。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgetri_batched.md">cgetri_batched</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>从LU及主元批量求逆，支持小矩阵和大矩阵；Device指针数组、独立输出与逐矩阵状态。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cheevj.md">cheevj</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core / Host CPU（按输入分发）</td>
    <td>单精度复数Hermitian矩阵特征值分解，可选计算特征向量。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sgetrs.md">sgetrs</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度线性方程组求解，利用LU分解结果求解单精度线性方程组 AX = B。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgetrs.md">cgetrs</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>复数线性方程组求解，利用LU分解结果求解复数线性方程组 AX = B。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cpotrf.md">cpotrf</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>复数矩阵Cholesky分解，计算复数矩阵A的Cholesky分解，根据需求解 A = L * L^H 或 A = U^H * U，L为下三角矩阵，U为上三角矩阵。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cpotrs.md">cpotrs</a></td>
    <td>✓</td><td>✓</td><td>AI Core</td>
    <td>利用复数 Cholesky 因子求解线性方程组。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cpotri.md">cpotri</a></td>
    <td>✓</td><td>✓</td><td>AI Core</td>
    <td>利用复数 Cholesky 因子计算所选三角的逆矩阵。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cpotrf_batched.md">cpotrf_batched</a></td>
    <td>✓</td><td>✓</td><td>AI Core</td>
    <td>设备指针数组的批量复数 Cholesky 分解。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cpotrs_batched.md">cpotrs_batched</a></td>
    <td>✓</td><td>✓</td><td>AI Core</td>
    <td>设备指针数组的批量复数 Cholesky 求解（nrhs=1）。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sgetri_batched.md">sgetri_batched</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度实数批量求逆，输入LU和主元；950PR Device列主序，独立输出，逐矩阵INFO。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/smatinv_batched.md">smatinv_batched</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度实数小矩阵直接批量求逆，n≤32；950PR Device列主序，独立输出，逐矩阵INFO。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cholesky.md#单精度实数-cholesky">cholesky（950）</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度实数正定矩阵的 <a href="./zh/spotrf.md#ascend-950-接口">Spotrf 分解</a>、<a href="./zh/spotrs.md">Spotrs 求解</a>、<a href="./zh/spotri.md">Spotri 求逆</a>，以及 <a href="./zh/spotrf_batched.md">SpotrfBatched 批量分解</a>和 <a href="./zh/spotrs_batched.md">SpotrsBatched 批量单 RHS 求解</a>。源码和测试共用 cholesky 目录。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sorgqr.md">sorgqr</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>由 Sgeqrf 输出的 Householder 表示生成正交因子 Q 的前 n 列，Q = H(1) H(2) ... H(k)。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sormqr.md">sormqr</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>用 QR 分解的正交因子 Q 乘矩阵 C，支持从左/右施加以及是否转置共六种组合，Q 以 Householder 表示给出、不需显式展开。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/xlarft.md">xlarft</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>由 k 个基本反射子构造块反射子的三角因子 T，I - V T V^T 即对应的块反射子，支持 FORWARD/BACKWARD 两种次序。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/sgels.md">sgels</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度实数超定最小二乘求解，min ||A X - B||（m >= n），内部走 QR 分解：A = Q R 后解 X = R^-1 (Q^T B)(1:n, :)。</td>
  </tr>
</tbody></table>

复数 QR 与最小二乘接口（Ascend 950PR，AI Core）：

| 算子目录 | API 文档 | 功能 |
| --- | --- | --- |
| `src/cgeqrf/arch35` | [Cgeqrf](zh/cgeqrf.md) | COMPLEX64 QR 分解 |
| `src/cungqr/arch35` | [Cungqr](zh/cungqr.md) | Householder 反射器生成显式 Q |
| `src/cunmqr/arch35` | [Cunmqr](zh/cunmqr.md) | 左右酉变换 |
| `src/ccgels/arch35` | [CCgels](zh/ccgels.md) | 方阵及超定复数最小二乘 |

## 算子接口

ops-solver提供类似cuSolver风格的C API接口，通过handle管理上下文：

### Handle管理接口
| 接口 | 说明 |
|------|------|
| `aclsolverCreate` | 创建solver handle |
| `aclsolverDestroy` | 销毁solver handle |
| `aclsolverSetStream` | 设置计算流 |
| `aclsolverGetStream` | 获取计算流 |

### 算子计算接口

| 接口 | 说明 |
|------|------|
| `aclsolverCgetrf_bufferSize` | LU工作空间查询，Host输出复数元素数 |
| `aclsolverCgetrf` | 复数矩阵LU分解，Device列主序 |
| `aclsolverCgetri_bufferSize` | 求逆工作空间查询，Host输出复数元素数 |
| `aclsolverCgetri` | 从LU及主元原地求逆，Device列主序 |
| `aclsolverSgetrf_bufferSize` | 单矩阵LU分解workspace查询，返回FLOAT32元素数 |
| `aclsolverSgetrf` | 单精度实数矩阵LU分解，Device列主序 |
| [`aclsolverSpotrf`](zh/spotrf.md) | 单精度 Cholesky 分解；A2/A3 与 Ascend 950 使用不同原型，详见接口说明 |
| `aclsolverSgeqrf` | 单精度矩阵QR分解 |
| `aclsolverSgeqrf_bufferSize` | 单精度矩阵 QR 分解 workspace 查询（FLOAT32 元素数） |
| `aclsolverSorgqr` / `aclsolverSorgqr_bufferSize` | 显式生成正交矩阵 Q 及工作空间查询 |
| `aclsolverSormqr` / `aclsolverSormqr_bufferSize` | 正交变换乘法及工作空间查询 |
| `aclsolverXlarft` / `aclsolverXlarft_bufferSize` | 块反射子三角因子 T 生成及工作空间查询（字节） |
| `aclsolverSSgels` / `aclsolverSSgels_bufferSize` | 单精度线性最小二乘求解及工作空间查询（字节） |
| `aclsolverCgeqrf` | A2/A3 复数矩阵QR分解（七参数 C++ 接口） |
| `aclsolverSgetri_bufferSize` | 单矩阵求逆workspace查询，返回FLOAT32元素数 |
| `aclsolverSgetri` | 单精度实数矩阵求逆，输入LU和主元 |
| `aclsolverSgetriBatched` | 单精度实数批量求逆，输入LU和主元 |
| `aclsolverSmatinvBatched` | 单精度实数小矩阵直接批量求逆，n≤32 |
| `aclsolverCmatinvBatched` | 原始复数小矩阵直接批量求逆，n≤32，Device指针数组 |
| `aclsolverCgetriBatched` | 从LU及主元批量求逆，Device指针数组 |
| `aclsolverCheevj` | 单精度复数Hermitian矩阵特征值分解 |
| `aclsolverCgeqrf` / `aclsolverCgeqrf_bufferSize` | 950 复数 QR 分解及工作区查询（C 接口） |
| `aclsolverCungqr` / `aclsolverCungqr_bufferSize` | 显式 Q 构造及工作区查询 |
| `aclsolverCunmqr` / `aclsolverCunmqr_bufferSize` | 酉变换及工作区查询 |
| `aclsolverCCgels` / `aclsolverCCgels_bufferSize` | 复数最小二乘及工作区查询 |
| `aclsolverSgetrs` | 单精度线性方程组求解 |
| `aclsolverCgetrs` | 复数线性方程组求解（C ABI，复数形参为 aclFloatComplex） |
| `aclsolverCpotrf` | 复数矩阵Cholesky分解 |

### 复数 LU/求逆接口迁移

上述四个复数计算接口及两个查询接口使用 `aclsolverStatus_t`、`aclFloatComplex` 和 Device 列主序约定。
这是破坏性变更（Breaking change）：旧 Host 调用方需修改内存管理、调用顺序和返回值判断，并重新编译链接。
单矩阵 `Workspace`、主元和 `devInfo` 位于 Device；查询输出 `Lwork` 位于 Host，单位为复数元素数。
批量接口使用 Device 指针数组，逐矩阵输出 `info`，不再以 `n=32` 划分两种接口；`CgetriBatched` 消费 LU，`CmatinvBatched` 消费原始矩阵且仅支持 `n≤32`。

完整步骤、示例及与实数接口 PR !116 的合入说明见[迁移指南](./zh/complex_solver_migration.md)。
`Sgetrf`、`Sgetri` 及实数批量求逆接口已统一为 Device 列主序约定；`Cheevj` 仍按其旧约定调用，详见迁移指南。

### 单精度复数 Cholesky 接口（Ascend950）

| 接口 | 说明 |
| --- | --- |
| `aclsolverCpotrf_bufferSize` | 查询分解工作区的复数元素数 |
| `aclsolverCpotrf` | 原地 Cholesky 分解 |
| `aclsolverCpotrs` | 因子只读，B 原地更新为解 |
| `aclsolverCpotri_bufferSize` | 查询求逆工作区的复数元素数 |
| `aclsolverCpotri` | 因子原地更新为逆矩阵的所选三角 |
| `aclsolverCpotrfBatched` | 批量分解，逐矩阵 info |
| `aclsolverCpotrsBatched` | 批量单右端项求解，单个参数 info |

上述接口使用 Device 列主序矩阵；主线 C++ `aclsolverCpotrf` 重载使用 Host 行主序矩阵。
完整参数及使用约束见[Cholesky 接口说明](zh/cholesky.md)和[两种分解接口的说明](zh/cpotrf.md)。

### 单精度实数 Cholesky 接口（Ascend950）

| 接口 | 说明 |
| --- | --- |
| [`aclsolverSpotrf`](zh/spotrf.md#ascend-950-接口) | Device 列主序单精度实数 Cholesky 分解 |
| [`aclsolverSpotrf_bufferSize`](zh/spotrf.md#函数原型) | Cholesky 分解 workspace 元素数查询 |
| [`aclsolverSpotrs`](zh/spotrs.md) | 基于 Cholesky 因子求解，因子只读 |
| [`aclsolverSpotri_bufferSize`](zh/spotri.md#函数原型) | 基于 Cholesky 因子求逆的 workspace 元素数查询 |
| [`aclsolverSpotri`](zh/spotri.md) | 原地输出逆矩阵的指定三角 |
| [`aclsolverSpotrfBatched`](zh/spotrf_batched.md) | Device 指针数组批量 Cholesky 分解 |
| [`aclsolverSpotrsBatched`](zh/spotrs_batched.md) | Device 指针数组批量求解，仅支持单 RHS |
