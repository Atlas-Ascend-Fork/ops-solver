# 算子列表

> 说明：
>
> - **算子目录**：目录名为算子名小写下划线形式，每个目录承载该算子所有交付件，包括代码实现、examples、文档等，目录介绍参见[项目目录](../README.md#目录结构)。
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
    <td>复数矩阵LU分解，计算复数矩阵A的LU分解，A = P * L * U，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgetri.md">cgetri</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>复数矩阵求逆，利用LU分解计算复数矩阵的逆矩阵。</td>
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
    <td>批量复数矩阵求逆，对一批复数矩阵同时进行求逆运算，提高计算效率。</td>
  </tr>
  <tr>
    <td>solver</td>
    <td><a href="./zh/cgetri_batched.md">cgetri_batched</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>批量复数矩阵求逆，适用于矩阵维度较大（n≥32）的场景，对一批复数矩阵同时进行求逆运算。</td>
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
</tbody></table>


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
| `aclsolverCgetrf` | 复数矩阵LU分解 |
| `aclsolverCgetri` | 复数矩阵求逆 |
| `aclsolverSgetrf_bufferSize` | 单矩阵LU分解workspace查询，返回FLOAT32元素数 |
| `aclsolverSgetrf` | 单精度实数矩阵LU分解，Device列主序 |
| [`aclsolverSpotrf`](zh/spotrf.md) | 单精度 Cholesky 分解；A2/A3 与 Ascend 950 使用不同原型，详见接口说明 |
| `aclsolverSgeqrf` | 单精度矩阵QR分解 |
| `aclsolverCgeqrf` | 复数矩阵QR分解 |
| `aclsolverSgetri_bufferSize` | 单矩阵求逆workspace查询，返回FLOAT32元素数 |
| `aclsolverSgetri` | 单精度实数矩阵求逆，输入LU和主元 |
| `aclsolverSgetriBatched` | 单精度实数批量求逆，输入LU和主元 |
| `aclsolverSmatinvBatched` | 单精度实数小矩阵直接批量求逆，n≤32 |
| `aclsolverCmatinvBatched` | 批量复数矩阵求逆（小矩阵，n≤32） |
| `aclsolverCgetriBatched` | 批量复数矩阵求逆（大矩阵，n>32） |
| `aclsolverCheevj` | 单精度复数Hermitian矩阵特征值分解 |
| `aclsolverSgetrs` | 单精度线性方程组求解 |
| `aclsolverCgetrs` | 复数线性方程组求解 |
| `aclsolverCpotrf` | 复数矩阵Cholesky分解 |

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
