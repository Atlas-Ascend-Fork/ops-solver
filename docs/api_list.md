# 算子列表

> 说明：
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
    <td>单精度矩阵LU分解，计算单精度矩阵A的LU分解，A = P * L * U，其中P为置换矩阵，L为下三角矩阵，U为上三角矩阵。</td>
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
    <td><a href="./zh/sgetri.md">sgetri</a></td>
    <td>✓</td>
    <td>✓</td>
    <td>AI Core</td>
    <td>单精度矩阵求逆，利用LU分解计算单精度矩阵的逆矩阵。</td>
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
| `aclsolverSgetrf` | 单精度矩阵LU分解 |
| `aclsolverSgeqrf` | 单精度矩阵QR分解 |
| `aclsolverSgetri` | 单精度矩阵求逆 |
| `aclsolverCmatinvBatched` | 批量复数矩阵求逆（小矩阵，n<32） |
| `aclsolverCgetriBatched` | 批量复数矩阵求逆（大矩阵，n≥32） |
| `aclsolverCheevj` | 单精度复数Hermitian矩阵特征值分解 |
