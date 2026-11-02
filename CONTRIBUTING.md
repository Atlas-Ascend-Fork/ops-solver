# 贡献指南

## 了解行为准则

本项目欢迎广大开发者体验并参与贡献，请在参与社区贡献之前参见[cann-community](https://gitcode.com/cann/community)先了解行为准则，进行CLA协议签署，了解源码仓的贡献流程。

## 签署CLA

在参与项目贡献前，您需要签署CANN开放项目贡献者许可协议（CLA）。

请根据您的参与身份，选择签署法人CLA、法人贡献者CLA、个人CLA、企业管理员CLA，请点击[这里](https://clasign.osinfra.cn/sign/68cbd4a3dbabc050b436cdd4)签署。

- 法人CLA：以企业身份参与贡献，代表企业签署CLA，一般由企业管理人员进行签署。
- 法人贡献者CLA：如果您属于签署了法人CLA的企业员工，请申请签署法人贡献者CLA，在申请页面选择您所属的企业，申请后将由企业管理员进行审批，审批完成即可参与贡献。
- 个人CLA：以非企业员工的个人身份参与贡献，请签署个人CLA。
- 企业管理员：以企业管理员的身份参与贡献，请签署企业管理员CLA，企业管理员有权对签署法人贡献者CLA的申请进行审批和人员管理。

## 参与贡献

在签署了CLA协议之后，就可以开始您的贡献之旅啦！贡献的方式有很多种，每一种贡献都将受到欢迎和重视。

所有您发现的问题或想贡献的新想法都可以通过[Issue](#提交issue)进行反馈、讨论和跟踪，并在后续[贡献编码](#贡献编码) Pull-Request 合入后关闭关联Issue。

> 📝 **提示**
>
> - 当您在提交PR过程中遇到问题，可在 [Issue](https://gitcode.com/cann/ops-solver/issues) 中提问。

### 贡献分类

- 算子bug修复
   
  如果您在本仓库中发现了某些算子Bug，并想对其进行修复，欢迎您在仓库中新建Issue进行反馈和跟踪处理。

  您可以按照下方[提交Issue/处理Issue任务](#提交issue)指引新建 `Bug-Report|缺陷反馈` 类Issue对Bug进行描述，
  然后在评论框中输入“/assign”或“/assign @yourself”将该Issue分配给您进行处理。

- 算子优化
   
  如果您对本仓库中某些算子实现有泛化性增强/性能优化思路，并想着手实现这些优化点，欢迎您对算子进行优化贡献。

  您可以按照下方[提交Issue/处理Issue任务](#提交issue)指引新建 `Requirement|需求建议` 类Issue对优化点进行说明，并提供您的设计方案，
  然后在评论框中输入“/assign”或“/assign @yourself”将该Issue分配给您进行跟踪优化。

- 贡献新算子
   
  如果您有全新的算子想基于昇腾芯片进行设计实现，欢迎您在Issue中提出新的想法和设计，并与昇腾团队成员进行交流讨论。

  您可以按照下方[提交Issue/处理Issue任务](#提交issue)指引新建 `Requirement|需求建议` 类Issue提供您的新算子说明和设计方案，
  昇腾团队成员会与您进行沟通确认，并为您的算子提供一个合适的`contrib`目录分类，您可以将您的新算子贡献到对应分类目录下。

  同时，您需要在提交的Issue中评论“/assign”或“/assign @yourself”，认领该Issue并在后续将新算子提交到代码仓库。

- 文档纠错
   
  如果您在仓库中发现某些算子文档描述错误，欢迎您在仓库中新建Issue进行反馈和修复。

  您可以按照下方[提交Issue/处理Issue任务](#提交issue)指引新建 `Documentation|文档反馈` 类Issue指出对应文档中的问题，
  然后在评论框中输入“/assign”或“/assign @yourself”将该Issue分配给您纠正对应文档描述。

- 帮助解决他人Issue
   
  如果社区中他人遇到的问题您有合适的解决方法，欢迎您在Issue中发表评论交流，帮助他人解决问题和痛点，共同优化易用性。

  如果对应Issue需要进行代码修改，您可以在Issue评论框中输入“/assign”或“/assign @yourself”将该Issue分配给您，跟踪协助解决问题。

### <a name="提交issue"></a>提交Issue/处理Issue任务

- 找到Issue列表：
  
  在[ops-solver](https://gitcode.com/cann/ops-solver)项目Gitcode主页内，点击“Issues”，即可找到 Issue 列表。

- 提交Issue
  
  如果您准备向社区上报Bug或者提交需求，或者为社区贡献自己的意见或建议，请提交[Issue](https://gitcode.com/cann/ops-solver/issues)。

- 参与Issue讨论

  每个Issue下面都支持开发者们交流讨论，如果您感兴趣，可以在评论区中发表自己的意见。

- 找到愿意处理的Issue

  如果您愿意处理其中的一个 Issue，可以将它分配给自己。只需要在评论框内输入“/assign”或 “/assign @yourself”，机器人就会将问题分配给您，您的名字将出现在负责人列表里。

### 贡献编码

1. 准备CANN开发环境
  
   如果您想参与编码贡献，需要准备CANN开发环境，请参考[环境准备](./docs/zh/install/quick_install.md)。

2. 了解ops-solver的开发注意事项

   1）请参考[工具版本要求与安装](./docs/zh/install/quick_install.md#安装基础依赖)，了解编码贡献的一些环境和工具要求。

   2）ops-solver软件编码遵循许可协议：CANN Open Software License Agreement Version 2.0，详细的协议说明请参见[LICENSE](./LICENSE)文件，如果您贡献代码到ops-solver源码仓，请遵循此协议。
   
     请在新建的cpp、cc、h等源码文件头部增加如下声明：
   
     ```txt
     /**
      * Copyright (c) 2026 [Name of the copyright owner] All rights reserved.
      * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
      * CANN Open Software License Agreement Version 2.0 (the "License").
      * Please refer to the License for details. You may not use this file except in compliance with the License.
      * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
      * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
      * See LICENSE in the root of the software repository for the full text of the License.
      */
     ```
     
     请在新建的py、sh等文件头部增加如下声明：
     
     ```txt
     # Copyright (c) 2026 [Name of the copyright owner] All rights reserved.
     # This program is free software, you can redistribute it and/or modify it under the terms and conditions of
     # CANN Open Software License Agreement Version 2.0 (the "License").
     # Please refer to the License for details. You may not use this file except in compliance with the License.
     # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
     # INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
     # See LICENSE in the root of the software repository for the full text of the License.
     # ================================================================================================================
     ```

    - 如果您仅代表个人贡献，并且您个人拥有贡献内容的版权，请将第一行中的 `[Name of the copyright owner]` 替换为您的署名。
    - 如果您代表您的雇主贡献，或者您的雇主拥有您贡献内容的版权，请将第一行中的 `[Name of the copyright owner]` 替换为您的雇主的名称。
      
      如果您对于贡献内容的版权归属存在任何疑问，请您咨询法律顾问或您雇主的法律团队。
      
    - 第一行中`2026`为您创建或修改该文件的年份，请根据实际时间修改。

3. 代码下载与贡献流程
 
   (1) 进行代码开发前，请先将需要ops-solver仓库fork到个人仓库，然后将个人仓库下载到本地。并在本地分支进行代码修改。    
   (2) 代码验证满足贡献要求后，提交Pull-Request，将代码贡献到ops-solver，在[Pull-Request列表](https://gitcode.com/cann/ops-solver/pulls)，可以找到提交的Pull-Request。  
   (3) 在提交的Pull-Request的评论区，评论`compile`触发编译。  
   (4) 请注意查看门禁测试结果，若未通过，请根据问题提示进行本地代码修改；若通过，此PR会被分配给committer检视，请关注committer的检视意见。  
   (5) 当您的PR检视通过后，代码会合入ops-solver源码仓。
