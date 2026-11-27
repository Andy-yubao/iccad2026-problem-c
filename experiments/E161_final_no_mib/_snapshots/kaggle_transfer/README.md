# E161 Kaggle迁移

2026-11-23台北16:43：已通过Kaggle API核实E161四阶段训练完成并回传全部70个远端文件（97962747字节）。四阶段best均通过冻结SHA256与本地CPU读取检查；S1完成43轮/预算结束、best40，S2完成26轮/正常早停、best21（own-Loss0.247930124）。S1/S2 latest含终末模型/Adam/RNG/游标/调度/投影及累计预算。远端因打包漏带evaluation/budget.json在dev2000入口导入阶段出错，2000例未开始、没有Q结果；原evaluation/plan.json也未入包，两份配置已保留供本地评价。原本地暂停运行未覆盖。远端版本1已ERROR终止，额度接口两次确认GPU time_reserved=0秒，无需取消已终止会话；未删除Notebook/数据集/结果。后续在本地继续，本次未启动本地评价。[本地回传记录](downloaded/20261123T084205Z/RETURN_TO_LOCAL.md)、[哈希与恢复审核](local_return_audit.json)。无需重传、重训或重新部署；以下为历史过程。

已准备私有数据集上传包与私有GPU脚本。训练复用原train36000/dev监测的完整静态张量及role缓存，不重复搬运原train NPZ；终末评价所需的原dev2000 NPZ单独保留，test载荷不打包。原L0/S0冻结best和S1第7568次更新latest/best保留；本地暂停标记及源码/权重不修改。远端接续S1后训练S2并自动原reference/no-MIB dev2000。

项目根目录的PowerShell运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\experiments\E161_kaggle_transfer\deploy.ps1
```

脚本提示输入Kaggle用户名；使用当前PATH中的Python3.11+创建独立`.cache/kaggle-cli-py311plus`环境，明确安装`kaggle>=1.8.4`，检查版本后再隐藏读取Token（仅存在子进程环境中）。可用`-CliPython <python.exe>`指定解释器。上传私有Dataset后提交私有T4×2 Notebook。只暴露GPU0，保持原batch64/FP32/seed/Loss，不隐式使用第二张卡。需要账户已取得GPU资格且有可用配额；不购买资源。GPU提交是正式训练，不运行试训/模型探测。

2026-11-23修复：首次本机部署使用训练Python3.10，pip因此选择不识别`KAGGLE_API_TOKEN`的kaggle1.7.4.5，认证阶段报缺失kaggle.json，尚未上传。当前脚本已改用独立Python3.11+工具环境（本机base为Python3.13.9），旧CLI环境和训练环境保留；0.98GB数据包无需重建。本修复不改变模型、数据或暂停状态。

PyTorch2.7.1+cu128和NumPy1.26.4等版本固定。仅在远端副本登记GPU身份迁移：原checkpoint另存，新的latest只改identity中的GPU名字；模型、Adam、RNG、排列/游标、LR/早停、投影和累计预算全部保留。跨GPU浮点与随机数执行不保证逐位相同，不冒充本地精确延续。软件/源码身份不一致直接停止，不重建缓存或Adam。S1已用4621.447秒计入原4h预算，剩余约2h43m；S2原4h不变。

数据/代码临时解包到`/tmp`，输出保存在`/kaggle/working/e161-results`，避免把7GB缓存再作为Notebook输出保存。会话约8h20时请求安全暂停，为平台时限留存盘余量。错误或暂停阻断后续，不自动重试/新增seed。安装、传输、解包与训练预算分别记录；上传包本身仍保存原始文件哈希。

完成后运行`download_results.ps1`，下载到独立目录，不覆盖本地E161。任务地址和提交状态由实际提交后生成的`submission.json`记录。未观察实际训练更新前，不把Kaggle的RUNNING状态写为训练成功。

本次未执行模型、训练、推理、评估或防御性测试；部署脚本及远端兼容性尚未经实际远端运行验证。

实际准备结果：上传包982,174,149字节（约0.98 GB），2766个文件，test载荷0。包SHA256为`614bd3cc4bf8de95aab59c33547c24dc3c2322357b8b4a03394119802a02c223`。已核对打包的恢复入口/依赖清单/暂停checkpoint与当前原件一致，原E161冻结源码67项未变化；这些是文件完整性核对，不执行模型或试训。

2026-11-23依赖补修：kaggle2.2.5自身上传模块导入six，但发布元数据未声明该依赖。已从本机旧CLI环境的官方six1.17.0安装原件重建临时wheel（逐文件核对原RECORD哈希）并离线安装到新CLI环境；pip check及kaggle --help通过。部署脚本明确要求six>=1.17.0,<2，兼容安装存在时不再重复联网安装。随后使用用户授权Token直接尝试官方认证API，但执行环境访问既定代理时被WinError10013阻断；operator_attempt.json仅保存脱敏错误，不保存Token。远端仍未提交，本地保持暂停。

2026-11-23上传后继续：用户本机已成功上传937MiB包及manifest，Kaggle接受私有Dataset创建；随后GetDatasetStatus返回403，不能据此认定处理失败或已经就绪。已记录upload_receipt.json并新增deploy_api.py：核验认证用户名、私有已发布版本、两个文件可读且大小一致，状态端点单独403时可用这些正面读取证据确认；不能读取则等待并报告，不改变可见性。使用`powershell -ExecutionPolicy Bypass -File .\experiments\E161_kaggle_transfer\deploy.ps1 -Username andyyubao -UseExistingDataset`，不重传。以后新上传被接受后立即记录receipt，查询失败也不重传。GPU提交使用结构化API返回值，拒绝/缺失输入不冒充成功；有提交记录时不重复启动新版本。离线9项验证通过，服务器端读取及GPU任务仍需实际运行确认。
