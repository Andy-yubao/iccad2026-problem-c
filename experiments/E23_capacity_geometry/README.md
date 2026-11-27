# E23：持续参考监督下的容量对照

当前状态：用户启动的正式训练及一次终末500例评价已完成。47轮早停、best39，三次学习率下降及最低档训练均完成；训练约39.74分钟。硬合法500/500、加权Q1.060957，与E17同版后处理1.060518基本持平。保留E25，不更换当前最佳，不自动续训或重跑评价。[完成分析](runs/20261020_capacity160_v5_seed0/analysis.md)与[身份/结果检查](runs/20261020_capacity160_v5_seed0/completion_audit.json)记录精确证据。

容量对照的关键结论：训练监测中心MSE改善约11.6%，完整dev2000中心MSE仅改善约1.0%；开发穿透平方和拥挤项分别恶化约3.1%/9.8%。扩大参数改善训练拟合，却没有明确开发/最终Q收益，不支持继续单纯扩容，也不足以证明表示能力在所有意义上已充分。下述启动命令作为历史复现入口保留；完成结果直接复用，不再次执行。

长期自主目标在E18大量后处理对照、E19新MIB解码、E20直接500例和E21原保留模型对照后，仍未同时达到加权Q<1.04/平均单例<0.25秒。E19一致性改动被同版评价否定；本实验回到E17几何模型，仅把attention宽度96提高到160，深度仍4层，参数从497,299增至1,370,771。假设是改善初始几何可减少昂贵修复；若原始代理/最终Q不改善或额外推理抵消修复收益，则否定本候选，不能把增加参数本身当进步。

其余保持E17：v5 train16k/dev2k、原参考中心/形状/相对/接触监督持续保留，不取消参考、不建新教师集、不使用E19 MIB形状聚合。只读复用E17准备的原参考张量缓存，身份与来源在provenance.json中严格核验；不跨实验导入私有算法。随机seed0从头训练，batch32、四档学习率1e-3/2.5e-4/1e-4/5e-5，15轮保护、3轮下降、8轮早停且实际3次下降与最低档完整训练验证。资源上限4小时/200轮。

模型选取仍为固定dev_monitor500的E10 geometry+occupancy，不在训练中运行后处理。训练结束冻结唯一best、核查完整dev2000原始代理，再以未改E17同版后处理做一次主开发500例对照；之后再评估快速算法，不按Q更换checkpoint或续训。新训练与正式CPU计时分时运行，训练期间只做明确标注的轻量机制诊断。

```powershell
& .\experiments\E23_capacity_geometry\launch.ps1 -Resume
```

五例实现检查及正式训练/终末评价均已完成。E24最终候选已整合为E25，本实验保持原定E17后处理对照，避免同时改变容量和后处理。历史实现检查覆盖batch32有限梯度、E10目标一致、四档LR、断点恢复、输入隔离及2例评价硬合法，见[runs/20261020_smoke5_seed0/results.json](runs/20261020_smoke5_seed0/results.json)。正式模型目录为runs/20261020_capacity160_v5_seed0；原权重与数据不覆盖，测试继续封存。

## 清理后的启动与日志

在工作区根目录运行上述`launch.ps1 -Resume`：无正式断点时从头训练，存在兼容断点时恢复，随后自动执行一次冻结best的500例开发评价。预算4小时/200轮，batch32，四档学习率不变；终末评价2进程×2线程。正式输出为`runs/20261020_capacity160_v5_seed0`。

如需同时显示并保存输出：

```powershell
& .\experiments\E23_capacity_geometry\launch.ps1 -Resume 2>&1 | Tee-Object -FilePath .\experiments\E23_capacity_geometry\runs\training_console.log -Append
```

日志为`runs/training_console.log`，分阶段状态为`runs/20261020_background/pipeline.json`。原来文中stdout/stderr仅指手动后台重定向时的位置，当前命令统一以上述日志为准。训练中请求安全停止可创建正式运行目录下的`STOP`文件；恢复前删除该STOP并重用`-Resume`，不重置累计预算。迁移校验在[E25迁移检查](../E25_integrated_baseline/runs/20261020_migration_check/results.json)，原smoke断点是实现检查，不是正式权重，不用于正式续训。
