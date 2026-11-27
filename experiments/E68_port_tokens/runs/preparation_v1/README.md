# E68：完整端口分布与均值压缩的配对实验

用户2026-10-28认可E69后续诊断的首选方案，要求准备入口并交付启动命令。本实验只准备并短查，不由助手启动正式训练，不恢复旧E65长期任务或自动唤醒。预登记范围见[plan.json](plan.json)。

## 比较与机制

两组都从原E42/E64 rectangle96×12主干从头训练，FFN dropout0.05、AdamW decay0.0003保持一致。每层新增24维模块—端口读入器，读取当前模块状态和端口几何，按真实连接权重与学得注意力聚合，输出投影零初始化、残差系数0.1。原全局模块attention和连接/分组消息会在后续层传播该信息，无直连模块不会被伪造一个精确端口位置。不会叠加E65几何消息、E66循环、E67组token或新Loss。

| 组 | 新分支可见的端口位置 |
|---|---|
| centroid_ports | 对每个模块，将所有连接端口坐标替换为该模块加权平均坐标；保留相同槽位、权重、参数与执行形状 |
| full_ports | 完整公开端口坐标与连接权重 |

均值对照中的相同token经加权池化后只保留总权重/均值能表达的信息；原17维特征相同。参数数量相同不代表有效自由度相同。完整分布由公开原始p2b/pins构建，按原总面积尺度归一化，不用自由reference坐标。通过本地复制代码维护，不跨实验导入私有算法；来源及受保护权重见[provenance.json](provenance.json)。

每个模块只打包其实际正权连接端口，重复的同一端口边先加权合并；这精确保留连线目标所需的端口分布，排除零权/未连接端口，不按TopK截断。两组使用相同的端口槽位数padding，避免将全部全局端口展开为巨大的模块×全局端口激活。

## 冻结条件与预算

使用同一v5 train16000和固定train/dev_monitor500。原八项G的项、权重、归一化、等例聚合冻结；反传、监测、best、LR和早停全部使用同一G，未增加线长项。两组seed0、batch32、FP32、相同主干/分支参数初始化、相同样本顺序；不做多种子确认。四档LR为1e-3→2.5e-4→1e-4→5e-5，第15轮起允许下降，最低档至少完成两轮；保持已有clip1与早停口径。

每组最多3小时或200轮，两组累计最多6小时；失败/恢复计入累计预算，不清零。预算包括训练入口的数据装载/验证墙钟，终末评价另计。两组独立早停，报告共同更新数曲线，不宣称实际计算时间完全相同。GPU沿用原共享冻结cache，并只在内存附加公开端口张量；不复制或修改原cache。最大模块数/端口数自动统一padding，不截断实际端口。显存不足时失败保存，不静默改batch、精度或机制。

## 启动命令

在PowerShell执行：

```powershell
Set-Location 'E:\project\ICCAD\2026\problemC'
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E68_port_tokens.pipeline --resume
```

控制器依次运行centroid_ports、full_ports；每组结束后固定唯一G best，执行raw dev2000、原flash dev500、同100例reference/current/E42无评分图，最后生成[runs/review.md](runs/review.md)。只准备入口时可预览命令：

```powershell
& '.\.conda\envs\iccad2026\python.exe' -B -m experiments.E68_port_tokens.pipeline --dry-run
```

源码须先通过[五例预检](runs/preflight/results.json)。若用户修改源码，重新运行本实验preflight；已开始的正式运行会拒绝源码变更，不能边训边改或通过重新预检绕过恢复身份核对。

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E68_port_tokens.preflight
```

训练日志在runs/centroid_ports_train.log和runs/full_ports_train.log；[state.json](state.json)记录阶段。输出分别为runs/20261028_centroid_ports_seed0和runs/20261028_full_ports_seed0。best/latest分离，latest保留优化器、调度器、RNG、样本游标、已见成员、累计预算。

按Ctrl+C或创建runs/STOP会在保存边界停止，不进入下一组。恢复时仅移除本次自己创建的STOP，然后执行同一pipeline --resume；不要删除checkpoint或重置预算。

## 如何判断是否解决问题

固定端口覆盖≤30%/≥80%的开发子群，稀疏定义为精确N内密度不高于原dev2000中位数；重点比较低覆盖且稀疏子群。完整2000例报告G/分项/重叠；既有500草稿报告无直连模块局部中心误差、reference近邻相对误差及清晰轴次序翻转。终末flash Q仅辅助，不用于选择checkpoint。每组只一次固定500后处理，不重复重选。

用户同例观感优先。若只改善富端口组，必须明确主要弱覆盖问题仍未解决；若平均G降低却难组重叠/相对关系退化，也不能宣布成功。开发集已暴露、两组都仅seed0，不宣称独立泛化或跨种子可靠性。原E42/E50/E64/E65权重保护，测试封存；两组完成即结束，无后台自动续做其他研究。
