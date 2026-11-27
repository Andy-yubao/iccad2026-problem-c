2026-10-26已完成：44轮早停／best44，后训练32.86分钟；同dev500固定G 0.444657→0.354743，flash500全硬合法、加权Q1.033945。相对E55改善；与E42相比P95和最差尾部更好，但加权Q和均时变差，未替换基线。[结果复核与曲线](runs/20261026_e55_best40_e42loss_seed0/review.md)、100张无评分图已完成，E55原best/latest保持不变。

# E56：E55 best40 的 E42 Loss 后训练

2026-10-26用户要求“对E55进行后训练（要保留原始参数），Loss用和E42相同的”。从冻结E55 best40只加载模型权重，输出全部写入本实验，E55 best/latest与历史运行记录不覆盖。单独记录来源SHA256并在流程开始、结束复核。

## 目标与历史口径

使用E42真正反向传播的八项Loss及原权重：

```text
G = 1.00 center_local + 0.25 shape + 0.25 relative_local
  + 0.50 overlap + 0.50 contact + 0.25 boundary
  + 0.10 mib + 0.10 bbox
```

各项定义、mask、局部尺度、布局内聚合与布局等权平均沿用E42；其中bbox为参考外框面积比例损失。Loss独立复制在本实验内，不跨实验导入私有算法。CUDA图仅使用已验证的等价算子替换，独立复制的原E42 Loss作为短检查参照。

E42历史上前两轮对五项渐入，且用G+occupancy作选模代理。按当前唯一固定Loss规范，本轮从第一步启用完整G，训练反传、train_monitor500、dev_monitor500、best／LR／早停共用G；没有渐入，也不加入仅供旧选模的occupancy四项。这是E42训练目标的后训练，不宣称完整复现其历史训练协议。

## 参数、状态与预算

- 起点：E55 `runs/20261026_reference_position_shape_seed0/checkpoints/best.pt`，best40，SHA256 `714cc2aa62ccc84c6b2765d93b304721f29465305568ce8256d42298daed5cb3`。不从latest43出发；原权重保留在原路径，不另造模型仓库。
- 96×12 rectangle、FFN dropout .05、FP32 CUDA graph、batch32、AdamW decay .0003、clip1保持不变。新目标使用新AdamW动量、调度／早停记录；不继承旧目标已经早停的优化器状态。来源、预训练耗时与新阶段累计耗时分别保存。
- 学习率按既定1e-3→2.5e-4→1e-4→5e-5，第15轮起按原平台规则下降／早停。前期可能扰动既有参考拟合，需要同时看初始和最佳固定面板曲线，不根据单batch调参。
- 新阶段最多2小时／200轮，预计约30–120分钟；恢复只延续本阶段原状态和累计预算。源E55已花费的训练时间另外记录，不抹去既有预算。
- v5 train16000；固定train_monitor500／dev_monitor500；共享E42缓存原位引用，测试继续封存。

## 执行与验收

沿用本会话的后台串行流程：[pipeline.py](pipeline.py)依次训练→唯一Loss-best的同dev500原flash检查→同100例reference／本轮draft／E42 draft无评分图→报告。原E55和E42已存结果直接复用作比较，不重跑对照，不按终末Q重选参数。结果未达到要求也保留并报告，不自动扩展新实验或恢复旧自主任务。

输出为`runs/20261026_e55_best40_e42loss_seed0/`，控制台／状态在同名`_control/`。训练保留best和latest；完成报告为`analysis.md`和`analysis.json`，终末输出为`terminal/results.json`。首先比较用户草稿反馈、G及共同的参考分项、重叠和修复负担，再看Q和耗时；纯参考Loss与G总数值不能直接比较。后训练改善也不能单凭本单seed归因于课程顺序，需要额外控制方可建立因果结论。

```powershell
& 'E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E56_e55_geometry_finetune.pipeline --resume
```

在项目根目录执行。命令只恢复未完成阶段；完成训练和终末输出会复用，不覆盖历史结果。安全停止：在本实验`runs/STOP`或本轮输出目录下创建`STOP`，当前保存边界后停止且不启动后续阶段；移除自己创建的STOP后再用同一命令恢复，身份与累计预算校验保持严格。

[来源与受保护权重](provenance.json) · [固定Loss配置](loss_config.py) · [五例短检查](runs/20261026_preflight/results.json)
