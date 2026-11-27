# E6 首轮几何模型：架构 × Loss 与局部监督消融

日期：2026-10-17。正式评价完成前本报告仍为草稿；以下最终表格由同目录结果生成。

<!-- CONCLUSION -->

## 1. 问题、控制与信息边界

希望模型产生只需小幅调整的优秀布局，优先改模型，再设计能保留其几何意图的简单修复器。所有网络只返回中心XY与log(w/h)，没有关系、接触、意图或拓扑预测头。注意力内部使用相对坐标和公开连线/MIB/grouping信息，不增加输出关系。

四组正式控制分别为原GNN/MSE、原GNN/几何、新注意力架构/MSE、新注意力架构/几何。原架构80,131参数，新架构497,299参数；架构、宽深、容量和交互一起改变，结论只能归因于这一组合，不能单独归因于注意力或参数量。四组按相同最大墙钟预算训练、相同数据顺序和监测准则早停，实际步数/耗时不同，不是等FLOPs控制。

在观察dev500监测、但尚未运行正式dev100评价时，另行预设 `baseline_local`，移除五项显式规则，仅保留局部中心、形状与近邻相对监督。它是适应性机制消融，不是事前五组比较；只能判断五项显式规则在该监督组合之上的整体作用。旧E1冻结权重仅是历史对照，其预算和停止条件不同。

训练train20,000；train/dev_monitor各500；终末完整train20,000/dev2,500监督评价；固定dev_quality100做求解评价，覆盖100worker。没有读取local_test张量。参考布局用于训练标签和外部评价，推理接口仅接受公开七元组；模型输入过滤target/reference，网络拒绝带这两字段的字典。未用参考线长/bbox选择求解候选。

## 2. 新目标和调度

\[
L=L_c+0.25L_s+0.25L_r+\alpha(t)(0.5L_o+0.5L_t+0.25L_b+0.1L_m+0.1L_a).
\]

中心误差按自身面积平方根归一化后做SmoothL1；形状为自由log宽高比MSE；参考近邻的相对中心差按局部尺度归一化。显式规则分别约束最小轴重叠穿透、参考同组公共边、当前预测bbox上的指定边/角、兼容面积MIB对的log宽高、参考bbox面积尺度。参考接触只作几何训练信号，推理不输入接触标签。详见[逐项定义](../../../design.md#83-已实现几何损失的逐项解释)和[实现](../../../losses.py)。

后五项在前2个epoch线性加入；监测始终全权重。没有直接HPWL训练项，bbox项是双向尺度匹配而非直接最小化面积。几何Loss不保证硬合法，也不等价于5%位移/形变内可修。MIB只用面积log差<0.019作成对软偏好，不能当作整组可行性判定。

AdamW初始lr=1e-3、wd=1e-4、clip=1，batch32、CPU2线程、RTX5080Laptop16GB、float32、确定性运算。ReduceLROnPlateau每次减半、patience3/cooldown1、相对阈值0.002、下限1e-5；早停要求至少5epoch、至少2次降学习率、连续12次监测无0.2%显著改善。所有新组均以同一dev500完整几何组合选best/调度，不按dev100反复选择checkpoint。

<!-- TRAINING_TABLE -->

五组均触发预设早停并经历7次降学习率，不能据此宣称全局收敛。含五例预检查、数据准备、训练/监测/保存，两个运行累计 **4596.406秒=76.607分钟**，低于7200秒；提前停止后没有填满预算。best/latest及优化器、调度器、随机数和数据游标均保存。终末评价另行计时。

## 3. 完整训练/验证拟合

<!-- SUPERVISED_TABLE -->

这些分项为每布局平均后对布局等权，原MSE与几何组合的量纲/加权不同，不能直接比较它们的数值大小。attention_mse的train/dev MSE为0.04482/0.05615，已有可见泛化差距；attention_geometry为0.06403/0.06381，但不能只由这一点断言容量够或不够。所有新组best均已看过全部20,000训练例。

<!-- SUPERVISED_EFFECTS -->

数据审计：train/dev分别21–120块；自由形状目标log(w/h)均位于[-1.0986123,1.0986123]，没有目标超出当前解码截断[-5,5]。因此本轮标签范围不受该截断限制；这不排除其他表示/解码问题。

## 4. 固定dev100原始布局与微调必要条件

<!-- RAW_TABLE -->

所有原始提议的Q与非法布局线长/bbox只能作诊断，不能当可交付质量。块对重叠面积之和可能重复覆盖同一区域，不能理解为几何并集面积。中心误差以各块sqrt(effective_area)归一化；严格参考接触召回只统计正长度公共边，近接触召回不代表官方分组合法。

<!-- MICRO_TABLE -->

固定形状下界来自最难分离块对。联合下界更乐观：自由宽高均可独立缩至exp(-eta)，暂时放弃面积保持，fixed/preplaced规格仍锁定；允许的集合包含真正的有限形变方案。因此超过5%的下界能排除对应联合微调，低于5%不能证明可修。没有运行新的有限修复器，也没有实际有限修复成功率；这里绝不能将“未被下界排除”计为成功。

<!-- RAW_EFFECTS -->

## 5. 冻结后处理后的质量与代价

本轮复制并冻结E1的12轮碰撞修正、顺序、单候选pack、row_control兜底。packing/feasible逐字节相同，修正函数AST核对相同。它是控制组，未优化为新的意图感知修复器。

<!-- FINAL_TABLE -->

Q为官方公式在中性时间因子下的块数加权质量，越低越好，失败保留。真实推理耗时另报，不能称正式比赛分数。所有模型预加载，逐例CUDA同步；每个模型均计公开特征准备/传输、前向、解码、修正、pack与兜底，参考读取和评分在外。模型载入时间单列。相同GPU/CPU2线程、15秒/例上限。

<!-- PAIRED_TABLE -->

<!-- SIZE_TABLE -->

<!-- FINAL_EFFECTS -->

## 6. 表示限制、研究判断与下一步

<!-- REPRESENTATION -->

本轮结论仍受单种子、dev监测选best、适应性消融和冻结旧修复器限制；不能宣称测试泛化或模型与新修复的联合成功。几何损失包含监督参考的接触/外框偏好，官方则允许其他优质布局，二者目标不完全相同。相同最大预算与早停不等于优化到相同误差或达到各模型最优。

<!-- NEXT_STEPS -->

## 7. 证据、复现与文件整理

[完整结果](results.json) · [逐例唯一坐标记录](cases.jsonl) · [全量监督分项](supervised.json) · [数据范围审计](dataset_characteristics.json) · [评价配置](config.json) · [图集](index.html) · [训练曲线](training_curves.png)。本文件为唯一完整分析；索引和实验README仅保留结论入口。

四组训练的原始[配置](../config.json)、[运行总结](../training_results.json)、源码快照和各best/latest保留；新增消融位于 `../../20261017_local_supervision_ablation_seed0/`，不改原四组记录。原配置command记录sys.argv，复现使用以下模块命令而非直接将脚本文件路径执行。重新训练、停止和恢复规格见[设计第8节](../../../design.md#8-本轮实际四组实验协议)，本轮不再启动训练。

```powershell
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
python -B -m experiments.intent_aware_repair.evaluate --run-dir experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0 --extra-run experiments/intent_aware_repair/runs/20261017_local_supervision_ablation_seed0
python -B -m experiments.intent_aware_repair.plot --run-dir experiments/intent_aware_repair/runs/20261017_geometry_only_factorial_seed0
```

评价首次拒绝覆盖现有目录；中断后在原命令加 `--resume`，校验checkpoint哈希后复用已完整监督结果及逐例记录，不训练、不重置预算。图集仅从保存结果绘制，不重读数据。所有Python源文件语法解析通过，评价源码快照哈希与当前实现一致。

冻结文件核对：split_v1 SHA256 `a1f23edae03194c1155d260415482d892e4cb5ad66170abc93af04aafc8cefbd`；workset_v2 `b520f75d67d12a00aa4f52953e1b9f3cf4ad1c55f1af67206b4c3bc75e9a5bb3`；旧best `dca8ddc179308a30623edc8b8f6536a1c254b1452d3b63216f0a3846f7d5fe76`。测试未重新访问，冻结文件与旧权重未改。
