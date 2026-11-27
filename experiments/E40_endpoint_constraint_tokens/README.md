# E40：外部端点与约束对象的分层交互

2026-10-23，在E38后台容量训练期间，按已批准计划准备第二类结构候选。现已锁定96×12，CUDA及实际训练入口检查通过，首轮已完成；完整开发代理无实际收益，停止扩展当前具体实现。E38的模型、Loss、训练源码及既有状态不改动。

## 机制与可证伪假设

当前17维特征把一个模块的外部连接压成总权重与加权中心。相同中心可对应不同端点分布，Manhattan代价及最优位置可能不同；扩大网络无法恢复输入中已丢失的信息。本实验保留每个真实外部端点，按连接权重聚合由当前几何条件化的非线性消息；训练目标仍为E17原geometry Loss。

历史E28曾保留完整端点，但使用冻结草稿上的有界修正器及不同训练目标，未完成端点/迭代的完整消融。本轮不恢复该修正器或权重，检验当前参考监督下从头训练的分层网络；不能因旧负结果而预判本轮，也不能因本轮准备完成宣称端点保留已有模型收益。

同时将每个多成员group/MIB视为独立约束对象：它从成员模块的学习式注意力读取信息，保存跨层状态，再向成员返回消息。对象只由公开约束分组，端点只用公开坐标/连接；不将参考接触、标签或评分输入网络。不额外输出关系头，不调用后处理参与训练。

模型以原普通主干为基础，新增端点与约束分支采用零初始化的有界残差门；同种子下主干参数与初始输出保留。门开始更新后，新增分支开始学习。该设计首先检验信息交互机制，不强制合法化或绑定形状。

| 配置 | 意义 |
|---|---|
| geometry_control | 同深度、同主干宽度的原普通网络 |
| full_tokens | 保留各外部端点＋跨层group/MIB状态，主候选 |
| pooled_tokens | 使用相同编码器/参数，仅将端点换为原加权中心；约束分支保持，隔离端点信息贡献 |
| parameter_control | 普通网络适当加宽，近似匹配full_tokens参数，排除仅因参数增加的解释 |

先比较普通主干与full_tokens。只有出现值得继续的信号，才补充相同参数的pooled和普通加宽控制，并根据结果追加端点/约束单独消融。若信息增加只带来成本、没有验证和draft收益，停止该具体候选；一个实现失败不能概括所有分层结构。

## 实现与评价边界

[model.py](model.py)保存分层网络，[public.py](public.py)只从公开solver tuple构造端点/成员矩阵；[data.py](data.py)与[inference.py](inference.py)分别接入训练和在线构造。新增张量按batch即时构建，不重复保存原始数据或缓存。原[features.py](features.py)、[batches.py](batches.py)、[loss.py](loss.py)、[losses.py](losses.py)保持原语义，公开输入筛选在单独public模块扩展，不改变冻结缓存的身份。独立普通参考在[reference_model.py](reference_model.py)，来源见[provenance.json](provenance.json)。

v5 train16k/dev2k、固定500监测、原Loss、选模代理、batch32、四档LR与最低档两轮规则保持。每组4小时/200轮保护，best/latest与完整恢复；正式运行后源代码冻结。目标为尽可能优秀的模型，本轮无固定Q或耗时通过线。冻结唯一best后核对完整dev2000原始指标，再用同版E25后处理一次终末500，计入额外公开特征、CPU推理与全部修复时间。

E25参照保持冻结。测试封存，不利用既有测试draft100例作为新验证。GPU正式检查与训练等待容量结果；准备期只做三例同成员CPU检查及两个端点分布反例，最多五例。所有新算法独立维护，不从E38/E39导入私有实现。

## 准备结果与入口

[五例CPU检查](runs/20261023_structure_check5_cpu_r2/results.json)完成：两个train、一个dev及两个构造反例。普通96×4为497,299参数，full/pooled均876,315参数，普通128×4参数控制为879,763，二者规模差约0.4%。中性门初始化输出与普通主干完全相同，更新门后分支梯度非零；在线/训练额外公开张量一致。

相同原始17维特征、相同加权中心的两个端点分布，full分支可区分、pooled分支输出完全相同。这只是表示能力反例，不是实际布局收益。三例中有效端点数114/237/398，完整模块×端点展开会浪费显存；实现只编码真实连接边，再按权重聚合，与同定义稠密消息最大差约1.9e-6，不改变信息或参数。原始稠密检查与稀疏检查使用同五例，未扩大样本；后者约8.6秒。

2026-10-23审阅发现初版临时检查的两步参数更新曾包含一例dev；临时模型未保存或复用，正式E40训练尚未开始。已改为仅两例train参与梯度/更新，dev单独eval/no_grad，并通过[角色隔离复查](runs/20261023_train_dev_isolation_cpu/results.json)；旧结果按发生时状态保留。CUDA检查改为同两例train重复组成batch32，并开启正式训练的确定性算法，以实际检查稀疏index_add的CUDA支持和显存；稠密数值定义仍只在CPU上的同三例核对，避免检查路径重新造成大规模稠密展开。GPU检查等待容量结果。

[launch.ps1](launch.ps1)与[pipeline.py](pipeline.py)默认`pair`仅执行geometry_control/full_tokens；pooled及参数控制须在出现信号后显式执行，不默认运行。[analyze.py](analyze.py)仅读取已保存结果。容量确定后再用明确Width/Depth、新tag运行；当前没有正式E40后台任务，不占用E38 GPU。

```powershell
# 仅展示入口格式，96×4尚未作为正式主干选定。
& .\experiments\E40_endpoint_constraint_tokens\launch.ps1 -Arm pair -Width 96 -Depth 4 -Tag YYYYMMDD_tokens_w96d4 -Resume
```

日志为`runs/<tag>_pipeline_seed0/console.log`，每组输出为`runs/<tag>_<arm>_seed0/`。每组4小时/200轮，STOP、安全恢复、完整dev及唯一best终末评价遵守同一协议；所有状态与E38/E39隔离。

E39首个运行发现共用训练入口只传width/layers、遗漏结构参数；E40尚未正式启动，也同步修为v2完整model_config构造并核对实际参数量。[实际入口CUDA检查](runs/20261023_training_entry_cuda/results.json)通过：真实full_tokens为2,605,131参数、checkpoint严格加载、新分支门实际更新、两步中断恢复参数差为0。[结构CUDA检查](runs/20261023_w96d12_check_cuda/results.json)也通过，确定性稀疏聚合及同三例batch32有效；峰值约2.15GB只代表检查样本。v2正式运行使用新tag，不复用E39无效运行或临时检查权重。

2026-10-23容量完整dev已齐，主干锁定96×12。按[跨阶段顺序入口](../E38_model_architecture/runs/20261023_structure_followup_r2/runner.py)，相同三例CUDA batch32检查通过后，在E39正式训练及终末计时全部结束后，才运行tag `20261023_tokens_w96d12_r2`的full_tokens、seed0。普通控制只读复用E38 deep96x12，候选仍从头初始化；同Loss、监测、LR/早停与原普通模型的[身份核对](../E38_model_architecture/runs/20261023_structure_followup_r2/control_reuse.json)已完成。E40仅扩展公开端点/约束输入，不改原缓存标签。pooled、近似参数匹配控制及进一步消融在候选有信号后执行。实际启动以跟进日志为准，本段取代准备期“主干待定”的状态。

full_tokens首轮已完成46轮、best43，完整dev代理0.491113，较普通96×12仅低0.014%；平均重叠面积占比增加3.03%、边界Loss增加18.33%。终末500全硬合法，加权Q1.039813、均时0.324206秒，推理与修复负担未受益。[结论与配对证据](runs/20261023_tokens_w96d12_r2_full_tokens_seed0/analysis.md)支持停止扩展该具体实现，不再追加pooled/参数控制或种子；不将未拆分的结果归因到单个分支。研究转入E39机制消融及核心深度的匹配种子确认。
