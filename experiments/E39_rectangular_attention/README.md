# E39：矩形感知注意力

后续用户指定只做一次AB联合正则化，已独立实施[E42矩形96×12联合AB](../E42_joint_regularization/README.md)入口并交付命令，尚未训练；本目录的未正则化12层继续作为只读参照，不重训或重评。

最新状态：no_gap已结束51轮、best48、固定monitor500代理0.488556，完整开发与终末评价未运行；旧后台队列及定时跟进保持关闭。用户要求继续扩深但当前只交付命令，因此已独立准备[E41矩形96×16](../E41_rectangular_depth/README.md)，未启动训练，避免修改本实验冻结源码。MIB平均面积处理继续保留，MIB Loss加权调整留作后续。

2026-10-23已完成用户要求的[失败模式与MIB监督一致性审计](runs/20261023_failure_mode_audit/analysis.md)：既有dev500轨迹、固定train/dev各500监督检查及原公开面积核对。group占比与修复负担的条件关联弱；碰撞严重度和预放置障碍冲突更有解释力。确认平均面积输入与原中心/长宽比/接触监督存在冲突；模型仍已明显学习同形趋势。附逐例表、统计图及五个反例三相图。本轮未训练、未重跑求解器、未改变冻结数据/Loss，也未恢复定时跟进。新增深度/监督修订均未执行。

2026-10-22，在用户授权E38后台长训练、等待期间实施下一实验后建立。本实验承接E38的矩形感知方向，独立复制源码，保持E38训练源码冻结。2026-10-23已根据四组结果锁定96×12，完成CUDA及实际训练入口检查，rectangle的v2正确配置首轮已完成，no_gap机制消融已在E40结束后后台执行；原始准备期记录保留如下。

假设：普通attention主要通过中心位置与聚合关系交互，直接输入当前矩形宽高、间隙、穿透及固定/预放置标记，可让相同规模的主干更容易协调局部几何。保留E17的所有Loss、权重和监督；增加的几何信息全部来自公开面积、约束、规格及网络当前预测，不增加标签、教师或后处理反馈。

历史E7的矩形感知网络曾退化，不能忽略这个反例。本次按当前批准计划检验一个范围更窄的新实现：保留普通主干与原状态更新，只扩展pair-bias输入，并匹配初始化；同时采用当前v5与固定Loss对照。它不是恢复E7的有界残差网络或旧权重，结果仍可能否定本候选。

## 对照

| 配置 | 交互 | 用途 |
|---|---|---|
| geometry_control | 原普通geometry attention | 与E38所选主干保持相同深度、宽度的同期对照 |
| rectangle | 原7维pair特征＋归一化宽高/间隙/穿透/锚点标记，共19维 | 主候选；只增加小pair-bias编码器输入，不堆主干参数 |
| rectangle_no_gap | 与rectangle完全相同参数，间隙/穿透四通道置零 | 有信号后使用的消融，不自动训练；保留尺寸、锚点及同一解码中心 |

增加pair输入不改变模块输出与解码。固定/预放置矩形按公开规格解码，面积及尺度来自公开输入；标签只进入Loss。每层从当前state解码尺寸与中心，使几何信息随推理更新。[model.py](model.py)与[arms.py](arms.py)保存机制和配置；[train.py](train.py)、[data.py](data.py)、[loss.py](loss.py)为本实验独立入口，来源见[provenance.json](provenance.json)。

结构对照保持同种子下普通主干、原7通道及pair编码器其余参数的初始化一致；新增12通道权重从零开始，不消耗后续层的随机初始化序列。原7通道仍采用原始中心定义，额外间隙/穿透采用解码矩形；因此初始化输出应与普通主干在数值精度内一致，后续差异来自新增通道的学习。去间隙消融保留同形状参数和同初值。

主干通过显式环境`E39_WIDTH`/`E39_DEPTH`选择，写入每个checkpoint的model_config和恢复身份，禁止恢复时静默换容量。E38完成后，先根据固定验证代理、完整开发核对和draft分项决定主干，再锁定E39种子、容量、预算及运行命令。若深度和宽度均有值得继续探索的信号，可先补充容量组合，不能将其收益混作矩形结构收益。

## 评价与边界

使用v5 train16k/dev2k与原500监测，固定geometry反向Loss及geometry+occupancy验证代理。所有训练从头初始化；batch32、四档学习率与最低档两轮规则、4小时/200轮保护、best/latest和完整恢复状态沿用E38。正式评价采用同版E25修复并完整计时；先冻best再评价，不按Q反选checkpoint。模型目标为尽可能优秀，不设置固定Q/耗时通过线；报告Loss、冲突、接触/边界/MIB、泛化差距、CPU成本及实际修复负担。

若验证代理无收益，或冲突下降伴随接触/边界明显退化，先检查机制再决定是否停止；一次具体结构失败不等于所有矩形感知方法无效。容量选择后才做最多三例功能检查，GPU训练和CPU正式计时避免资源干扰。E25保持冻结、测试继续封存，既有测试draft100例暴露事实不改变。

## 实现状态与入口

三例CPU检查使用与E38相同的两例train与一例dev，不新增测试暴露。96×4普通主干497,299参数，rectangle及no_gap均498,451参数，新增仅1,152参数；同种子共有参数差为0，初始化输出最大差约4.2e-7。所有分支梯度有限，新通道梯度非零，no_gap的间隙/穿透通道梯度为0。该检查约5.85秒，不代表正式训练结果或GPU速度；正式GPU检查等待容量选择后进行，继续使用相同三例。

2026-10-23审阅发现初版临时检查曾将三例一起做梯度验证，其中包含一例dev；没有保存、复用检查权重，也未进行正式E39训练。现已修正为仅两例train参与梯度、dev单独eval/no_grad，并通过[角色隔离复查](runs/20261023_train_dev_isolation_cpu/results.json)。旧记录保留原样，新检查显式记录gradient_members与validation_members。未来`--device cuda`使用同两例train重复组成batch32，并开启与正式训练相同的确定性算法，记录显存峰值；GPU检查仍等待容量选择后执行。

[launch.ps1](launch.ps1)与[pipeline.py](pipeline.py)默认只运行geometry_control/rectangle两个配置；no_gap须显式选择，不自动增加消融。完整源码与恢复行为独立于E38，监测与终末评价逻辑沿用固定协议。[analyze.py](analyze.py)汇总已保存结果，普通主干参考源码独立保存在[reference_model.py](reference_model.py)，用于接入E25冻结权重时的语义核对。

容量确定后，使用新tag并明确Width/Depth运行，例如下列仅展示入口格式，不代表已经选定96×4：

```powershell
& .\experiments\E39_rectangular_attention\launch.ps1 -Arm pair -Width 96 -Depth 4 -Tag YYYYMMDD_rectangle_w96d4 -Resume
```

输出为`runs/<tag>_<arm>_seed0/`与`runs/<tag>_pipeline_seed0/console.log`，每组4小时/200轮。STOP、安全恢复、完整dev2000及终末500的边界与E38相同，但状态文件、锁和进程各自独立。

首个正式入口尝试暴露了构造缺陷：v1在实际训练时只传width/layers，遗漏rectangle配置，尽管身份记录与独立结构检查使用了完整参数。已安全停止于725步，约169.12秒，未终末评价；[无效运行说明](runs/20261023_rectangle_w96d12_rectangle_seed0/invalid_architecture.json)及原checkpoint/源码快照保留，该权重不作为候选、不续用。v2实际构造改用完整model_config，并立即核对实际参数量。通过[实际训练入口CUDA检查](runs/20261023_training_entry_cuda/results.json)：真实1,471,539参数、断点严格按矩形模型加载、新通道实际更新、中断恢复两步参数差为0；使用原两train/一dev，无新增测试。此前[结构CUDA检查](runs/20261023_w96d12_check_cuda/results.json)仍有效，但它单独不足以证明训练入口正确。修复后从头新训的tag为`20261023_rectangle_w96d12_r2`，无效运行开销另计保留，没有静默重置其预算。

2026-10-23四组容量的完整dev已齐，后续主干锁定96×12；不按终末Q选择。使用[跨阶段顺序入口](../E38_model_architecture/runs/20261023_structure_followup_r2/runner.py)，先检查CUDA batch32，再以tag `20261023_rectangle_w96d12_r2`仅训练rectangle、seed0，普通控制只读复用E38 deep96x12，权重仍从头初始化。训练主体、Loss、特征、缓存及诊断的[身份核对](../E38_model_architecture/runs/20261023_structure_followup_r2/control_reuse.json)已完成；数据与LR/早停一致，原普通模型逐字节相同。待核心流水线完全结束才启动，避免与终末计时并行。rectangle_no_gap与多种子留待候选信号；本段取代上文“主干待定”的准备期状态，实际GPU/训练启动以跟进日志为准。

正确v2首轮已完成50轮、best49，完整dev代理0.468025，相对同主干控制低4.71%，重叠面积占比低13.38%，但边界/MIB Loss分别增加24.44%/18.10%。终末500全硬合法、加权Q1.026867、均时0.280588秒，算术Q/尾部未一致改善，暂不宣布胜出。[阶段分析](runs/20261023_rectangle_w96d12_r2_rectangle_seed0/analysis.md)与[no_gap机制消融计划](runs/20261023_rectangle_w96d12_r2_rectangle_seed0/gap_ablation_plan.json)已保存；E40正在训练，消融等待其训练与终末结束后单GPU顺序运行。训练源码继续冻结。

E40首轮结束后，已通过[后续顺序流程](../E38_model_architecture/runs/20261023_mechanism_confirmation/plan.md)启动no_gap，之后补做核心普通4层/12层的seed1/2匹配确认；无GPU并行、Loss与数据不变。消融输出为`runs/20261023_rectangle_no_gap_w96d12_rectangle_no_gap_seed0/`，best选择及唯一终末评价规则相同。

用户随后要求仅继续已运行训练、不再启动新训练并停止定时任务。当前no_gap继续，后续种子及自动接续评价已取消，定时跟进已PAUSED；[成果汇报](../E38_model_architecture/runs/20261023_mechanism_confirmation/research_summary.md)保存本次结论与边界。上述旧计划不再授权执行。

按用户要求导出rectangle 96×12的[100张三相图文件夹](runs/20261023_rectangle_triptych100/images/)与[浏览索引](runs/20261023_rectangle_triptych100/index.html)：固定取既有开发500例的前100例，依次为原始参考监督、E39草稿、已保存后处理结果，不按Q挑选。原始参考宽高可能不满足v5派生面积约束，图中按实际情况显示；草稿只补算CPU推理并核对既有raw指标，最终坐标直接复用，不重跑求解器、不训练、不读取测试。[绘图入口](runs/20261023_rectangle_triptych100/render.py)、[配置](runs/20261023_rectangle_triptych100/config.json)及[完成验收](runs/20261023_rectangle_triptych100/summary.json)归该运行；共2进程×2线程，绘图统一调用visualization。原始pilot配置保留，后续只修正PNG高度验收容许Matplotlib的1988/1989像素取整差异，首张图未重复推理或渲染。
