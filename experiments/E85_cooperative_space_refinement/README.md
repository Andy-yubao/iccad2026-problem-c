# E85：E74上的协同空间修正

**已完成并核查，无活跃训练。** [完成检查](runs/completion/inspection.md)：协同组dev2000交面积1.942%（control 2.120%），困难组3.025%（3.309%），加权Q1.014672对1.014964的配对区间跨零。大部分重叠差在训练前固定算子对照中已存在；无端口模块相对位置/次序改善很小，保留几何候选，不升级冻结基线。两套同100例四阶段图、唯一best与全部运行证据已核验，测试封存。

2026-10-30用户明确授权后台长训练；随后允许探索期对已有权重后训练，有价值后可从头训练。本轮只开展两组同起点后训练，不扩大原18小时总预算，不恢复旧控制器。

冻结E74 best42，两组均从F81 best26的241,547个修正参数继续训练。control沿用独立残差；cooperative在最后中心提议后加入8步真实交面积解析梯度联合更新，每步预设4档二分步长并按每例真实交面积严格下降接受。碰撞压力由当前矩形决定，所有模块同步调整，新碰撞参与下一步传播；不使用reference方向。每步投影回同一E74来源框和最后阶段±0.15个自身模块尺度范围，预放置位置/固定形状/公开面积不变。有限步、有界更新不保证成功换序或合法，完全包含的交面积平坦区也可能不动，需保留失败证据。

两组完全相同参数量、数据、起始可训练权重、固定目标和资源上限。唯一Loss沿用E81单来源`G+4I+5E+.25A`；不增减或重新猜测权重。两组都只用E74，因此不需要旧双来源稳健聚合。v6冻结前32k、原dev2000、train/dev500不变，测试封存。原G和历史公开面积归一化几何诊断分别保留。

方案、停止规则和比较边界见[plan.json](plan.json)；来源与哈希见[provenance.json](provenance.json)。每组2小时/200轮，合计最多4小时；历史34,604.40秒计费保留，总18小时不变。FP32、seed0、batch32、原四档学习率/早停，完整best/latest与恢复身份保存。初始算子直接收益单列，不全部归功训练。

入口：独立环境`python -u -B -m experiments.E85_cooperative_space_refinement.preflight`，通过后`python -u -B -m experiments.E85_cooperative_space_refinement.pipeline`；恢复加`--resume`。创建`runs/STOP`安全停止，移除后仍须显式续跑。控制器依次完成两组训练、各唯一best完整开发/一次flash500、同100四阶段图及对照报告，不重复历史终末，不按Q选模。进度见[state.json](state.json)，[新阶段账本](study.json)，完成后见[runs/comparison.md](runs/comparison.md)。

实现由E81独立复制并在本实验内维护，数据/评分/绘图分别复用datasets/scoring/visualization；没有跨实验导入私有算法。旧实验原件不变。

首版无条件邻对方向约束在5训练例中4例增加交面积，长训练前淘汰；原证据见[runs/preflight_v1_unconditional_neighbor/results.json](runs/preflight_v1_unconditional_neighbor/results.json)。正式方案仅真实碰撞施压，不复用不必要的参考/旧轴分离目标。

第二版仅真实碰撞的穿透力仍有4/5例交面积增加，证据见[runs/preflight_v2_penetration_force/results.json](runs/preflight_v2_penetration_force/results.json)。正式第三版改为现有I的解析梯度和严格下降接受，不以穿透深度冒充交面积下降，不改变Loss或训练参数。

解析交面积梯度与独立autograd在同5训练例最大差2.98e-8；发现按当前碰撞度数定步长会越过较窄的下降区域，第三版全部步被拒绝。正式步长按全部潜在邻居数N归一化，保留固定二分后备，不搜索Loss权重。证据见[plan.json](plan.json)及[runs/preflight_v3_oversized_step/results.json](runs/preflight_v3_oversized_step/results.json)。

启动记录（当时处于control数据加载/训练阶段，现已全部完成）：受管理的持续进程记录见[runs/launch_managed.json](runs/launch_managed.json)，初始完整开发结果见[runs/initial_summary.json](runs/initial_summary.json)，所有实现预检和冻结分析的额外开销见[runs/preparation_accounting.json](runs/preparation_accounting.json)。初始整体交面积2.145%→1.974%、弱端口稀疏157例3.341%→3.088%，仅为固定算子直接效果，不是后训练收益。首次隐藏进程未进入pipeline即退出，已改用持续进程；没有重复训练或重置权重。

2026-10-30用户取消后训练学习率保护期。协同组在11974步安全保存后迁移：完整保留模型、Adam、RNG、样本游标、训练历史和累计耗时；用已有第0～11轮监测初始化调度基准与停滞计数，不伪造过去的降档。后训练从第0轮基准起计数；从头训练仍保留原保护期。已完成control不追溯修改，终末对照不能单独归因为协同算子，报告已加入这一限制。[迁移与核验](runs/lr_policy_change/migration.json) · [源码兼容审阅](runs/lr_policy_change/source_review.diff)。原预注册plan保持原件，本段及迁移记录是明确的新授权修订。

下一阶段用户将高质量draft和清除严重重叠模块列为首要目标，固定500例加权Q<1.01为后续目标。[研究提案与新指标基线](runs/20261030_q101_n5_plan/plan.md)仅复用保存坐标制定方案，尚未启动训练。
