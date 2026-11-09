# E70：reference只监督位置与形状，其余使用无参考任务目标

2026-10-28用户要求制定一套组合Loss：reference只用于位置、形状；其余项必须是在没有参考布局时也能计算的任务目标。当前只形成设计与候选配置，不实现或启动新训练，不修改正在运行的E68。详细公式见[loss_design.md](loss_design.md)，拟用常数见[proposal.json](proposal.json)。

核心变化：删除旧reference-near relative_local和reference指定接触边的contact，取消reference bbox分母；加入实际b2b/p2b线长、穿透与交面积双重避碰、可自主选择拓扑的分组连通、公开边界/MIB、无参考紧凑度与数值参数域保护。只有位置与形状两个项读取自由reference标签。

本版为优先验证的整体组合，取得明显改善后再做回溯消融。系数是预先提出的起点，未经新Loss量级或梯度检查，不称已标定或最优。正式实施后训练、监测、best、学习率下降和早停须使用同一冻结版本，seed0，不做多种子。原G仅作共同诊断，Q辅助，不据Q重选checkpoint；测试封存。

当前状态：**proposed；没有可启动的训练入口，也没有训练结果。** 正在执行的E68保持原Loss和源码身份。
