# 交付与研究提案

- [首稿生成器 v2：四轮隐藏推理，最后一次读出](first_draft_generator_v2_20261108.md)：四块 Attention＋SwiGLU、全局上下文输出，共209891参数，比原四块+0.915%。无中间几何预测/反馈、独立模块对消息或持续边状态；已落实为[E116](../experiments/E116_first_draft_comparison/RUNNING.md)，获用户授权后台串行训练，仅比较最终Loss。

- [首稿生成器设计与参数对比](first_draft_generator_proposal_20261108.md)：206,467 参数模块对候选在核对已完成 E65 后撤回推荐，不推进实施/训练；原设计与历史差异核对保留。首稿独立训练、只比较共同 Loss 的研究目标保留。

- [A基线下关键子网络深度](critical_network_depth_proposal_20261107.md)：用户已选A为重要研究基线；设计几何关系网络/预测头加深与反馈节奏对照，尚未实施。

- [逐层学习方案](layer_learning_proposal_20261107.md)：针对前层学习责任，比较逐层直接监督、独立head与分段任务；推荐先做统一新协议基线的最小对照。仅提案，未实施或启动。

- [final_delivery](final_delivery/)：历史交付产物，具体模型身份和当前入口以[E101](../experiments/E101_final_c5_solver/README.md)为准。
- [结构方案原稿](model_architecture_proposals_20261105.md)：方案一后来实现为E105并完成；方案二/三未实施。原稿中的“尚未实施”是写作时状态。
- [公开能量/无reference优化提案](reference_free_optimization_proposal_20261105.md)：尚未实施或启动，已撤回父稿固定2%总位移限制；不构成解除E107 STOP的授权。

最新接手入口在[导读](../docs/START_HERE.md)与[历史综合](../docs/research_history.md)。提案、已完成证据和当前执行权限分别阅读，不将本目录当后台任务清单。
