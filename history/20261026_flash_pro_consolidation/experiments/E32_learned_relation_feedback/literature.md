# 本轮文献定位

2026-10-22通过arXiv连接器检索标题/摘要；未下载全文、未复现作者实验。下面只作为机制背景，不能代替本项目对照或支持本地性能结论。E32假设和代码先于本次检索登记。

- [CSF: Fixed-outline Floorplanning Based on the Conjugate Subgradient Algorithm Assisted by Q-Learning](https://arxiv.org/abs/2504.03796)，Meng等，v2。摘要提出用Q-learning调节解析求解中的步长，说明ML可参与优化过程控制。其固定轮廓/硬模块实验与本派生任务不同，不能直接移植作者收益。
- [A Hybrid Learning-to-Optimize Framework for Mixed-Integer Quadratic Programming](https://arxiv.org/abs/2511.19383)，Le、Xie、Mangharam，v2。摘要将学习离散变量与连续QP层结合；与E32分离“结构选择”和“连续合法化”的动机相关。E32没有实现可微QP层，也不声称相同方法或理论保证。
- [Learning to Optimize for Mixed-Integer Non-linear Programming with Feasibility Guarantees](https://arxiv.org/abs/2410.11061)，Tang、Khalil、Drgoňa，v10。摘要强调整数修正与约束投影，提示可行性保证需要独立机制。E32仍使用既有硬校验和合法incumbent，不能借用其收敛结论。
- [Design Space Exploration of Reinforcement Learning Configurations for Post-Legalization Placement](https://arxiv.org/abs/2610.20224)，Oikonomou等，v1，2026-10-14。摘要考察18种GNN-PPO配置并讨论局限；支持本轮按动作、状态和奖励分别做对照，而不是默认RL优于现有解法。未从摘要推断任何具体胜率。

本轮不接入在线LLM推理、远程求解或新模型服务，所有实际求解仍在本地资源预算内。复杂RL与可微锥求解暂作为可能的后续机制；先用可审计的离线结构监督检验价值。
