# E97：GPU混合神经网络与几何求解

第二研究窗口已按用户“不再训练，直接审计收尾”关闭：原起点北京时间2026-11-02 13:04:02保持，截止曾延长至17:45；7个正式候选均完成审计，候选8只有预检、未训练。保留C5完整dev500 Q1.012685893，优于当前C3控制1.013366283；500例全部硬合法、零回退，完整均时.204464秒/P95 .521080秒，126例超过.25秒，质量与尾部时间目标仍未达。C6自身Loss选epoch0严格等C5而复用；C7新教师精确WH候选Q1.013677462未升级。单线程串行、测试封存，自动唤醒已删除，无活动训练/评价。[第二窗口完整报告](runs/round2_final_report.md) · [完成审计](runs/round2_completion_audit.json) · [账本](round2_study.json) · [状态](round2_state.json)。

用户2026-11-02新授权的旧E85完整后处理单线程补测已完成并审计：10臂各500例，全合法、零行回退。相同单线程完整配置下，E85 Q1.015922，C3 Q1.013332/完整均时.26247秒/P95 .58088秒，C6 Q1.013265/.27734/.59692，均未同时达成质量和速度目标。原“新模型普遍远差于E85”的判断须修正。[结论修订](runs/fullrepair_singlethread_findings.md) · [完整数值与同100图](runs/20261102_fullrepair_singlethread_dev500/report.md) · [完成审计](runs/20261102_fullrepair_singlethread_dev500/completion_audit.json)。没有新训练或自动唤醒，原八候选研究的关闭记录保持不变。

此前三步流程的8个正式候选及其唯一own-Loss best评价已全部完成，因候选上限收尾。**没有达到固定dev500加权Q<1.01的质量目标**；硬合法与本次完整逐例速度目标已实现。自动唤醒已关闭，无后台训练或评价。测试封存，未升级达标模型。

[完整研究报告](runs/final_report.md) · [12小时/8候选账本](study.json) · [状态](state.json) · [固定计划](plan.json) · [全部保存结果审计](runs/final_saved_audit.json)。原窗口UTC2026-11-01T14:58:12至2026-11-02T02:58:12，恢复/唤醒未重置预算；第8候选完成后不启动新候选。

同3步预算中，C2完整Q1.169365、12回退、500/500合法、完整均时.12626秒/P95 .20378秒，是本轮最低Q开发候选，但raw重叠恶化，不能认定为达标。最后C8完整Q1.233626、11回退、500/500合法、均时.14017秒/P95 .21869秒；raw覆盖<=10%模块82.0353%，最坏模块仍100%。C8的93.79%超额Q来自11个低质量合法回退，所有回退保留在完整成绩中。

原五train公开输入到draft GPU图约10ms仅是短检。C8完整dev500的公开输入到draft均时40.08ms/P95 44.84ms、repair均时98.48ms/P95 175.33ms；完整请求含全部公开输入构建/传输、NN与内部几何、CPU修复/回退、返回及必要硬检查，冷启动单列，不以批量吞吐冒充单例。端口对NN保持每模块总权重/加权中心完全压缩，精确Loss/线长仍保留完整连接。

研究证据：[初始draft延迟](runs/latency_report.md) · [五例机制](runs/slim_mechanism_report.md) · [同dev500必要对照](runs/slim3_dev500_report.md) · [C1](runs/candidate1_result_report.md) · [C2](runs/candidate2_result_report.md) · [C3](runs/candidate3_result_report.md) · [C4](runs/candidate4_result_report.md) · [C5](runs/candidate5_result_report.md) · [C6](runs/candidate6_result_report.md) · [C7](runs/candidate7_result_report.md) · [C8](runs/candidate8_result_report.md) · [C8固定同100例四阶段图](runs/20261102_candidate8_public_task_seed0/evaluation/gallery/index.html)。

公共teacher16500已全部生成并只读复用，无自由reference/评分分母输入。oracle仅为不可部署诊断，不能作为最终成绩或质量下限。全部固定Loss用于反传/监测/best/LR/早停，未按Q另选checkpoint；best/latest与Adam/RNG/排列游标/调度/预算保留。8个候选绑定源码及100张C8图SHA核验通过，C8三阶段OS锁释放。历史E85完整修复Q1.014672使用不同求解/计时条件，保留参照而不混为本轮部署成绩。

本目录nn85/nn96及几何源码独立复制，来源见[provenance](provenance.json)；只共享datasets/scoring/visualization，无跨实验私有算法导入。所有旧评价/teacher/诊断不重跑，原split/冻结索引/唯一权重保护。下一研究方向建议见最终报告；本轮未追加训练、求解器强度或其他自动化。
