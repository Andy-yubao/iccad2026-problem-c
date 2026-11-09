# Factbase：E42方法、结果与官方参照

日期：2026-10-24；修订v2。所有官方比较数据来自仓库references/official/results。主报告不用内部旧版本作比较。

## F01 最终1000例结果

精确字段见factbase.json对应条目和下列证据。

证据：[evidence/E42_final_summary.json](evidence/E42_final_summary.json)

解释边界：本地MIB平均面积任务；不是官方隐藏集成绩。

## F02 官方比较对象

Beta为100例隐藏测试；Alpha为100例公开验证集的接口演练。Beta归档包含raw_score和total_score，Alpha仅包含total_score。

证据：[evidence/C_beta_evaluation_report_hidden_final.txt](evidence/C_beta_evaluation_report_hidden_final.txt) · [evidence/C_beta_leaderboard_20260823.csv](evidence/C_beta_leaderboard_20260823.csv) · [evidence/C_Alpha_Top5_20260720.csv](evidence/C_Alpha_Top5_20260720.csv)

解释边界：用Beta作主要历史参照，不将Alpha带时间总分与本地Q直接相减。

## F03 官方质量与速度

精确字段见factbase.json对应条目和下列证据。

证据：[materials/official_beta_reference.json](materials/official_beta_reference.json) · [evidence/C_beta_leaderboard_20260823.csv](evidence/C_beta_leaderboard_20260823.csv)

解释边界：total_runtime/100仅为归档平均时间描述。数据和硬件不同，不可据本地数值推断比赛名次或同预算领先。

## F04 raw_score解释

本材料按无时间质量分理解raw_score；官方综合第3与第10名的total_score/raw_score均约0.7，与运行时间奖励下限吻合。

证据：[materials/official_beta_reference.json](materials/official_beta_reference.json) · [evidence/C_beta_evaluation_report_hidden_final.txt](evidence/C_beta_evaluation_report_hidden_final.txt)

解释边界：归档没有对raw_score列单独下定义；这是有数值支持的工作解释，不冒充额外官方声明。

## F05 官方硬件

Beta报告记载赛方为每次提交提供专用48核CPU和NVIDIA A100 80GB。

证据：[evidence/C_beta_evaluation_report_hidden_final.txt](evidence/C_beta_evaluation_report_hidden_final.txt)

解释边界：不知道各队实际用了多少资源；本地最终评价使用4个单线程CPU进程。

## F06 数据变换

MIB组内取算术平均面积，保留组总面积和固定/预放置规格；仍不兼容者排除。原参考几何和评分分母保留。

证据：[evidence/protocol.md](evidence/protocol.md) · [evidence/mean_area_and_registry.py](evidence/mean_area_and_registry.py)

解释边界：改变了面积输入及样本分布。参考列是原任务答案，不保证对变换后面积仍然硬合法。

## F07 训练与测试规模

v5为训练16000、开发2000、本地测试2000例，训练/开发各固定500例监测；原始分片80/10/10不变。最终1000从余下1900中按实际模块数分层选取，排除曾看过草稿的100例。

证据：[evidence/protocol.md](evidence/protocol.md) · [evidence/E42_final_config.json](evidence/E42_final_config.json)

解释边界：开发集用于研究；本次模型评价是在同面板前序结果已知后追加，不能称新的独立盲测；剩余900例封存。

## F08 输入长什么样

每块17个输入数值，包括面积、公开尺寸/位置、固定和预放置标记、边界要求、同组/同MIB数量、连接总强度及外部引脚加权平均位置；另外保留模块两两连接、同组和同MIB关系矩阵。

证据：[evidence/E42_features.py](evidence/E42_features.py) · [evidence/E42_inference.py](evidence/E42_inference.py)

解释边界：外部引脚在输入特征中被聚合，不声称网络逐一读取了全部引脚节点。自由块参考位置不进入输入。

## F09 网络信息流

每块17个数映射为96个内部数值，先预测中心x/y及对数长宽比；经过12个独立参数的更新单元，每个单元按内容和当前矩形关系加权汇总其他模块，并另汇总连接/分组/MIB邻居，随后更新位置形状猜测。

证据：[evidence/E42_model.py](evidence/E42_model.py)

解释边界：注意力权重通过训练学习，不是保证强连接必然权重大；12层是一次前向内部的12轮，不是对新样例训练12次。

## F10 矩形信息与输出

两两关系由原7维增至19维，新增当前宽高、边缘空隙、穿透及固定/预放置标记。自由块宽高由面积与长宽比恢复；固定/预放置按公开规格覆盖。模型共1471539参数。

证据：[evidence/E42_model.py](evidence/E42_model.py)

解释边界：参数量与隐藏宽度用于说明规模，本身不能解释方法。草稿仍可能重叠。

## F11 训练事实

精确字段见factbase.json对应条目和下列证据。

证据：[evidence/E42_training_results.json](evidence/E42_training_results.json) · [evidence/E42_training_config.json](evidence/E42_training_config.json)

解释边界：54轮、27000步、best51、正常早停，约112.72分钟；训练记录postprocessing_calls=0不代表后续评价未执行。

## F12 训练目标和模型选择

训练使用位置、形状与几何约束误差；选模综合geometry和occupancy（冲突、拥挤、空洞、松散程度），训练与监测不运行求解器。Dropout0.05、AdamW decay0.0003，单seed0、batch32。

证据：[evidence/E42_losses.py](evidence/E42_losses.py) · [evidence/E42_train.py](evidence/E42_train.py) · [evidence/E42_training_config.json](evidence/E42_training_config.json)

解释边界：occupancy参与选模，不在本次E42反向目标中；联合正则化不能拆开归因两项的独立效应。

## F13 泛化证据

相对未正则化的同结构12层，完整开发代理0.468025→0.453814，降低约3.04%；训练—开发差距缩小约34.39%。

证据：[evidence/E42_completed_analysis.json](evidence/E42_completed_analysis.json) · [evidence/E42_full_dev.json](evidence/E42_full_dev.json)

解释边界：代理不是比赛Q；仅联合配置、单种子证据，不能用它代替最终1000例评价。

## F14 几何优化器

根据当前草稿选择左右/上下分离关系，数值优化位置与形状；必要时固定宽高调整坐标、另造可行起点，再保留硬合法候选并做数值修正。

证据：[evidence/E42_solver.py](evidence/E42_solver.py)

解释边界：算法只用公开输入目标，不用真实参考Q逐例选布局；限时优化的路径会受运行速度影响。

## F15 网络和优化器各自作用

精确字段见factbase.json对应条目和下列证据。

证据：[materials/derived_metrics.json](materials/derived_metrics.json)

解释边界：开发500例草稿0例硬合法，经后处理500例硬合法；最终100%合法属于整个流程，不归因于网络单独输出。

## F16 速度口径

public graph features + CPU model + decode + full steady optimization/rescue; exclude file/model loading, fixed task transformation, JIT initialization, scoring and evidence IO

证据：[evidence/E42_final_config.json](evidence/E42_final_config.json) · [evidence/E42_environment.json](evidence/E42_environment.json)

解释边界：平均单例0.222522秒；批量墙钟109.14秒含初始化、IO、评分，不能除以1000替代单例延迟。原代码未更改，运行库线程钳制为1。

## F17 Q解释

Q越低越好；硬非法记10；先算每例的线长/面积正向差和软违反惩罚，再按exp(模块数/12)加权。中性时间Q取消了速度奖励。

证据：[evidence/official_problem_statement.md](evidence/official_problem_statement.md) · [evidence/scoring_evaluation.py](evidence/scoring_evaluation.py)

解释边界：Q=1表示此评分下无正几何差和软违反，不是数学全局最优证明；Q1.028不能解释成比最优只差2.8%。

## F18 三相图来源

选择最终Q排序第500例、最终Q最大例、完整耗时最大例；参考来自对应原始样本；草稿仅补生成这3例并缓存；最终布局、最终Q和耗时沿用原1000例记录。

证据：[materials/layout_examples.json](materials/layout_examples.json) · [materials/E42_draft_examples.json](materials/E42_draft_examples.json)

解释边界：草稿是报告阶段重建，不是原计时运行保存的草稿；参考/草稿未重新评分，图中指标n/a；没有重跑最终求解。
