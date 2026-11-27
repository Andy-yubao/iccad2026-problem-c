# E11：短预算下草图与结构的作用诊断

用户于2026-10-18批准先检验“更好的草图能替代多少后处理计算”。本实验不训练模型，E10保持冻结。

## 假设与反证

更准确的中心或分离/接触结构，能使相同E10后处理在较短预算内达到原质量。如果这些诊断输入在短预算下仍明显退化，则继续优化模型的坐标代理不足，应优先检查解码成本或输出表示。完整E10成品回灌只检验成品保留与预算开销，是已知答案的上限校验，不能证明可学习性。

## 预先固定的对照

- 面板：workset_v2.dev_quality中worker 0/24/49/74/99各1例，与E9诊断及E10迁移检查相同。仅5例机制证据，不用于泛化/模型选择，不读取local_test。
- 草图：原冻结模型；替换为reference中心；替换为既有E10成品中心；原草图加E10成品的全对分离关系与组内接触树；完整E10成品。中心替换保持原草图宽高并恢复公开锁定规格，负坐标裁切副作用如实计入。
- 后处理总目标预算：0.10、0.25、0.40、5.00秒；最后一档使用原普通路径1.3秒与总目标5秒救援，前三档把整个后处理限制为对应目标。目标不是硬实时上界，报告所有实际超时。
- 原草图、教师成品复用E8已存记录，哈希及来源写入config；E10继承的模型/后处理与之相同。reference仅用于明确标记的oracle及事后评分，不进入可部署原草图路径。
- 主指标：各预算全部5例平均Q、逐例相对原草图完整预算的Q差、官方与独立硬合法性、最差Q；同时记录HPWL/bbox gap、边界/分组/MIB违反、实际核心/通信时间、迭代与原生求解时间。
- 探索性“接近原质量”：Q≤同例原草图完整预算Q+0.002且硬合法；同时报告严格不退化计数。阈值在运行前固定，不能解释为统计非劣证明。离线首次达到时间只作诊断，不可用参考Q作为在线停止规则。
- 成对顺序用seed0打乱，所有组合均执行，不按分数筛选；每个worker预热1次，预热开销单列，不计入稳态求解时间。原草图推理时间沿用来源记录，所有oracle统一加此时间只作可比预算记账，尚无实际oracle网络。
- 单次累计墙钟保护900秒，最多5例×5草图×4预算=100次求解；不自动扩大样本或启动训练。创建运行目录STOP后，当前请求返回即停止保存；`--resume`显式补齐同一冻结运行的缺失单元，保留原配置、源码快照及invocations中的失败事实，算法改动拒绝续跑；完整重跑用新目录。

## 实现与运行

后处理源码独立复制于[backend](backend/)，复制前哈希在[provenance.json](provenance.json)。仅增加可选的诊断结构输入；无guidance路径保留原逻辑。共享datasets/scoring；如绘制布局统一使用visualization。

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E11_short_budget_diagnosis.run --output experiments/E11_short_budget_diagnosis/runs/20261018_budget_oracle_dev5_seed0_r3
```

预计数分钟、CPU两线程、约数百MB至1GB内存、输出数MB；不使用GPU，不加载checkpoint、不训练、不扩读测试。源码快照、版本、环境、冻结划分、实际数据路径与来源哈希随运行保存。评分在求解计时后执行，任何失败以Q=10计入。

## 必要归因对照（主对照结束后、补充运行前固定）

主对照中teacher_relations还省去了反复构造拓扑/接触树的开销，不能直接把全部收益归因于教师关系更好。补充[control.py](control.py)：同5例、仅0.40秒与完整预算，把**原草图自身的分离关系与接触树**同样预先计算并固定；输入坐标、宽高、求解算法与teacher_relations一致，只改变关系来源。共10次新增求解、300秒保护，不重算主对照，也不新增样例。若原草图关系冻结后同样改善，应将其解释为固定结构/计算缓存的贡献；若教师关系仍明显胜出，才支持教师结构信息的价值。

## 五例阶段结论（历史）

已完成同5例主对照100次＋必要归因10次，全部硬合法。教师结构在短预算下有明确的机制信号，固定原关系也贡献收益；教师中心替换不能稳定减少修复工作。当前尚未达到“0.5秒内维持原Q”，不扩大到泛化结论，不自动启动蒸馏数据生成或训练。

完整数字、逐例反例、计时边界、保存错误与恢复事实归[阶段报告](runs/20261018_budget_oracle_dev5_seed0_r2/analysis.md)；[主结果](runs/20261018_budget_oracle_dev5_seed0_r2/results.json)、[归因对照](runs/20261018_raw_relation_control_dev5_seed0/results.json)、[首次启动错误](runs/20261018_budget_oracle_dev5_seed0/results.json)。

只从已存结果重建报告与图，不加载数据、不求解：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E11_short_budget_diagnosis.report --main-run experiments/E11_short_budget_diagnosis/runs/20261018_budget_oracle_dev5_seed0_r2 --control-run experiments/E11_short_budget_diagnosis/runs/20261018_raw_relation_control_dev5_seed0
```

## dev100扩展（2026-10-18，查看扩展结果前固定）

用户要求继续研究并扩大数据量。本轮使用冻结workset_v2.dev_quality全部100例，覆盖全部100个worker、每worker1例，包括之前5例，不按Q或成功与否筛选。该面板已参与E8/E9开发，不作为独立未见测试。旧5例不支持泛化结论，保留全部旧结果，扩展结果另存。

冻结7组：原草图完整预算5秒（普通路径1.3秒＋必要救援）；原草图0.40秒；固定原草图分离/接触结构0.40秒；教师分离关系＋依据原草图选择接触树0.40秒；原分离关系＋依据教师选择接触树0.40秒；教师分离/接触结构0.40秒；完整教师成品0.10秒上限校验。所有固定结构组保持原草图坐标和宽高。接触树按选定分离方向构造，仅改变用于树权重/选边的几何来源；这是条件化2×2对照，不把不一致的接触方向强塞进另一套分离关系。教师几何仅用于离线生成关系，worker只收到关系三元组；完整成品组单独标记。

共100×7=700次求解，seed0逐例随机组顺序，CPU两线程，后处理源码与先前实验不变。预计6–12分钟，**整轮累计墙钟900秒保护**，不能在中断后重置预算；不训练、不启动训练集教师数据生成、不读取封存测试。超过预算就保存部分结果，不拆分规避20分钟规则。`--resume`只补缺失单元并保留调用账本、原配置与源码快照，拒绝算法变化；STOP在当前请求返回后停止。

主指标仍为全部样例平均Q、官方/独立硬合法、相对同例在线完整基线Q差、预定+0.002容差、实际时间；新增P95/最大Q、最差10%均值、合法行排保底保留率、时间P95/超过0.5秒比例、同时满足质量容差及0.5秒的比例。分别报告教师结构与原固定结构、分离/接触两项及交互；不因均值好而忽略尾部。原推理时间统一沿用既有记录，未来关系预测网络时间未知，oracle时延不能称为新模型速度。

入口[panel.py](panel.py)，每例只保存一份坐标池，逐组原子保存，不重写整个100例坐标文件：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E11_short_budget_diagnosis.panel --output experiments/E11_short_budget_diagnosis/runs/20261018_factorial_dev100_seed0
```

扩展100/100例、700/700次已完成；[结构消融报告](runs/20261018_factorial_dev100_seed0/analysis.md)保留全体质量、尾部和逐例反例。后续当前解码器与接纳修正进一步改变了研究判断，最终结论见下方总报告，不能继续沿用五例的乐观解释。

### 当前解码器可表示的几何上限（主对照途中识别表示边界后、补充结果前固定）

当前decode用目标面积与log长宽比恢复自由块宽高，不能原样表达利用面积容差的E10成品。因此完整成品回灌只检验已知答案保留成本。本轮主dev100结束后，使用同100例追加两组：完美教师中心与log长宽比预测经**原冻结prepare_inference/decode**输出，再做0.40秒普通后处理；相同几何加教师关系再做0.40秒后处理。公开fixed/preplaced规格按E10流程恢复；没有新模型、checkpoint或reference坐标输入，教师仍为既有E10成品。

两组共200次求解，复用主700次的在线基线，均匀记账原推理时间；oracle生成时间单列排除，不当作实际新网络速度。[representation.py](representation.py)要求主对照完成；主对照与此补充**合计900秒累计保护**，不能用新目录重置预算，剩余预算不足时不自动启动。该补充检验当前输出形式的理想潜力，不证明可学习性。所有100例与失败均保留，不按Q筛选。

### 数值接纳反例复查（两阶段结束后固定）

补充组出现候选“求解成功、规范化前诊断无违反”但被拒绝的记录。仅对decoder_oracle_relations中Q≥5的全部5例（worker20/29/52/66/86）执行[acceptance_audit.py](acceptance_audit.py)，重建同一首步输入，给原生求解1秒保护，比较canonicalize前后严格hard_valid及原流程微小负坐标裁切；合法候选再交官方与独立检查。不改任何守卫、算法或900秒共享预算，不把这一事后反例面板当作总体效果评估。此项只定位潜在规范化问题，不调整配置择优。

### 保留合法原候选的最小修正复核（反例确认后固定）

5例审计确认4例在规范化前硬合法、规范化后产生新重叠。新增独立[conic_geometry_safe.py](backend/conic_geometry_safe.py)与[robust_safe.py](backend/robust_safe.py)，保留原E10/E11后处理不变：规范化结果不合法时，仅在原候选经相同微小负坐标裁切并通过**原严格hard_valid**时保留原候选；不放松任何容差、不改官方评分。原规范化合法路径继续沿用。

[safe_acceptance.py](safe_acceptance.py)在同dev100执行3组0.40秒目标：原草图、原草图＋教师关系、当前解码oracle＋教师关系；与已存原路径配对，共300次，不新增成员或训练。所有阶段（700＋200＋5审计＋300）继续共享900秒累计计算保护，剩余不足时不启动。该复核由已见反例触发，属于自适应开发，不是独立确认；仍需报告官方Q及软违反，合法原候选不必然拥有正确数值接触，不能以硬合法替代质量成功。

## 扩大研究最终结论与入口

全100例正式对照1200次及5例首步审计均完成，正式返回布局全部硬合法。仅给教师关系的五例乐观结论未保持；接纳审计确认4个严格硬合法候选被规范化改出新重叠。保留合法原候选的独立E11修正版已在全100例复核，不改变E10、硬阈值或官方规则。

教师中心/形状经当前解码、配教师关系和合法候选保留，已接近质量/时间目标，但平均Q与尾部仍未保持原基线；这仍是oracle，未训练新网络。下一阶段优先研究教师标签、学生可表达几何与廉价投影/修复的匹配，暂缓仅训练关系头。

数字唯一归属：[最终研究总报告](runs/20261018_safe_acceptance_dev100_seed0/analysis.md)。[修正复核结果](runs/20261018_safe_acceptance_dev100_seed0/results.json)、[解码器诊断](runs/20261018_decoder_oracle_dev100_seed0/analysis.md)、[数值审计](runs/20261018_canonical_acceptance_audit5_seed0/results.json)、[主结构消融](runs/20261018_factorial_dev100_seed0/analysis.md)。

只读取保存结果重建最终报告：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -m experiments.E11_short_budget_diagnosis.safe_report --main-run experiments/E11_short_budget_diagnosis/runs/20261018_factorial_dev100_seed0 --representation-run experiments/E11_short_budget_diagnosis/runs/20261018_decoder_oracle_dev100_seed0 --audit-run experiments/E11_short_budget_diagnosis/runs/20261018_canonical_acceptance_audit5_seed0 --run experiments/E11_short_budget_diagnosis/runs/20261018_safe_acceptance_dev100_seed0
```
