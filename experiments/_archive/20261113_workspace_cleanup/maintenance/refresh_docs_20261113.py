from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[1]
FILES={
'AGENTS.md':'''# ICCAD 2026 Problem C

目标：研究 FloorSet 矩形布局生成与约束后处理，在最终硬合法的前提下改善布局质量和完整耗时。

- 总入口：`README.md`；执行与研究边界：`docs/protocol.md`。
- 当前结果：`docs/current_status.md`；研究摘要：`docs/research_history.md`。
- 数据路径与冻结身份：`protocol/`、`docs/datasets.md`；实验位置：`docs/experiment_catalog.md`。
- 源码和原始结果：`experiments/`；历史实验：`history/`；唯一布局展示入口：`visualization/README.md`。
- 当前研究基线 E119 best55：原 dev500 最终500/500硬合法，混合Q约1.021568；竞赛交付仍为归档 E101。E133 FM本轮混合Q约1.157321，未改善基线。

开始工作先读 README 和相关长期文档；具体运行状态、权重和授权以实验内记录为准，不把历史命令当新启动授权。
''',
'README.md':'''# ICCAD 2026 Problem C

研究 FloorSet 的矩形布局生成与后处理：网络生成全部模块的位置与形状，再由后处理满足固定规格和硬约束。评价同时看最终合法性、本地研究Q和完整逐例耗时。

| 入口 | 内容 |
|---|---|
| [当前结果](docs/current_status.md) | 模型对照、队列状态与结果原件 |
| [执行约定](docs/protocol.md) | 数据、预测边界、训练、评价与停止规则 |
| [研究摘要](docs/research_history.md) | 主要发现和证据局限 |
| [数据说明](docs/datasets.md) | 冻结集合、路径与共享缓存 |
| [实验目录](docs/experiment_catalog.md) | 现有源码和归档的实际位置 |
| [布局展示](visualization/README.md) | 唯一分组色/黑框/图例样式，可选中间层滑块 |
| [工作区结构](docs/workspace_structure.md) | 文件职责和保护对象 |

E119 global2 best55 为主要研究基线：原 dev500 最终500/500硬合法，混合Q **1.021568**。E120、后续监督和本轮 FM 尚未改善该结果。竞赛交付身份仍是归档 E101；本地无MIB研究Q不等于正式比赛成绩。

浏览 E119 已存结果：

```powershell
& .\\.conda\\envs\\iccad2026\\python.exe -B -m visualization --e119 --port 8779
```

访问本机 `http://127.0.0.1:8779/`。12层示例及PNG导出见可视化README。Python环境在 `.conda/envs/iccad2026/`，数据通过 `protocol/paths.json` 定位。

2026-11-13已精简长期文档并统一可视化；旧版位于 `history/20261113_workspace_cleanup/`。根目录 `output/` 已按用户要求删除，不再作为长期结果位置。实验原始记录、权重、数据和有效缓存保留；本次整理不启动或恢复研究任务。
''',
'docs/START_HERE.md':'''# 接手入口

先读根目录 [README](../README.md)，再按任务阅读 [执行约定](protocol.md)、[当前结果](current_status.md) 与 [实验目录](experiment_catalog.md)。研究判断补读 [研究摘要](research_history.md)；数据与性能任务分别读 [数据说明](datasets.md) 和 [GPU工程](gpu_training.md)。

当前主要研究模型为 E119 global2 best55，竞赛交付模型仍为 E101 原 C5。二者的数据/后处理口径不同，不混称同一模型。E133 本轮 FM 质量评价独立于原自动评价失败记录，见当前结果。

`AGENTS.md` 只作目标与导航，实验细节放实验内；状态以 `queue_state.json`、run 的 `results.json` 和实际进程为准。旧授权、STOP、预算和冻结源码不因文档整理而改变。清理后所有布局展示统一调用 [visualization](../visualization/README.md)，不再建专用界面。
''',
'docs/workspace_structure.md':'''# 工作区结构

| 位置 | 职责 |
|---|---|
| `README.md` / `AGENTS.md` | 项目入口 / 简短目标与导航 |
| `docs/` | 当前协议、结果摘要、研究结论、数据与工程说明 |
| `protocol/` | 冻结数据成员、面板、模型身份、机器协议与路径 |
| `experiments/` | E119/E120及后续E131–E133；各自维护源码、计划、状态和runs |
| `datasets/` | 原始数据、已生成NPZ与持久化CPU张量缓存 |
| `scoring/` | 官方评分适配、独立硬检查、几何诊断 |
| `visualization/` | 唯一已存布局展示与PNG导出；数据接口见其README |
| `history/` | 退役实验、旧文档及清理清单；不作待执行队列 |
| `references/` / `third_party/` | 官方材料、阅读资料、固定上游源码 |
| `.conda/` / `.cache/` | 环境、有效运行缓存及可视化服务日志 |
| `tmp/` | 一次性工作文件，不作长期运行入口 |

长期结果归所属实验的 `runs/` 或独立 `evaluation_*/`；不要重新建立根 `output/` 作为杂物目录。实验内部已登记的同名 `output/` 是历史评价证据，本次没有删除。

2026-11-11的29个退役实验保留在 `history/20261111_experiments_consolidation/experiments/`；E101深层隔离运行根部分路径超过MAX_PATH，读取用Windows长路径。2026-11-13旧可视化与旧文档移入 `history/20261113_workspace_cleanup/`，现行只保留一套样式。

保护权重、best/latest、失败现场、STOP、数据原件和有效缓存。整理前检查实际运行、依赖、链接与绝对路径；旧实验按原身份复现，不批改冻结源码/日志里的历史路径。恢复归档或旧STOP需明确授权。Git有既有迁移改动，不reset或批量暂存，见 [Git约定](git.md)。
''',
'docs/protocol.md':'''# 项目执行约定

本页只保留持续适用的规则；具体实验配置、授权和结果放所属实验。历史文档不构成新启动授权。

## 数据与模型边界

- 数据路径由 `protocol/paths.json` 解析。默认新研究使用固定 random40k 父集合的16k训练视图，见 [数据说明](datasets.md)；已登记36k等历史实验忠实保留，不追溯改写。
- 相同冻结成员与预处理优先复用已有NPZ和兼容的持久化CPU张量缓存，先核对成员顺序、原始清单、字段版本、dtype及任务语义；只构建缺失/失效部分。16k按冻结索引取36k视图，不重新抽样。原始生成、静态预处理、已有缓存加载、模型初始化/图捕获分开记时。
- 网络输入只含公开面积、连接、端口、约束和允许的草稿。reference自由坐标/形状、教师标签和参考评分分母不作模型或求解器输入；reference只用于登记监督、展示与评价。
- 后续研究使用原始面积、忽略MIB：不做平均面积/同形变换，不在输入、Loss、后处理或选模中施加MIB；软违规分子与分母都排除MIB。官方原件不改，本地研究Q不冒充正式成绩。机器约定：`protocol/research_no_mib_v1.json`。
- 预测全部模块的位置与形状，包含F/P。公开固定规格可作条件，不mask冻结预测、不覆盖草稿；约束满足交给后处理，最终按原规格独立硬检查。机器约定：`protocol/model_prediction_boundary_v1.json`。历史模型按其原登记复现。
- train用于训练；dev用于固定监测、Loss-best和终末评价。test默认封存，只有明确的成员与用途授权才能读取；已有测试曝光不能称全新未见测试。本次清理不读取test载荷。

## 训练与停止

- 启动前冻结初始化、源码/权重、数据/监测成员、Loss项与聚合、seed、batch/精度、LR/早停/数值门禁、预算和后续阶段。新提案和准备完成不等于启动授权。
- 反传、固定监测、best、LR与早停使用同一登记Loss；用固定dev面板算术均值选唯一own-Loss-best。不同Loss总值不直接横比，旧例外只用于原实验。
- 门禁在Adam前检查；错误不跳批、不放宽阈值、不自行降LR或无限重试。实施短检通常用已登记同五训练例；可丢弃更新后正式scratch须重新初始化。五例成功不证明泛化。
- best用于冻结选模；latest保存模型、Adam、RNG、排列/游标、调度与累计预算。恢复不清零预算；改变目标、数值路径或重新建Adam须登记迁移，保留原件。
- 正常早停、时间耗尽、用户停止与数值失败分开报告；dev早停不等于train充分拟合。用户STOP应阻断后续阶段、安全保存并核实进程退出。旧STOP不自动清除。
- 长任务默认给出命令；用户已明确授权的范围可后台执行，不反复确认。隐藏后台任务记录PID和日志，错误/用户停止阻断队列；不自动追加候选、seed、测试或旧实验恢复。

## 评价与展示

- Q评价前冻结唯一own-Loss-best及成员/后处理预算，不按Q重选checkpoint、步数或预算。优先复用同身份已完成结果，只补已授权的缺失部分。
- 草稿Loss/几何/硬合法、最终硬合法/Q与完整耗时分别报告。独立硬检查不放宽原规格。加权Q、平均Q、混合Q=(前两者)/2明确标注，Q越低越好。
- E101原完整后处理与E98.depth_budget.v1固定工作量不是同一口径。单seed、反复使用的dev、历史对照、不同训练量和多因素变化均限制因果解释。
- 完整逐例耗时包含输入复制/遮罩、特征、传输、网络、全部后处理/回退及必要同步/硬检查；报告均值/P95/最大与>250ms数。磁盘、加载、图捕获/预热、评分和绘图单列；吞吐不替代延迟。
- 唯一展示入口见 [visualization/README](../visualization/README.md)：G分组色、无组灰色、黑框、编号/G/F/P、边界徽记和底部图例，无MIB。reference／草稿／后处理同坐标范围；滑块只切真实已存中间层，缺失明确标注，不插值或复制末层。不新增人类评分界面。
- 展示工具只读保存坐标，不自行训练、推理、修复或重评；采集新模型中间输出的调用方须另有相应授权。

## 维护

模型/Loss/后处理源码由各实验独立维护，公共数据/评分/展示分别归共享目录，不跨实验导入私有算法。性能改动先测量，见 [GPU工程](gpu_training.md)。

研究基线为 `protocol/research_baseline_v3.json` 的E119 best55；交付仍为归档E101原C5。冻结身份、权重、STOP、原始数据、有效缓存和失败证据受保护。清理先查运行/依赖与目标路径；必要原件同盘归history并留清单，不批改历史路径。根output已按2026-11-13用户请求删除；长期结果归实验。Git边界见 [git.md](git.md)。
''',
'docs/datasets.md':'''# 数据与共享缓存

路径统一由 [protocol/paths.json](../protocol/paths.json) 解析；原始清单见 `datasets/datasets_manifest.json`。不要重复下载、解压或重建已有数据。

| 冻结索引（均在protocol/） | 用途 |
|---|---|
| `random_reference40k_v1.json` | 原始面积父集合36k train / 2k dev / 2k test，固定train/dev500 |
| `random_reference16k_v1.json` | 父train36k内固定随机无放回16k，新train500，沿用原dev/test |
| `research_training16k_v1.json` | E114监督实验后新研究默认16k；特定36k授权只对登记实验有效 |
| `split_v1.json` | 历史原始分片80/10/10，7200/900/900分片 |
| `workset_v2.json` / `derived_workset_v3.json` | 历史20k任务 / MIB派生任务 |
| `distillation_workset_v4.json` | 历史教师选优分布 |
| `reference_workset_v5.json` / `reference_workset_v6.json` | 历史reference任务 / 嵌套扩展 |

random40k从9000原始分片重新登记互斥8100/450/450池，在池内均匀无放回抽样；不按MIB相容性、N或评分筛选。历史曝光可能重合，不将其称旧模型的全新未见测试。新研究保留原始面积、忽略MIB，历史索引不覆盖。

`datasets.random_reference.RandomReference` 只允许train/dev及固定监测面板；test载荷默认拒绝。`datasets.random_reference16k.RandomReference16k` 仅按固定索引取视图。reference标签与公开solver_inputs始终分离。E119于2026-11-09授权的test500已评价并曝光，名单与结果留在该实验；其余1500例不因此开放。

持久化CPU张量缓存已实现于 `datasets/static_tensor_cache.py`，数据在 `datasets/static_tensor_cache_v1/`。兼容36k与16k实验共享：核对成员/顺序、原始载荷清单、生产者字段版本、dtype、原面积/无MIB/全模块预测语义；16k映射父索引，只重建失效部分。模型状态、动态几何和CUDA图不是静态缓存。

日志应区分原始数据生成、静态预处理/首次构建、已有缓存加载及模型初始化/图捕获。当前缓存已存在，不能继续沿用历史“尚未实现缓存”的描述。准备新实验前检查兼容性；不要为可视化重算训练特征或读取test。
''',
'docs/gpu_training.md':'''# GPU训练工程

目标是提高实际吞吐、缩短完整epoch和逐例推理时间。GPU利用率、功耗和显存只作解释，不追求100%占用。

1. 先分解缓存加载、CPU组批、传输、前后向、Adam/门禁、监测与存盘；记录样本/秒、步耗时P50/P95和完整epoch，注明设备及并发。
2. 优先复用已核验的持久化CPU张量缓存。按需使用pinned staging、non_blocking与预取；公开输入和reference标签分开，只搬所需张量。
3. 全量GPU常驻先估算Adam、激活、监测和图捕获余量，避免共享显存换页。不能为提速静默更改batch、精度、成员顺序或mask。
4. 合并有限性/梯度/参数检查，减少逐参数CPU同步；Adam前门禁必须保留，失败不跳批。日志分项按登记频率汇总。
5. CUDA graph先做固定形状、预创建缓冲与无更新预热，保存/恢复RNG；不同LR、随机性和可变路径须正确处理。捕获不支持时可登记eager方案，保留失败证据，不静默更换数学目标。
6. 任何等价算子替换、GPU优化器或执行顺序改变都需必要数值/梯度/更新检查，并说明浮点差异；五例速度不能外推整轮收益。

E101历史双图短检约205→102ms属于特定FP32/batch32路径，非bit-exact恢复；E133训练CUDA捕获失败后登记使用eager FP32。原预算和失败现场保留，不能据此自动迁移其他运行。

训练吞吐、批推理微基准与完整逐例延迟分别报告。评价耗时口径见 [protocol](protocol.md)。
''',
'docs/research_history.md':'''# 研究摘要

只保留可复用结论；原始配置和结果在 [实验目录](experiment_catalog.md)。这是研究判断依据，不是待运行队列。

## 当前主线

E119在原始面积/无MIB/全模块预测任务上，用两层初始全局交互生成初稿，再以12层几何主干和共享头改进。best55原dev500混合Q1.021568，500/500最终硬合法，是当前主要研究基线。草稿仍0/500硬合法，最终可用性依赖后处理，不能把最终成绩全部归因于网络。原件见 [E119评价](../experiments/E119_global_initial_generator/runs/global2_seed0/evaluation_no_mib_dev500/)。

E120 Node12替换逐对几何MLP后，碰撞相关Loss改善没有转化为最终Q改善：混合Q1.039004。E119/E120后续监督调整和16层实验也未在相应固定dev500上改善原基线。Loss、重叠、最终软约束、Q及耗时必须分别解释；较低Loss不能代替实际质量。E122/E123等归档结果见20261111实验目录。

E119 public_priority_v1加强公开固定尺寸和预放置监督后，F/P尺寸及位置更准，但草稿组织退化；最终混合Q1.033645，高于原E119。原件见其 [变体评价](../experiments/E119_global_initial_generator/variants/public_priority_v1/runs/global2_seed0/evaluation_no_mib_dev500/)。不同own Loss总值不直接比较。

E132关系输入96×6的full/pure均正常早停；own dev Loss为0.403183/0.775958。该任务只授权训练，没有后处理/Q，不能据Loss宣布最终质量优胜。E131旧64×4在用户停止处保留完整状态，不自行恢复。

E133 FM best239用16k、纯速度MSE和Euler8，固定dev500混合Q1.157321，500/500最终硬合法；草稿重复覆盖10.136%（E119为4.795%），F/P尺寸误差较低但组织/连接较差。原自动评价0例失败与后续独立成功评价分别保留。五训练例4/8/16步诊断不构成按Q选步数；16k对36k、单seed及目标/参数差异不支持FM普遍无效的因果结论。见 [完整报告](../experiments/E133_flow_matching/evaluation_quality_20261113/report.md)。

## 仍适用的历史经验

- E101原C5＋完整后处理为竞赛交付；它和后来固定深度后处理的研究任务不可直接混排。其已停止的64k链路不是新默认模型。
- E98调权搜索已放弃；旧搜索和单步调权结论不作为新研究选型依据，STOP保持。
- E106代理对跨题结果的预测不等于局部可微方向有效；布局器训练未获该实验授权。E107在数值门禁停止，后续修订提案未实施，不自动放宽保护或重试。
- E109共享头逐层监督、E114监督位置研究曾形成阶段基线，现由E119接替。旧MIB/固定通道任务的好结果不自动迁移为新任务结论。
- 反复使用的dev、单seed、训练量/参数/监督同时变化及历史运行环境限制因果解释。正常dev早停不等于训练充分拟合。

## 下一步如何判断

先定位草稿组织、后处理软违反与耗时的主要误差，再设计一个能区分假设的受控改动。现有证据不支持只扩大步数、减重叠或降低Loss就必然改善最终Q。新Loss、反馈、逐步训练、FM变体、seed和测试均需单独明确范围。

详细历史综述和清理前诊断文字留在 `history/20261113_workspace_cleanup/documents_before/docs/`；其中指向已删除output的原始脚本/结果不再可用，不据此声称可完整重放。
''',
'docs/diagnosis_20261111.md':'''# E119诊断摘要（2026-11-11）

基于原E119 best55的固定dev500历史只读诊断：少数最终软违反承担主要超额Q，原报告估计soft占可约超额约70.7%；简单重叠指标不足以解释坏例。公开固定尺寸/位置监督能改善局部精度，却不保证全局组织和最终Q改善，后续public_priority_v1已有相应反例。

该轮H预算、容差及其他后处理改动没有被升级为新默认或交付。相关性、局部小样本与历史自适应开发不构成归因或全局最优证明。

2026-11-13按用户要求删除根output，原诊断脚本和逐例派生输出不再保留；完整旧文字在 `history/20261113_workspace_cleanup/documents_before/docs/diagnosis_20261111.md`。这里的70.7%是保留的历史汇总，不是本次重新核验的结果。可复查原E119逐例评价在其 `runs/global2_seed0/evaluation_no_mib_dev500/`，新研究应从原件验证需要的结论。
''',
'experiments/README.md':'''# 实验入口

| 目录 | 角色 |
|---|---|
| `E119_global_initial_generator/` | 主要研究基线global2 best55；含独立public_priority_v1变体和已授权test500历史评价 |
| `E120_node12_collision/` | Node12结构对照，训练/原dev500已完成 |
| `E131_relation_input/` | 旧64×4关系输入，用户停止，不自动恢复 |
| `E132_relation_input_96x6/` | full/pure训练已完成，无后处理/Q授权 |
| `E133_flow_matching/` | FM训练已完成；原自动评价失败现场保留，独立成功评价在evaluation_quality_20261113 |

具体身份看 `plan.json`、`authorization.json`、冻结权重记录；实际状态看 `queue_state.json` 和run内结果。文档和旧true开关不构成新的启动授权。长期结果留所属实验，公用数据/评分/展示不在实验间复制调用。

退役实验实际位置见 [完整目录](../docs/experiment_catalog.md)。E101交付及E98、E102–E118、E121–E130位于 `history/20261111_experiments_consolidation/experiments/`。本次整理未恢复旧实验、训练或评价。
''',
'history/20261113_workspace_cleanup/README.md':'''# 2026-11-13 工作区精简

按用户要求缩短AGENTS与长期文档，布局展示统一为E6分组色/黑框/底部图例的一套入口，支持真实已存中间层滑块。

- `documents_before/`：修改前文档原件，含AGENTS、README及docs；不作为现行授权。
- `visualization_legacy/`：旧可视化全部版本及已有图片/数据，同盘rename保留；现行入口仅 `visualization/README.md`。
- `manifest.json`：迁移文件清单/哈希和删除的根output文件清单。旧可视化首次迁移后的读取遇MAX_PATH，已用长路径补完整清单；不声称这部分有持久化的迁移前后双哈希证据。其余迁移前后逐文件哈希核对一致。
- `maintenance/`：本次一次性整理脚本和验证记录。

E133唯一成功的质量评价移回 `experiments/E133_flow_matching/evaluation_quality_20261113/`，关键启动登记/捕获失败记录也归该实验。根 `output/` 其余内容直接删除，不保留整目录副本。旧文件里的output路径是历史文字，可能已不存在。实验内部的 `output/` 未删除。

训练源码/权重、原始评价、数据、静态缓存、预算和STOP不变；本次未新增研究实验。旧可视化曾有专用评分器和图像任务，不再是现行布局展示方式，旧调用复现需先核对归档路径。
'''
}

for name,body in FILES.items():
    path=ROOT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(body,encoding='utf-8')

# Current state numbers come from owning result files, not old status prose.
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8'))
e119=read('experiments/E119_global_initial_generator/runs/global2_seed0/evaluation_no_mib_dev500/results.json')
e120=read('experiments/E120_node12_collision/runs/node12_seed0/evaluation_no_mib_dev500/results.json')
priority=read('experiments/E119_global_initial_generator/variants/public_priority_v1/runs/global2_seed0/evaluation_no_mib_dev500/results.json')
test=read('experiments/E119_global_initial_generator/evaluation_test500_20261109/recovery1_20261109/output/results.json')
body='''# 当前结果

核对日期：2026-11-13。这里是静态摘要，实时运行以实验内状态和实际进程为准。所有数字为已有结果，本次没有训练或重评。

## 原固定dev500，原预算后处理

| 模型 | best epoch | 混合Q↓ | 最终硬合法 | 草稿硬合法 | 完整均时/P95 ms |
|---|---:|---:|---:|---:|---:|
'''
for label,r in [('E119 global2（研究基线）',e119),('E120 Node12',e120),('E119 public_priority_v1',priority)]:
    body+=f"| {label} | {r['best_epoch']} | {r['mixed_Q']:.6f} | {r['hard_legal']}/500 | {r['raw_hard_legal']}/500 | {r['mean_complete_seconds']*1000:.1f}/{r['p95_complete_seconds']*1000:.1f} |\n"
body+='''| E133 FM（独立评价） | 239 | 1.157321 | 500/500 | 0/500 | 206.4/499.3 |

Q为原面积/无MIB本地研究分数；混合Q为加权Q和平均Q的平均，越低越好。E133训练16k、E119/120为36k，目标/参数不同；单seed、长期复用dev和历史运行环境不支持纯架构因果比较。完整耗时尾部仍存在，不能称逐例250ms达标。

## 状态与原件

| 实验 | 已核对状态 | 主要结果位置 |
|---|---|---|
| E119 / E120 | 队列complete | 各自runs下evaluation_no_mib_dev500/results.json |
| E119 public_priority_v1 | 队列complete；best56，新Loss未改善最终Q | variants/public_priority_v1/runs/global2_seed0/ |
| E131 64×4 | user_stop；full在93轮处停，pure未启动 | E131/queue_state.json及runs；STOP保留 |
| E132 96×6 | complete；full153轮/best145/Loss0.403183，pure144轮/best136/Loss0.775958 | E132/runs/full_seed0与pure_seed0/results.json；无后处理/Q |
| E133 FM | 247轮正常早停/best239；原queue保留needs_attention | runs/fm_seed0；独立成功评价evaluation_quality_20261113/ |

E133原自动评价因重复dict字段在0例退出；本次独立评价的500例成功结果不回写原失败队列、不重训、不修改权重、不清零原8h预算。其[完整报告](../experiments/E133_flow_matching/evaluation_quality_20261113/report.md)及逐例坐标均保留。冻结权重SHA256为 `1510d786df358ad3e4390596b948dd2c64289945b7f4639ef1ed9acef80d352f`。

E119 best55已授权的独立test500也已完成：'''
body+=f"混合Q{test['mixed_Q']:.6f}，最终{test['hard_legal']}/500硬合法。结果位于 `experiments/E119_global_initial_generator/evaluation_test500_20261109/recovery1_20261109/output/results.json`。该500例已曝光，其余1500例继续封存。\n\n"
body+='''竞赛交付保持归档E101原C5＋完整后处理；E119研究身份见 `protocol/research_baseline_v3.json`。本次未更换模型或预算、恢复旧任务、读取test载荷。当前没有由整理产生的待执行研究任务。

布局展示统一入口：[visualization/README](../visualization/README.md)。根output已删除；旧长期文档与可视化原件见 `history/20261113_workspace_cleanup/`。
'''
(ROOT/'docs/current_status.md').write_text(body,encoding='utf-8')

# Compact directory index, limited to actual top-level experiment roots.
entries=[]
for parent in [ROOT/'experiments']+[p/'experiments' for p in (ROOT/'history').iterdir() if p.is_dir() and (p/'experiments').is_dir()]:
    for p in parent.iterdir():
        m=re.match(r'E(\d+)_',p.name)
        if p.is_dir() and m:entries.append((int(m[1]),p))
entries.sort(key=lambda x:(x[0],str(x[1])))
catalog='# 实验实际位置\n\n2026-11-13按磁盘顶层实验目录生成。目录存在不代表可启动；状态/身份看该目录内记录，历史路径与命令保留当时语境。\n\n| 实验 | 实际目录 |\n|---|---|\n'
for _,p in entries:
    rel=p.relative_to(ROOT).as_posix()
    catalog+=f'| {p.name} | [{rel}](../{rel}/) |\n'
catalog+='\n更早未按E编号命名的实验见对应history归档说明。E101隔离运行根深层路径须用Windows长路径工具读取。唯一布局展示入口位于visualization，不从历史界面选默认。\n'
(ROOT/'docs/experiment_catalog.md').write_text(catalog,encoding='utf-8')

archives=sorted(p for p in (ROOT/'history').iterdir() if p.is_dir())
history='# 历史归档\n\n旧目录是原始证据，不是待执行队列。具体实验查 [实际位置](../docs/experiment_catalog.md)；当前结论查 [研究摘要](../docs/research_history.md)。旧授权、命令、output路径均按发生时解释，不批改冻结原件。\n\n'
history+='本次文档和旧可视化在 [20261113_workspace_cleanup](20261113_workspace_cleanup/README.md)；E101及2026-11-11退役的29个实验在 [20261111_experiments_consolidation](20261111_experiments_consolidation/README.md)。\n\n| 归档 | 导航 |\n|---|---|\n'
for p in archives:
    target=f'{p.name}/README.md' if (p/'README.md').exists() else f'{p.name}/'
    history+=f'| {p.name} | [查看]({target}) |\n'
(ROOT/'history/README.md').write_text(history,encoding='utf-8')
print(json.dumps({'rewritten_files':len(FILES)+3,'catalog_entries':len(entries)},ensure_ascii=False))
