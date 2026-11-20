# 数据与共享缓存

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

2026-11-18 E153获准恢复MIB：复用reference_workset_v6训练前32000和原train500/dev500；原2000dev池及监测成员不变。新增全模块预测/增强角色缓存由E153独立管理，不能直接使用旧F/P冻结mask。仅重用已核验的旧公开输入/标签，原MIB平均面积派生语义与原始面积任务分别标注。

2026-11-19新增 [MIB dev500结构挑战集](../datasets/mib_dev500_challenge_v1/report.md)，登记于 `protocol/mib_dev500_challenge_v1.json`。从原MIB dev2000排除旧dev500后选取：按原36k训练N分布分配500例，在每个N内按公开grouping/MIB/贴边/固定与复合约束、连线密度评分选择。覆盖N21–120、每种3–7例；与旧500及MIB train64k成员/分片均无交叉。仅用于独立的新验证面板，不回写历史监测集。500例全模块预测/MIB/增强角色CPU缓存已完成。选择不使用Loss、Q或自由reference几何；未读取test、未做模型推理；有意富集结构复杂题目，不作为总体无偏估计。

2026-11-19用户改为重新随机抽样全新40k，允许历史重叠：[random_mib40k_v1](../datasets/random_mib40k_v1/README.md)按新分片池8100/450/450、seed2026111901均匀无放回扫描候选，剔除MIB不兼容后补齐36k train/2k dev/2k test；不按Loss/Q筛选。train/dev共38k构建predict-all增强角色/MIB缓存；监督同MIB统一形状、保留原中心并单独审计几何冲突。新test仅公开兼容检查，不索引自由标签或构建缓存。新train/dev监测500各按自身N分布分层随机选取。处理已通过5例及兼容性短检，后台进度见数据目录state.json；旧40k及其协议不改，不启动训练。

新random_mib40k_v1已完成：扫描112000，兼容54599，冻结36k/2k/2k，train/dev38000缓存就绪。E154 best50在完整新train/dev的Loss分别0.365063/0.359637，非reference项合计0.070874/0.071035，见 `experiments/E154_l0_no_bottleneck/runs/fresh_mib40k_loss_v1/report.md`。用户2026-11-19授权E155在此新36k上从头重训L0，固定新train500/dev500；不改旧实验和test封存边界。
