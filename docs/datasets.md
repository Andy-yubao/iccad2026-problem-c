# 数据版本与位置

数据由[protocol/paths.json](../protocol/paths.json)解析，官方原件和[安装清单](../datasets/datasets_manifest.json)保留，不重复下载/解压。原训练约9000分片；官方公开验证100例仅作兼容和公开对照。

| 索引 | 作用 |
|---|---|
| random_reference16k_v1.json | **E114第4/12层监督之后的新默认训练视图**；父train36k内固定随机16k；新train500，原dev2000/dev500/test2000不变 |
| random_reference40k_v1.json | 新研究父集合；全新随机原始面积40k，36k/2k/2k；当前E114按原36k登记；MIB忽略 |
| split_v1.json | 原始完整分片80/10/10；7200/900/900分片，始终冻结 |
| workset_v2.json | 历史20k/2500/2500任务，不再作新训练默认 |
| derived_workset_v3.json | 公开MIB平均面积派生任务16k/2k/2k |
| distillation_workset_v4.json | 历史教师选优分布，只用于对应实验 |
| reference_workset_v5.json | 历史reference任务16k/2k/2k，固定train/dev500 |
| reference_workset_v6.json | 嵌套扩展集合；E95读取经核验的原16k前缀 |

索引均在protocol/，旧索引不覆盖或重新抽样。2026-11-06用户授权建立独立新集合：新随机seed，原始9000分片重排为8100/450/450互斥池，在池内均匀无放回抽36000/2000/2000；没有旧名单、MIB相容性、模块数配平或质量筛选。固定train/dev各500监测子集一并冻结。历史曝光可能重合，不称旧模型独立未见测试。

## 后续16k训练子集

2026-11-07用户要求：E114第4/12层监督之后的新实验用16000训练数据，成员从现有36000训练池中挑选。三组E114保持原数据、Loss、调度和身份。

入口为 `datasets.random_reference16k`，规则在 [research_training16k_v1.json](../protocol/research_training16k_v1.json)。固定seed2026110716000均匀无放回抽16000，按父索引顺序枚举，仅为了稳定索引；不读取标签/预测/得分选题，不做N配平。一次生成 [random_reference16k_v1.json](../protocol/random_reference16k_v1.json) 后拒绝覆盖，不在每次实验中重抽。它是新random40k子集，不是历史MIB/v5的旧16k。

train_monitor500以独立seed2026110716001从子集内抽取，面板索引相对16k；dev2000及原dev_monitor500完整保留，test2000封存。父索引、成员顺序、规则、开发/测试成员均有哈希核验。原NPZ/PNG直接复用，无新载荷目录，不下载、复制或渲染，不改变面积/规格。模型图特征与Loss继续由实验拥有，继承无MIB、全模块预测规则。

```powershell
.\.conda\envs\iccad2026\python.exe -B -m datasets.random_reference16k prepare
.\.conda\envs\iccad2026\python.exe -B -m datasets.random_reference16k check --payloads 5
```

prepare只建立冻结索引，已有文件则拒绝重抽；check验证16000唯一成员、父集合归属、监测面板和开发/测试元数据，最多检查5个train载荷。后续读取：`from datasets.random_reference16k import RandomReference16k`，支持train、train_monitor500、dev、dev_monitor500，返回字段与原RandomReference相同；test角色在任何载荷访问前拒绝。读取某成员用 `dataset.records[dataset.indices[i]]`，需要父训练索引时用 `dataset.parent_indices[dataset.indices[i]]`。自由reference/图片/存储指标仍只作监督，不能进入模型。

处理检查结果写入 [random_reference16k_preparation.json](../protocol/random_reference16k_preparation.json)。本次不改变已运行或已登记实验，不启动新训练，不开放测试。

新集合保留原始面积、全部约束字段、连接及原始reference；不平均面积、不协调MIB形状、不修复。图片为512×512 RGB、固定ID配色和亚像素覆盖，画布只依原始公开面积、端口和预放置范围，禁止裁剪或用自由reference拟合画布。每例PNG及NPZ，逐文件/来源SHA256、原始成员和种子可追溯；目录由paths.json的random_reference40k_root解析。后续模型必须忽略原始MIB字段，free reference/存储指标及图片仅作监督，不作输入。

准备入口[E108](../experiments/E108_random_reference40k/README.md)，新reader为`datasets.random_reference.RandomReference`；test只制作后封存，reader拒绝测试，不因本次制作授权而开放模型评价。仅后台准备，无训练；失败不换题，不隐瞒坏例。历史MIB平均面积任务继续只用于原登记复现，不能把任务变化当成模型提升。

原始训练为datasets/floorset_lite/worker_*/layouts_*.th，公开验证为datasets/LiteTensorDataTest/，压缩原件为datasets/archives/。实际位置始终以paths.json为准。E101不同运行使用各自已登记成员；64k运行复用已验证E58嵌套成员与原E42有效缓存，转换为相同FP32磁盘映射供按批传输；三阶段64k链路现已停止。E58仍在原历史目录，E42缓存现位于history/20261103_final_delivery_consolidation/experiments/E42_joint_regularization，由E101路径映射解析。旧缓存均保护，不是待删临时文件。

训练仅用冻结train白名单，dev不反传。测试默认封存，旧v2已暴露957例/39个worker的历史不改写；v5选择已排除相关暴露。新测试须用户明确发起。公开固定/预放置规格可输入，自由reference只作监督/展示/计分。

共享datasets处理路径、成员读取、面积变换与标签遮罩；图特征/采样/Loss归实验。训练原标签WHXY须转换成输出XYWH，公开验证多边形不能套用同字段。缺失时先核实路径和安装证据，再查历史下载脚本的实际位置与安装清单，明确需要后恢复，日常不运行重解压。
