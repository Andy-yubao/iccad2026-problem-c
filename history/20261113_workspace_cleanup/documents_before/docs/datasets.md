# 数据版本与位置

2026-11-09恢复实施：共享缓存已实现于datasets/static_tensor_cache.py，E117已接入；仅生成当前授权train36k/dev_monitor500的缺失张量，256索引一块，内容与版本哈希校验、原子写入，最多4个CPU工作进程。完整原始数据保留，兼容实验可复用；16k reader应映射原parent indices。首次仍需填充缓存，之后不重复转换。五训练例全部字段与旧sample/public_ports逐位一致；不改变模型输入边界或开封test。下文“缓存尚未实现”是修复前现场记录。

## 2026-11-09：跨实验复用静态预处理

2026-11-09数据复用约定：相同冻结成员及相同预处理语义的训练/监测数据，必须优先复用已有原始NPZ和兼容的持久化CPU张量缓存，禁止每次启动都逐例解压、重算静态图特征及Loss端口张量。先核对缓存身份（成员/顺序、原始数据清单、字段/预处理版本、dtype及面积/MIB/全模块预测口径）；若无兼容缓存，只首次构建并持久化，后续36k/其固定16k视图及模型实验共享，16k仅按冻结索引取视图。仅失效部分重建，保留来源和校验，不得混用旧MIB/固定通道任务缓存；标签仍与公开模型输入隔离，test继续封存。正常磁盘加载、按批传输及各模型独立CUDA图捕获仍需要，日志必须分别标明“原始数据生成/静态预处理或缓存构建/已有缓存加载/模型初始化与图捕获”，不得统称重新准备数据。该约定不修改当前E117活跃源码/数据路径/状态或累计预算；本次尚未实现共享张量缓存，后续新入口应先落实再启动。

当前E114/E115/E117的load_rows每次读取已有NPZ、验证载荷并重算sample/public_ports，结果仅驻留进程内存；原始数据并未重生成，但尚无这条路径可直接复用的持久化张量缓存。改进应在共享datasets中建立有版本身份的缓存，兼容实验只加载/索引已有张量，按原顺序和FP32语义取批；不把模型相关的Node26动态几何或CUDA图当成静态数据缓存。日志与结果须如实说明缓存是否已实施，写入约定不等于实现。

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
