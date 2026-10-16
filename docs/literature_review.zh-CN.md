# FloorSet Challenge 高相关文献与研究切入点

检索与核对日期：2026-10-16。依据赛题导读、执行协议和公开约束要求筛选，优先考虑可变形矩形、fixed/preplaced、boundary/grouping/MIB，以及生成结果进入合法化后的质量保持。本文是文献筛选与方法建议，不是复现报告。

## 先读哪几篇

建议顺序：**FloorSet → Trinity → PeF → PARSAC → FlowPlace**。其中 FloorSet 解释任务来源；Trinity 提供生成与修复分工的方法参照；PeF 对合法化更直接；PARSAC 提供复杂约束搜索参照；FlowPlace 提供不同的生成机制。

本页只保存通用文献与方法分类，不为后续实验预设研究路线；具体实验决策在所属实验中预登记。

## 第一优先级：与赛题约束直接相关

### 1. FloorSet — a VLSI Floorplanning Dataset with Design Constraints of Real-World SoCs

Uday Mallappa、Hesham Mostafa、Mikhail Galkin、Mariano Phielipp、Somdeb Majumdar；arXiv 2024，ICCAD 2024 对应 DOI：`10.1145/3676536.3676814`。原题参考表记载的年份与会议记录存在差异，引用时以具体版本为准。

- [论文](https://arxiv.org/abs/2405.05480) · [PDF](https://arxiv.org/pdf/2405.05480) · [官方代码](https://github.com/IntelLabs/FloorSet) · [已有本地 PDF](../references/papers/FloorSet_2405.05480.pdf)
- **帮助**：理解数据生成过程、约束家族和 FloorSet/Lite 的区别，为标签偏差与泛化分析建立依据。
- **重点阅读**：数据生成、约束标注、基准与局限性。
- **边界**：原始数据集设定不等同于比赛修改后的规则；固定 outline、宽高比等仍按当前比赛 README/Q&A 解释。
- **核对深度**：论文摘要、官方仓库与本地赛题资料；已有 PDF 保留原件。

### 2. Trinity: One Differentiable Physics for Training, Refining and Scoring Generative Floorplanners

Shih-Ying Yeh 等；2026-10-04，arXiv:2610.04957v1，预印本。

- [论文及全文](https://arxiv.org/html/2610.04957v1) · [PDF](../references/papers/Trinity_2610.04957v1.pdf) · [作者项目页](https://wuqian-tang.github.io/projects/trinity/)
- **帮助**：在 FloorSet 上统一训练能量、生成后修复和分阶段评价；最适合借鉴其生成器/修复器/合法化器交叉对照。
- **重点阅读**：§3、§4、附录 E/F/R。
- **迁移限制**：论文修改训练集 MIB 形状；其公开验证成绩与我们的冻结划分不可直接比较。相关观察和数字仍是作者报告，尚未本地复现；项目入口已找到，未验证补充代码和权重可复用性。
- **核对深度**：全文的方法、结果与限制相关章节定向核读。

### 3. PeF: Poisson’s Equation-Based Large-Scale Fixed-Outline Floorplanning

Ximeng Li、Keyu Peng、Fuxing Huang、Wenxing Zhu；arXiv 2022，IEEE TCAD 42(6):2002–2015，2023；DOI：`10.1109/TCAD.2022.3213609`。

- [论文](https://arxiv.org/abs/2210.03293) · [PDF](../references/papers/PeF_2210.03293.pdf)
- **机制**：将软模块宽度纳入连续优化，以 Poisson 势能处理拥挤；随后用水平/垂直约束图清除残余重叠。
- **帮助**：为合法化中可能破坏中心与连线关系的问题提供结构参照；可以先借鉴约束图合法化，不必重做完整电势求解器。
- **迁移限制**：原任务有固定 outline，本题没有；其边界和密度项不能照搬，且摘要没有证明覆盖本题全部软约束。
- **核对深度**：原始摘要与元数据；PDF 已保存，尚未逐式审查。

### 4. PARSAC: Fast, Human-quality Floorplanning for Modern SoCs with Complex Design Constraints

Hesham Mostafa 等；2024，arXiv:2405.05495v3。代码仓库使用的相关论文标题为 “Design Planning of Modern SOCs with Complex Constraints: A Path Towards Human-Quality FloorPlanning”。

- [论文](https://arxiv.org/abs/2405.05495) · [PDF](../references/papers/PARSAC_2405.05495v3.pdf) · [官方代码](https://github.com/IntelLabs/parsac)
- **机制**：Constraints-Aware SA 与并行搜索；强调仅将放置约束加入普通 SA 的目标函数可能产生差解或非法解。
- **帮助**：阅读边界块和预放置块如何进入搜索表示与可行操作，建立比失效官方模板更有意义的传统对照。
- **迁移限制**：独立 PARSAC 仓库不等于比赛模板，也不能据其名称认定支持我们所有接口、软块变形和计分语义；需要适配审计。
- **核对深度**：原始摘要、版本记录和官方仓库说明。

### 5. Placement Constraints in Floorplan Design

E. F. Y. Young、C. C. N. Chu、M. L. Ho；IEEE TVLSI 12(7):735–745，2004。

- [作者机构记录](https://research.cuhk.edu.hk/en/publications/placement-constraints-in-floorplan-design-2/) · [作者 PDF](https://www.cse.cuhk.edu.hk/~fyyoung/paper/tvlsi04_7.pdf) · [本地 PDF](../references/papers/PlacementConstraints_2004.pdf) · [作者源码与数据页](https://www.cse.cuhk.edu.hk/~fyyoung/source.html)
- **机制**：统一处理 preplace、range、boundary、alignment、abutment、clustering 等放置约束。
- **帮助**：适合查 grouping 接触、边界关系与预放置冲突能否在拓扑层表示，避免所有约束都靠坐标罚项拉扯。
- **迁移限制**：原文约束是否强制满足、分组如何定义，必须逐项映射；本题 grouping 是非零公共边连通，不能只用中心距离近替代。
- **核对深度**：作者机构摘要、作者 PDF 入口及源码页；尚未复核源码。

## 第二优先级：生成机制与架构备选

### 6. FlowPlace: Flow Matching for Chip Placement

Peng Xie 等；DAC 2026，arXiv:2604.23658。

- [论文](https://arxiv.org/abs/2604.23658) · [PDF](../references/papers/FlowPlace_2604.23658.pdf) · [官方代码](https://github.com/lamda-bbo/flowplace)
- **机制**：mask 引导合成数据、flow matching 和处理不重叠的硬约束采样。
- **帮助**：研究如何在生成过程中保留可行性，以及少步生成能否减少完整 solve 成本；适合作为直接几何回归以外的候选。
- **迁移限制**：宏布局任务的固定尺寸和画布假设，与本题软块形状及解的 bbox 边界不同；论文的零重叠结论不能扩展成本题全部硬合法。
- **核对深度**：原始摘要、官方仓库与引用信息；代码未运行。

### 7. Chip Placement with Diffusion Models（ChipDiffusion）

Vint Lee、Minh Nguyen、Leena Elzeiny、Chun Deng、Pieter Abbeel、John Wawrzynek；ICML 2025，PMLR 267:33499–33514；arXiv:2407.12282。

- [正式论文页](https://proceedings.mlr.press/v267/lee25y.html) · [PDF](../references/papers/ChipDiffusion_ICML2025.pdf) · [官方代码](https://github.com/vint-1/chipdiffusion)
- **机制**：利用合成数据训练扩散模型，推理时用几何/连线引导，实现对新电路的零样本宏布局。
- **帮助**：参考连接条件编码、合成训练分布与真实电路泛化的关系，以及 guidance 和后处理的明确分工。
- **迁移限制**：不是本题完整约束求解器；官方示例包含多样本、长后处理循环，复用时要计算总求解时间，不能只比较网络前向。
- **核对深度**：正式摘要和官方运行说明；PDF 保存，未运行。

### 8. Physics-Guided Geometric Diffusion for Macro Placement Generation（MacroDiff+）

Jongho Yoon、Jinsung Jeon、Seokhyeong Kang；IJCAI 2026，arXiv:2605.16451。

- [正式论文页](https://www.ijcai.org/proceedings/2026/735) · [arXiv](https://arxiv.org/abs/2605.16451) · [PDF](../references/papers/MacroDiffPlus_2605.16451.pdf) · [代码](https://github.com/jhy00n/MacroDiff-plus)
- **机制**：异构 GNN 编码连接，Transformer 编码全局几何，在采样时引入显式物理梯度。
- **帮助**：给“连接拓扑 + 当前几何”双通路提供具体架构参照；适合比较相同编码器下，训练约束损失与采样引导的作用。
- **迁移限制**：原任务是宏布局，不可默认覆盖软块、MIB、grouping 与动态 bbox 边界。代码入口来自作者论文，尚未审查仓库。
- **核对深度**：原始摘要与正式会议记录。

### 9. DiffPlace: A Conditional Diffusion Framework for Simultaneous VLSI Placement Beyond Sequential Paradigms

2025 年预印本，arXiv:2510.15897；以具体版本元数据为准。

- [论文与 PDF 入口](https://arxiv.org/abs/2510.15897)
- **机制**：同时生成宏坐标，结合 energy guidance 与约束流形扩散。
- **帮助**：为“逐块插入还是整体生成”提供另一种对照，并检查同时生成怎样处理重叠。
- **迁移限制**：低重叠率不等于通过严格不重叠检查，更不等于满足本题全部约束。
- **核对深度**：原始摘要与论文页；未下载、未复现。

### 10. Generalizable Floorplanner through Corner Block List Representation and Hypergraph Embedding

Mohammad Amini、Zhanguang Zhang、Surya Penmetsa、Yingxue Zhang、Jianye Hao、Wulong Liu；KDD 2022:2692–2702；DOI：`10.1145/3534678.3539220`。

- [出版入口](https://doi.org/10.1145/3534678.3539220) · [会议官方日程](https://kdd.org/kdd2022/assets/ADS_Paper_Schedule.pdf)
- **帮助**：CBL 拓扑表示与超图嵌入为自回归构造提供候选，重点问题是“预测拓扑是否比回归坐标更能保留合法关系”。
- **迁移限制**：本题传入的是加权二元边表，未提供原始高阶 net 身份；不能把二元边包装为超图后就宣称恢复了高阶连接。
- **核对深度**：官方赛题引用和会议日程核对题录；出版平台全文访问失败，方法细节待全文核读。未找到并验证作者代码，不推断其开源状态。

### 11. GraphPlanner: Floorplanning with Graph Neural Network

Yiting Liu 等；ACM TODAES，2022 年上线，28(2)；DOI：`10.1145/3555804`。

- [出版入口](https://doi.org/10.1145/3555804) · [作者论文记录](https://ephonic.github.io/publication/2022-01-01-Graphplanner-Floorplanning-with-graph-neural-network)
- **帮助**：为图到布局的直接学习路线提供基准阅读，检查连接编码、预测目标与后续放置流程。
- **迁移限制**：不能仅因用了 GNN 就认为它能解决本题复杂约束；应比较相同合法化器下的有效几何信息。
- **核对深度**：作者页题录及本地官方参考表；出版全文访问失败，具体网络与损失待核读。

## 第三优先级：按需查阅的传统方法

### 12. Modern Floorplanning Based on B*-Tree and Fast Simulated Annealing

Tung-Chieh Chen、Yao-Wen Chang；IEEE TCAD 25(4):637–650，2006；DOI：`10.1109/TCAD.2006.870076`。

- [作者机构记录](https://scholars.lib.ntu.edu.tw/entities/publication/eb0c505d-edc3-48e5-a388-94868dc5098d) · [出版入口](https://doi.org/10.1109/TCAD.2006.870076)
- **帮助**：B*-tree、紧凑打包与 Fast-SA 的经典依据，用于理解传统质量参照与候选解码。
- **边界**：它不是复杂 preplaced/MIB 约束的现成解决方案；读机制即可，当前不建议继续围绕温度参数反复打磨。
- **核对深度**：作者机构摘要及官方题录，未全文精读。

### 13. Piano: A Multi-Constraint Pin Assignment-Aware Floorplanner

Zhexuan Xu 等；2025，arXiv:2508.13161。

- [论文与 PDF 入口](https://arxiv.org/abs/2508.13161)
- **帮助**：阅读预放置、空白压缩与局部优化如何协同。
- **边界**：其模块边界 pin assignment、feedthrough 和固定 outline 不属于本题当前目标；我们的固定端口是输入，不应新增可学习 pin assignment 来解决另一个任务。
- **核对深度**：原始摘要与版本页；未下载、未复现。


## 文件与检索记录

- 已归档重点 PDF 共 7 篇，位于 `references/papers/`；原有 FloorSet PDF 保留在同目录，不重复下载。
- [下载来源、大小及 SHA-256](../references/papers/manifest.json)。文件头已检查为 PDF；下载成功不代表已阅读全文或复现。
- 核心检索词：`FloorSet floorplanning diffusion flow matching`、`PARSAC complex design constraints`、`PeF Poisson floorplanning`、`placement constraints floorplan design`、`ChipDiffusion FlowPlace MacroDiff`、`Generalizable Floorplanner Corner Block List`。
- 来源优先级：原始论文/arXiv、正式会议论文页、作者机构与官方仓库。搜索聚合页仅用于发现线索。
- arXiv 检索接口遇到限流，随后通过网页搜索和原始论文页核查；部分 ACM 全文入口不可访问，已在条目中明确标注。
- 本次未安装论文依赖、未运行外部仓库、未启动训练或推理、未读取封存测试集。
