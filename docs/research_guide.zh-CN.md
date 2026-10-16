# FloorSet Challenge 中文研究导读

本文件是阅读与实验准备说明，**不是官方题目译文或算法结果**。官方题目全文见 [Markdown](../references/official/problem_statement.md) / [PDF](../references/official/C_20260522.pdf)，澄清见 [Q&A 全文](../references/official/official_qa.md)。资料核对日期：2026-10-15。

## 要解决什么

给定 21–120 个模块的面积目标、模块间连接、固定端口及连接、放置约束，输出每个模块的矩形 `(x,y,w,h)`。目标是在保持合法性的同时，让布局紧凑、连接短、软约束违反少，并控制求解时间。

比赛允许传统启发式、机器学习及混合算法。题目中对 diffusion 与传统 SA 的比较是主办方的初步实验陈述，不是本工作区已复现的结论；官方未提供一个可直接复用的已训练神经网络。

建议阅读顺序：原题的 Inputs/Expected Output → Objective Function → Q&A → 官方评测器 → 优化器模板 → FloorSet 原始论文。需要训练时再研究训练 DataLoader 和损失函数。

## 硬约束、软约束与已放宽条件

| 类别 | 约束 | 意义 |
|---|---|---|
| 硬 | 无重叠 | 矩形可共享边，但不能相交出正面积 |
| 硬 | 普通软块的面积 | `abs(w*h-a)/a <= 0.01`，增大与缩小都受约束 |
| 硬 | Fixed-shape | 输入给定的 `(w,h)` 不可变，不能任意旋转交换宽高 |
| 硬 | Preplaced | 输入给定的 `(x,y,w,h)` 全部不可变 |
| 软 | Grouping/cluster | 组内通过非零长度公共边形成一个连通分量 |
| 软 | MIB | 组内 `(w,h)` 应相同 |
| 软 | Boundary | 触及**解的包围盒**指定边或角 |
| 已放宽 | 普通软块宽高比 | 比赛 README 允许任意宽高比 |
| 已移除 | 固定 outline | 不要求使用固定尺寸画布，通过包围盒面积及固定端口连接间接约束 |
| 已放宽 | 坐标精度 | 允许浮点数 |

Fixed-shape/preplaced 不使用软块面积误差检查替代尺寸要求；原题明确对它们忽略面积目标，以输入尺寸为准。若 preplaced 与 boundary 冲突，保持 preplaced 不动，计一次 boundary 软违反（Q&A A5）。

`constraints[:, :]` 五列为 `[fixed, preplaced, MIB, grouping/cluster, boundary]`。MIB/grouping 的 0 表示无该约束，其余值为组 ID。Boundary 是位掩码：left=1、right=2、top=4、bottom=8；top-left=5、top-right=6、bottom-left=9、bottom-right=10，必须满足所有置位的方向。

## 输入与标签不要混淆

官方文档将连接关系抽象成矩阵；代码实际传入 `(source_index,target_index,weight)` 形式的边列表张量。面积、模块、边和端口张量可能以 `-1` padding，不能对 padding 元素进行有效模块计算。

当前固定提交的实际接口是：

```python
def solve(self, block_count, area_targets, b2b_connectivity,
          p2b_connectivity, pins_pos, constraints,
          target_positions=None):
    # Return [(x, y, w, h), ...], one row per valid block.
    ...
```

`target_positions` 不是全部模块的参考答案。评测器构造一个初值全为 `-1` 的 `[n,4]` 张量：fixed-shape 只暴露尺寸，preplaced 暴露完整位置和尺寸，其余块保持 `-1`。研究算法必须使用这部分合法输入以满足硬约束。

训练标签 `fp_sol` 为 `(w,h,x,y)`，优化器输出和 `target_positions` 为 `(x,y,w,h)`，需要显式转换。公开验证数据的标签是多边形与 metrics，结构也不同。Q&A A16 指定监督学习/模仿以 `fp_sol` 为布局目标；`tree_sol` 需要特定 B*-tree 解码约定，并非比赛提交的必需数据。

## 怎么计分

对一个可行布局，先计算：

$$
Q=1+0.5\left(\max(0,\mathrm{HPWLgap})+\max(0,\mathrm{Areagap})\right),
$$

$$
V_{\mathrm{rel}}=\frac{V_{\mathrm{boundary}}+V_{\mathrm{grouping}}+V_{\mathrm{mib}}}{N_{\mathrm{soft}}},
\quad R=\max(0.7,\mathrm{RuntimeFactor}^{0.3}).
$$

然后 `Cost = min(Q * exp(2*V_rel) * R, 9.999999)`。任意硬约束违反则 `Cost=10`。`N_soft=0` 的无软约束情形，评测器采用 `V_rel=0`。HPWL 是连接端点的模块中心/固定端口之间的**加权 Manhattan 距离之和**；bbox 面积是所有模块的最小包围矩形面积。

Grouping 计每组的连通分量数减 1；MIB 计每组不同尺寸数减 1；boundary 每个受限块最多计 1。归一化分母是 boundary 块数加各组 `组大小-1` 的总和，不含 fixed/preplaced。

100 个用例按 `exp(block_count/12)` 归一化加权，越小越好。116–120 个模块的五例合计约占总权重 34%；**单个 120-block 例约占 8%**（由公式计算）。达到 baseline、无软违反、运行时间处于跨队伍中位数时分数为 1.0，速度收益封顶后理论下界为 0.7。

本地官方 CLI 把 `median_runtime` 固定为 1 秒，但仍传入实测求解时间，因此其 `RuntimeFactor=实测秒数/1秒`，**不是固定为 1**。研究中要得到纯质量与软违反的分数，须显式用 `runtime=median_runtime=1` 评分，并另记真实求解时间；这种中性 runtime 分数也不能直接与正式比赛总分比较。历史 Alpha/Beta CSV 中的每例中位数只能用于标注清楚的历史估算；正式评测是逐例、跨当次所有参赛提交求中位数。Alpha 是公开验证集上的流程检查，Beta/Final 使用隐藏集，不能把 Alpha 与 Beta 排名的差异当成同一测试集上的算法进步。

## 已核对的差异与疑点

| 问题 | 原始资料/代码 | 工作区采用的处理 |
|---|---|---|
| 数据版本与目录 | README 写 `LiteTensorData/`；`lite_dataset.py` 下载 v2 并检查 `floorset_lite/worker*` | 数据脚本选 v2，解压到仓库根，按代码检查；实际解压尚未执行 |
| 数据大小 | 通用 README、比赛 README 的旧估计不一致 | 使用当前官方 API 元数据：训练 6,627,425,876 bytes，验证 69,379,249 bytes |
| 固定 outline / hard blocks | 原题背景和数据介绍保留旧表述；比赛 README 明确去掉 fixed outline，并允许普通软块变形 | 保留原题；导读单列比赛规则，不把原始数据几何性质当作求解约束 |
| 参考解的软违反 | 题目 Related Work 称约束严格满足；Q&A A4 说明训练数据有软违反是正常现象 | 不假定参考标签的软违反计数必为 0 |
| 负 gap | 原题公式未显式展示 `max(0,gap)`；代码和 A15 明确截断 | 研究计算以代码/A15 所述截断版本为准，原文公式不改写 |
| 权重表 | 比赛 README 有一格把 `n=120 alone` 写成 34%，同时把 116–120 bucket 也写成 34% | 以 `exp(n/12)` 公式为准，单例约 8%，五例合计约 34% |
| 浮点容差 | 原题表述严格无重叠/精确不变；代码 overlap 双方向阈值 `1e-6`，固定规格检查容差 `1e-4` | 区分数学规则与该快照的数值实现；不依赖容差漏洞求解 |
| 运行时限 | 提交指南称无上限；下载的 `op_wrapper.py` 使用 `subprocess.run(timeout=60)` | 归档原样保留，未来实际包装时按预算处理；A20 允许修改提交的 wrapper |
| 冷启动 | A14、A28 排除加载/初始化；示例 wrapper 每次 solve 都新启子进程 | 不假定子进程冷启动会被该 wrapper 自动扣除；实例化一次与进程生命周期分别核对 |
| 内存 | A2/A3 写 128 GB RAM；提交指南写 250 GB | 原始两处都保留；当前无法据此证明唯一最终内存限额 |
| CUDA | 提交指南写 12.5 | A22 更正为 CUDA 13.0，驱动 580.82.07；PyTorch `2.12.0+cu130` 的文字记录不变 |
| A18 的回答 | 问宽高比，答“与公开验证样本相似” | 保留原文，不把这个可疑回答解释为新增宽高比限制 |
| 新版 PyTorch 兼容 | 上游 `torch.load` 未指定 `weights_only`；Beta 报告与指南的版本说明也存在历史差异 | 这里只归档，不声称已验证这些版本可安装或可运行；环境准备时单独验证 |

## 环境与提交

官方指南记录：Debian GNU/Linux 13、Python 3.13.14、PyTorch 2.12.0+cu130、GCC/G++ 14.2.0、GLIBC 2.41；A100 80GB VRAM、Intel Xeon 至多 48 核；CUDA 和 RAM 差异见上表。这是**官方文件的记录**，不是本机现有环境或对这些版本可获取性的验证。

样本顺序评测，单样本内部可使用多线程/多进程（A2/A3）。优化器实例在测试循环前初始化，计时包围 `solve()`（A28）。评测阶段无联网。提交指南要求 `.tar.gz`，提供可执行程序及必要依赖/外部文件，源代码可作 fallback，不允许 Docker；Beta 报告进一步要求顶层 wrapper 与完整依赖清单，A27 说明非空 requirements 会建立 venv。

官方 wrapper 通过 stdin 发送 JSON，包含 `block_count`、面积、连接、端口、约束和 `target_positions`；二进制 stdout 应只返回 `{"positions": [[x,y,w,h], ...]}`，诊断日志写 stderr。具体文件搜索逻辑见[原始 wrapper](../third_party/official/op_wrapper.py)。

## 后续研究的最小起点

当前两套公开数据均已就位，独立 conda 环境中的原版 baseline 100 例评测已完成，0/100 可行、无执行异常。随后只抽查 5 例，通过参考布局、仅合法输入的可行性对照及独立硬约束检查，定位到模板 fixed/preplaced 处理缺失，见[核查报告](../experiments/baseline/diagnostics.md)。后续默认只用 5 例，先建立满足硬约束的独立适配基线，再比较质量与速度。未进行训练。

若进入机器学习路线，先确定训练/内部留出/公开验证的职责，明确可使用的输入和预算，避免读取公开验证标签后再报告“独立泛化”。若进入传统搜索路线，同样记录随机种子、停止条件和质量随时间的变化。题目禁止以逆向生成器替代真实算法；通用建模与统计分析需和利用生成器特例的方案区分。
