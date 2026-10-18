# E2：分离关系 → 固定形状几何

状态：**最小关系模型、训练入口和几何闭环已实现；无训练几何探针已运行，学习优劣尚未验证。当前几何控制没有质量信号，暂停本轮训练投入。**

## 假设与固定比较

仅用连接图、面积、端口、约束和允许公开的 fixed/preplaced 规格预测块对方向，可能比直接预测坐标更容易获得合法布局。首轮固定普通块为面积完全保持的正方形，fixed/preplaced 原始宽高锁定；不输入普通块参考形状。启发式、seed=0 随机、学习关系使用相同尺寸、关系处理顺序规则、几何目标、兜底和预算。学习模型未训练时不以随机初始化模型冒充学习组。

`input_geometry.py` 的输入校验和连接图锚点处理于 2026-10-16 从 `baseline/packing.py` 复制后独立维护；没有跨实验导入私有算法。仅复用 `datasets/`、`scoring/`。关系方向为 i 左 j、j 左 i、i 下 j、j 下 i。

## 几何与训练实现

- `geometry.py` 固定保面积尺寸，按置信度依次插入分离关系。每轴最大加法闭包维护差分约束，拒绝方向环及锚点不相容路径，尝试下一方向。origin 与锚点的双向约束严格保持位置；这是已定尺寸与分离关系的约束图求解，不依赖 SciPy，不声称整个可变形问题已凸化。坐标为满足已接受关系的分量最小非负解，相当于最小化坐标和，尚未加入线长或软规则目标。
- 所有块对插入完成后独立检查全图。关系死局、预算用完或最终检查失败时，用相同尺寸在预放置障碍上方分行兜底。fixed/preplaced 不受软接边或 grouping 要求驱动而改动。当前软规则仅由官方评分报告，未强行作为几何硬约束。
- `model.py` 是 64 维、两轮图消息传递编码器及四分类块对头。标签允许同时成立的多个分离方向，Loss 为可接受方向集合的边际负对数概率，不把任意标签方向当唯一真相。标签仅在 train 构造监督目标；推理特征只来自上述合法输入。
- `train.py` 仅读取 E1 公共训练成员 JSON，逐成员核验冻结 train 归属、重复和行号，最多 20,000 布局。单布局 batch，AdamW 固定配置，无架构/超参搜索；每 batch 查累计预算，每 100 步和正常/中断结束保存一个 `latest.pt`，不自动续跑。不触及 local_test。

原始阶段是输入网络锚点提议；连续修正阶段是关系图坐标；最终阶段是全图检查后的输出或兜底。`run.py` 使用官方中性时间评分和独立硬检查分别记录三阶段，最终坐标仅保存一次。`modified_relation_fraction` 统计已处理块对中改用次选方向的次数除以总块对数；**关系处理中途终止时，它不是最终关系保留率**。凡发生全局兜底，原拓扑的几何贡献视为被抹去，不能用较低的该字段掩盖。

## 已有运行与判断

证据：[20261016_geometry_dev5_seed0/results.json](runs/20261016_geometry_dev5_seed0/results.json)、[实际配置](runs/20261016_geometry_dev5_seed0/config.json)；该目录保存可恢复的源码快照 `source.zip`。

运行后仅加固训练恢复与停止逻辑：恢复先核验身份，再接触运行目录；Ctrl+C 标记停止、在当前 batch 完成后保存。当前 `train.py` 因此与该历史快照不同；几何/评分实现未变，历史证据和快照保持原样，未重复运行 dev5。

冻结 dev5、CPU 单线程、seed=0、每例每组上限 15 秒、总求解预算 90 秒。实测十次求解合计 0.605 秒；这是直接图算法的短探针，未使用完整时间做搜索，不与历史 15 秒搜索声称匹配质量胜出。

| 控制 | 最终硬合法 | 中性时间加权 Q ↓ | 平均 HPWL gap | 平均 bbox gap | boundary / grouping / MIB 平均违反 | 全局兜底 | 求解合计秒 |
|---|---:|---:|---:|---:|---|---:|---:|
| 输入启发式方向 | 5/5 | 9.999999 | 1.9818 | 1.3000 | 23.6 / 18.0 / 0.4 | 4/5 | 0.315 |
| 随机方向 | 5/5 | 9.999999 | 7.6616 | 17.8974 | 25.2 / 18.2 / 0.4 | 3/5 | 0.290 |
| 学习方向 | 未运行 | — | — | — | — | — | — |

启发式合计拒绝 114 次锚点冲突；随机合计拒绝 7,440 次方向环、803 次锚点冲突。两组所有 Q 达到官方可行解质量上限附近，不能凭 Q 区分贡献；原始 HPWL/bbox 与兜底率说明随机关系引入冗长路径、当前启发式也频繁走入无法扩展的关系死局。hard 合法由兜底保障，尚不是关系表示带来优质几何的证据。未将失败归因为学习模型，因为尚未训练。

针对性反例已通过：三块方向环被拒绝；宽 3 自由块被宽 2 的锚点间隙夹住时拒绝该关系；fixed 非正方形规格和预放置坐标保持；矛盾左右接边软标记不改变锁定规格；向所有禁止参考字段填入极端数值不改变求解；多方向监督的合成 backward 梯度有限。所有模型 backward 检查使用合成三块标签，**未读取额外 train 样本**。

本轮判断：当前约束图实现完成合法性与信息边界验证，但大量兜底和质量饱和会抹平学习关系的价值，不满足晋级条件。先暂停真实训练；若后续继续，需要另列明确机制变化，例如约束插入回退而非整图重启，或能表达接边/连通性的结构。不得在此实现上直接扩大模型或反复调参挽救。

## 可复现入口

```powershell
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
$env:PYTHONDONTWRITEBYTECODE = '1'
python -B -m experiments.topology_geometry.run --seconds 15 --total-seconds 90 --limit 5 --seed 0 --output experiments/topology_geometry/runs/20261016_geometry_dev5_seed0_r2
```

独立合成检查可增加 `--counterexamples-only`，使用新的输出目录。不得覆盖已有运行。

以下训练命令为用户执行入口，**本轮没有启动，也不建议在当前几何探针失败后直接投入这笔预算**。成员索引已由 E1 建立，E2 不再创建不同子集；同一 `train_members.json` 是唯一跨实验数据依赖，不导入 E1 模型或算法。上限 1 GPU 小时，实际可能因 20,000 步先结束；未测实长、不称收敛。预计显存低于 2 GB、checkpoint/配置/源码和日志小于 50 MB，依硬件而变。

```powershell
python -B -m experiments.topology_geometry.train --members experiments/graph_proposal/train_members.json --device cuda --seconds 3600 --max-steps 20000 --seed 0 --output experiments/topology_geometry/runs/20261016_relations_train_seed0
```

Ctrl+C 只设停止标记，在当前 batch 的优化器完整更新后写 `latest.pt` 和 `status.json`；运行异常保留此前完整 checkpoint，不保存部分优化器更新。恢复需明确执行同命令并增加 `--resume experiments/topology_geometry/runs/20261016_relations_train_seed0/latest.pt`；累计预算和步数延续，成员、seed 或源码改变时在写文件前拒绝恢复，原始源码快照不覆盖。若源码更改，保留旧目录另建新运行，先审阅差异。

用户完成训练后可在相同 dev5 与预算上加入学习关系对照：

```powershell
python -B -m experiments.topology_geometry.run --checkpoint experiments/topology_geometry/runs/20261016_relations_train_seed0/latest.pt --seconds 15 --total-seconds 90 --limit 5 --seed 0 --output experiments/topology_geometry/runs/20261016_learned_dev5_seed0
```

这里的 Q 是官方 runtime=median_runtime=1 的本地质量对照；实测求解时间另列，不是正式比赛分数。五例只支持本轮机制排查，不支持泛化或路线永久无效的结论。
