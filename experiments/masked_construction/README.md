# E3：带规则候选的逐步构造

状态：已完成无训练 dev5 贪心/软规则消融、一个共享 train 样本的候选覆盖诊断和一次模仿优化批次。**没有完成模仿训练，也没有学习策略的 dev 结果。暂缓一小时训练，先审视候选表示。**

## 假设与实现

在固定顺序和相同候选生成器下，几何算子负责可行候选，策略只负责选择，可减少连续布局修复的负担。普通块按原始面积提供宽高比 `0.5/1/2` 三种候选；这是有限搜索限制，不是题目规则。fixed 宽高锁定且不旋转，preplaced 先作为不可移动障碍放入。顺序按面积降序，平局按输入编号，始终不读取软规则或参考自由块信息。

`construction.py` 独立实现，无跨实验私有算法导入。位置来自已放矩形的四侧接触与端点对齐，加原点、右侧/顶部无界货架。两策略共享上述池生成规则及非负/无重叠掩码；策略选择不同后，后续状态与实际池自然不同，不声称整个 rollout 的候选坐标逐项相同。

`greedy` 代价是归一化 bbox、已放邻居/固定端口的线长、bbox 周长，加 boundary 距离、已放同组最近接触距离、MIB 对数尺寸差。`no_soft` 仅将后三项权重归零，形状池、顺序和硬掩码保持一致。软规则从不删除候选或改面积。grouping 偏好只是距离代理，不能保证多子团连通；报告另计算全局正长度接边图连通分量，并保留官方软违反。

没有连续优化阶段。原始草稿记录每步在最终已放前缀下的未掩码最优候选，连续修正为恒等映射，最终阶段施加逐步硬掩码。草稿不是一个独立的无约束 rollout。所有阶段使用官方评分及独立硬检查；五例草稿均不可行，不能当成最终质量提升证据。最终记录中心位移、质量变化、候选数、空池、兜底、接边连通和全局 bbox 接边违反。无界货架保证可继续放置，所谓未来死局数零只表示未发生无界硬死局；不表示没有堵住紧凑区域。

## 已运行证据

全部固定 seed=0、CPU 单线程，无 GPU；仅冻结五个 dev 布局和与 E1 共用的一个 train 布局，无 local_test 或公开验证读取。

| 策略 | 硬合法 | 全样本加权 Q ↓ | HPWL gap 均值 | bbox gap 均值 | boundary / grouping / MIB 均值 | 求解总秒数 | 兜底 |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 5/5 | 4.48475 | 0.63461 | 0.15223 | 20.0 / 8.2 / 0.4 | 1.84277 | 0 |
| no_soft | 5/5 | 6.72691 | 0.67782 | 0.15458 | 23.4 / 13.4 / 1.4 | 1.84535 | 0 |

greedy 在五例均优于 no_soft，支持在这一候选表示内保留软偏好。仍有明显接边/分组违反，不能解释为已解决软规则。历史 baseline search Q=1.9090 只供定位，资源与计时口径并未重新配平，**本结果没有胜过 baseline，也没有证明学习收益。** Q 为官方中性时间质量分，不是比赛成绩。原运行计时覆盖输入准备、候选与输出构造，未单独计入随后官方计分及独立检查耗时；不能据此作严格端到端效率结论。单变体设 7.5 秒内部预算、10 次总求解实测 3.68812 秒，低于本次 90 秒求解上限；内部超时采用货架，边界检查不是操作系统抢占。

- [dev5 逐例、阶段、唯一坐标和汇总](runs/20261016_dev5_greedy_no_soft_seed0/results.json)，[实际配置](runs/20261016_dev5_greedy_no_soft_seed0/config.json)。
- [五个直接反例](runs/20261016_counterexamples_seed0/results.json)：两个紧密但互不相连的子团、不同面积 MIB 不可能同形、接近端口不代表触最终 bbox 边、当前占住紧凑空隙后未来只能走货架，以及非方 fixed/浮动 preplaced 保持。
- [候选覆盖](runs/20261016_train1_coverage_seed0/results.json)：共同 train manifest 第 6516 项 `worker_94/layouts_224.th:20`，28 次可移动动作；参考前缀下精确覆盖 4/28=14.29%，`中心距离/√总面积 + 对数宽高距离` 平均 0.22782。诊断只读 train 标签；这是 teacher-forced 乐观覆盖，不是 rollout 成功率。粗形状池不能表示多数参考动作，是扩大模仿训练前的明确限制。
- [模仿一次批次](runs/20261016_train1_batch1_smoke_seed0/results.json)：同一个 train 样本，仅一次参数更新，资源停止，CPU 2.92 秒；随后已验证 checkpoint 可加载且前向输出有限。不是训练完成或收敛。唯一 checkpoint 在该运行的 `checkpoints/latest.pt`，被 Git 忽略。

开发运行完成后，`run.py` 仅补上可选已训练策略 CLI；原配置源哈希保留运行时事实，构造算法未变。模型、训练入口和数据清单已准备，训练覆盖不足时不自动扩大预算或改候选比例。

原四次运行开始时没有保存完整源码快照，dev5 原始 `run.py` 已修改，故该运行的完整原版入口缺失，不把当前文件冒充历史版本。[可取得的历史源码](available_historical_source.zip) 保存修复前的训练器、构造器和反例入口：train 覆盖/一次批次的组合源码哈希已与原配置逐字节核对，dev5 构造器也与原哈希一致；反例入口未改，但原运行未记其源码哈希。压缩包内清单分别披露这些证据范围。历史配置/结果保持原样。[当前最终源码](final_source.zip) 是后续可靠性修复后的交付版本，明确区别于历史运行。

交付前仅追加[合成训练器可靠性检查](runs/20261016_trainer_reliability_synthetic/results.json)：批次上限后的恢复不多更新、SIGINT 在完整动作边界保存、原子保存失败保留旧 checkpoint、未正常退出时预算继续计费，以及数据路径漂移拒绝恢复；同时检查全部源码语法，没有再读取真实 train/dev 张量。原先一次批次 checkpoint 是旧版本烟测证据，新版本严格恢复校验会拒绝它。

## 入口与用户运行命令

```powershell
conda activate E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
python -B experiments/masked_construction/run.py --output experiments/masked_construction/runs/<新运行名>
python -B experiments/masked_construction/counterexamples.py --output experiments/masked_construction/runs/<新反例名>/results.json
python -B experiments/masked_construction/train.py --mode coverage --member-offset 6516 --max-layouts 1 --output experiments/masked_construction/runs/<新覆盖名>
```

`train.py` 内的 `6→32→1` MLP 对硬合法动作打分；监督是参考前缀下上述距离最近的可表示动作，用候选间交叉熵模仿。不是精确参考动作可用时的上限。候选生成、特征、训练损失均在本目录，只有 manifest JSON 跨实验引用；训练启动先验证冻结 split 哈希、train 成员和不超过 2 万布局。

下列是完整首轮训练入口，**目前建议不启动**。共同成员 19,936 布局、一次顺序遍历，最多累计 3,600 秒或 100,000 动作批次；CPU 单线程为主，通常候选生成占用大，不预期小 MLP 从 GPU 获益。RAM 估计 <1 GB 加官方单分片，checkpoint 小于 1 MB；不预处理全量缓存。实际停止可能因批次数/时间先耗尽，并不保证遍历完或收敛。启动开销和当前单步可使墙钟略超过预算检查点。

```powershell
python -B experiments/masked_construction/train.py --mode train --max-layouts 19936 --max-seconds 3600 --max-batches 100000 --device cpu --output experiments/masked_construction/runs/20261016_imitation_train_seed0
```

每个布局结束及每 30 秒保存，checkpoint 通过同目录临时文件原子替换；Ctrl+C 只设置停止标志，在当前动作完整结束后保存，再退出。以**同一完整命令追加 `--resume`** 恢复，累计步数/秒数不清零；源码、清单、成员、设备、数据路径、Python/PyTorch 和预算必须一致。每次动作更新前写入批次预算账本，因此从较旧 checkpoint 重放丢失动作也会再次消耗预算，不退回原已用批次。若进程被强杀导致活动计时租约未正常关闭，恢复保守计入从原启动到恢复时的墙钟（包括停机等待），最多计至原时间上限；可能直接耗尽本轮预算，不能借崩溃增加训练额度。正常 Ctrl+C 不计等待时间。若已累计达到资源上限，恢复不会追加一小时。一次批次烟测 checkpoint 不能当作上述完整成员训练的恢复源。

训练后可显式比较，仍限冻结 dev5 且累计求解上限 90 秒：

```powershell
python -B experiments/masked_construction/run.py --strategies greedy no_soft imitation --seconds 5 --checkpoint experiments/masked_construction/runs/20261016_imitation_train_seed0/checkpoints/latest.pt --output experiments/masked_construction/runs/<新学习对照名>
```

只有学习策略在同池规则、总时间和输入边界下优于贪心，才扩大下一笔预算。当前结论是软特征有信号，有限候选的参考动作覆盖不足，学习选择的价值尚未检验。
