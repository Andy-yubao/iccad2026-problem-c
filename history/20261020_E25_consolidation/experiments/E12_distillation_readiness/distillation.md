# 蒸馏接口与启动边界

## 本轮决定

固定的两组开发100例在下界面积表示、真实结构证书、一次局部SOCP及一次数值LP精修下，零误差和2%几何误差组均通过原定质量/合法性/时间门槛。证据见[runs/20261018_readiness_report/analysis.md](runs/20261018_readiness_report/analysis.md)。这足以进入有停止条件的首次蒸馏实验；尚未证明学生能学到所需精度，更没有完成0.5秒真实模型端到端目标。

旧模型相对教师的中心误差每例RMSE中位数约0.333个块尺度、log长宽比误差约0.503；两者离2%独立噪声很远。不能用“扩大参数就会成功”解释本轮结果。首次训练必须直接测模型误差形态、结构预测与最终Q，不能只看监督loss。

## 教师标签 v1

生成器[bank.py](bank.py)只允许固定train或仅评价用dev_monitor，不接受local_test。训练pilot是train_monitor内每个worker按冻结顺序取第一例，100例均衡，不依赖成绩挑样。冻结E10权重与独立复制的E10算法生成成品，求解接口只接收公开遮罩输入；官方参考仅用于评分，不保存为自由块训练目标。

pilot已完成：100例均硬合法、0标签错误；3例教师触发救援，其中2例Q>2，全部保留。固定5例标签往返与完成态恢复校验通过，详见[准备决定](runs/20261018_readiness_report/decision.md)。这100例用于验证标签生产，不足以支撑正式模型训练。

每例压缩标签包含：

| 字段 | 作用 |
|---|---|
| geometry | float32归一化中心x/y、log长宽比；公开已知通道由geometry_learned_mask屏蔽 |
| positions | float64教师成品，用于来源审计与解码核验 |
| separation | 每对模块四种有向分离关系，主对角线忽略 |
| group_contact_positive | 分组内全部实际相接边，供接触预测监督，避免把任选的一棵树当唯一正确答案 |
| contact_forest | 从实际接触中确定性抽取的生成森林，供当前oracle修复使用 |
| boundary_satisfied | 教师实际满足的公开边界位，未要求的位不监督 |
| mib_same | 仅在公开同MIB组内监督同形状关系；类别编号本身没有跨样例语义 |
| group_same_component | 教师实际分组连通分量，仅作结构诊断 |
| teacher_area_ratio | 面积审计；首轮解码不增加自由面积预测头 |

几何解码[targets.py](targets.py)以公开面积的0.9900002倍构造自由宽高，恢复fixed/preplaced规格，按预测MIB类别协调形状。该下界满足原规则；它是经过对照选出的表示选择，未修改官方面积容差。学生部署时所有结构标签必须由自己的输出或公开输入解出，不能使用教师成品。

## 首次蒸馏应回答的问题

建议在现有attention_geometry初始化上增加结构预测，而不是先堆参数。中心loss按sqrt(公开面积)归一化，log长宽比单独监测，使误差单位与本轮噪声实验一致。结构头分别学习边界位、组内接触和MIB同类；分离方向先从预测几何推导，再用明确的消融判断是否需要独立预测头。不要把稀疏接触的全矩阵准确率当成功指标，须报告召回、精度及预测关系冲突。

固定20,000训练/500开发监测，沿用学习率下降、早停、best/latest与累计预算。至少保留几何蒸馏、几何+结构蒸馏的同预算对照；重叠与留白先作监测及反例分析，不自动恢复已停止的occupancy训练路线。训练后分开检验真实预测、真实几何+教师结构、教师几何+真实结构，定位瓶颈。

开始训练的含义是测可学习性。结构有误差时，当前硬关系可能冲突；首次学生评估必须统计该问题、行排回退和尾部Q。2%高斯误差通过不证明相关误差、错误接触树或错误MIB分类也能通过。5%几何误差已有2/100回退，给出了当前修复的能力边界。最终成功仍需学生自身输入、原硬检查、保持Q和真实平均总耗时≤0.5秒共同成立。

## 可恢复全量教师生成命令

以下长任务未自动启动。在仓库根目录执行；新目录拒绝覆盖。完整20,000例通常需数小时，12小时是累计保护上限，不是必须跑满的训练目标。

```powershell
& .\.conda\envs\iccad2026\python.exe -B -u -m experiments.E12_distillation_readiness.bank --split train --max-seconds 43200 --output experiments/E12_distillation_readiness/runs/teacher_train20000_v1
```

固定500例开发监测标签只用于评价，不进入训练梯度：

```powershell
& .\.conda\envs\iccad2026\python.exe -B -u -m experiments.E12_distillation_readiness.bank --split dev_monitor --max-seconds 3600 --output experiments/E12_distillation_readiness/runs/teacher_devmonitor500_v1
```

新建对应输出目录下的`STOP`文件会在当前样例完成后停止；移除该文件，用原命令追加`--resume`恢复。恢复必须沿用相同代码、权重、成员、设备和累计时间上限，已有标签逐个校验哈希并跳过。失败样例记录error并保留，不自动改策略重试或删样例。若时间已耗尽，原预算不会因恢复而重置。各例npz标签留在本地labels/且不入Git，config、摘要和源码快照保留来源。

尚未编写/启动新的训练器；当前可直接执行的是教师标签生成入口。首次蒸馏的具体训练预算应在训练器实现前固定，不能把本轮1100秒诊断预算当长训练授权。
