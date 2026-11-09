2026-10-26：当前保留基线为 **flash 与两条基线共用 draft**。共用 E42 best51 权重原位保存；旧实验及共享缓存的迁移见[整理记录](../../history/20261026_flash_pro_consolidation/README.md)。历史命令与源码身份保持原样，不绕过冻结检查续跑。默认不启动评分页面。

# E42：12层矩形注意力的联合正则化AB

## 2026-10-24：“整齐且易微调合法”四项顺序检验

按用户明确授权完成前三项计算。第四项5组A/B被用户反馈为布局均不理想、差异太小，未取得有效比较信息，不要求继续补评；原图、原评分及准备事实保留。[完整报告](runs/20261024_orderliness_four_checks/analysis.md)与[事前计划/执行补充](runs/20261024_orderliness_four_checks/plan.md)保留失败、未知与来源。原模型、Loss、常规后处理及22张评分不改，未读取未暴露测试。

小预算完整合法化探测未得到合格见证，不能据此声称高分确实“只需很小移动”。成对必要下界揭示预放置/非负边界可让微小重叠需要改形；自然样本有4组不重复的相近重叠量跨评分配对，现有复杂团能量仍不足以解释全部整齐感。5个按模块数选定的开发样本中，最后一层均继续降低重叠；参数空间的真实总几何梯度在3/5例局部与重叠项相反，是目标折中的线索，不是训练因果证明。冻结权重未更新。

第四项独立前端：`http://127.0.0.1:8767/`，1=A更整齐/2=差不多/3=B更整齐，可选可修复性和留白偏好，保存后自动下一组。图像/来源/左右顺序冻结于本机`runs/20261024_orderliness_four_checks/comparison_review/manifest.json`，原评分会话保持原样。新会话0/5只是准备状态，不能称四项已全部验证完成。

重开第四项服务：

```powershell
& 'E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026\python.exe' -m visualization.review --session 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\runs\20261024_orderliness_four_checks\comparison_review' --port 8767
```

## 2026-10-24：评分后的堆积与无效留白诊断

用户表示本轮评分到此为止；实际保存22张评分（1/2/3分为4/9/9张）及另1张仅标签，原数据库保持这个事实。按用户解释量化“局部复杂重叠”和“极小代价可消除的留白”，见[指标定义](feedback_metrics.md)及[23例分析](runs/20261024_feedback_geometry23/analysis.md)。

覆盖负荷平方均值与22张分数的秩相关−0.881，普通最大连通块占比仅−0.285；强重叠图用于定位局部严重团。留白采用有位移/形状/线长/逐对重叠/软约束保护的收缩证据曲线，区分包围盒收缩与解重叠填空。标记留白的case_0063找到平均移动自身等效边长1.103%、小幅改形，消除原留白8.559%、线长下降0.203%的具体证据。该探测结果仅是可消除留白的收益下界，只有1张留白标签，不能称该指标已验证或证明其余留白不可消除。原模型、Loss、常规后处理及正式最终结果保持原样，无训练/推理。

## 2026-10-24：首轮人工草稿评分

用户确认：只评中间draft，1差/2一般/3好；从**已暴露的同1000例测试结果**中均匀无放回随机抽100例，随机顺序逐张展示。不按Q选图。此轮作为后续模型研究的自适应反馈，不再宣称这100例对于后续调整是独立未见测试；其余900例继续封存。研究参考次序与模糊反馈核验原则见[协议8.1](../../docs/protocol.md#81-草稿模型的人类反馈研究原则2026-10-24)。本次只建评分入口，不训练或改变模型、Loss、后处理。

实验入口[prepare_review.py](prepare_review.py)持有抽样、公开输入读取与冻结best51草稿补生成；最终布局、Q及原计时直接引用原1000例逐例文件，不重跑后处理或官方评分。原参考列是MIB平均面积变换之前的几何，不保证满足派生任务面积。三相图统一调用`visualization.layouts.plot_case`，前端和持久保存共用`visualization.review`。

首轮目录[runs/20261024_visual_review100_seed20261024](runs/20261024_visual_review100_seed20261024/)：`config.json`冻结候选池身份、种子、100例成员及顺序，`manifest.json`绑定图像与模型/源结果身份，`drafts_local/`只存新增草稿，`images/`保存100张三相图，`ratings.sqlite3`保留评分、备注、整轮评价和改分历史。原最终坐标不复制。**图片准备完成不代表用户已评分完成**，实际评分状态以数据库/导出记录为准。

一键准备并启动（已生成则校验复用，不重复推理）：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\review.ps1'
```

打开`http://127.0.0.1:8765/`即可开始。1/2/3快捷键打分，左右键切换；也可单击按钮。默认保存后自动下一张，可取消。支持原参考/草稿/后处理单栏放大、逐例观察标签、整轮评价和导出。评分实时保存在本机，关闭后可用同一命令继续。端口被占用可加`-Port 8766`。本次生成采用4进程各1线程，无训练；预计数分钟，累计推理/绘图预算20分钟。

本轮已生成100/100张图，实际准备耗时38.52秒；[准备汇总](runs/20261024_visual_review100_seed20261024/preparation_summary.json)及[功能核验](runs/20261024_visual_review100_seed20261024/verification.json)保留证据。100张PNG均通过读取检查；三例临时副本验证自动下一张、快捷键、刷新恢复、备注/新维度、整轮评价、改分持久保存和导出，正式轮次没有写入测试评分。临时数据已清理；随后的真实评分按用户要求结束于22张分数＋1仅标签，见上方诊断。数据库保留未完成100例的实际状态，不强制标成100例完成。

## 2026-10-24：常规后处理同1000例最终比较

按用户明确要求，冻结best51及原常规steady，在E25刚完成的**完全相同1000例、相同顺序、相同4进程×1线程与计时口径**下评价一次。未使用E47长预算、辅助关系头或参考信息，未重跑E25。原E42训练/模型/求解代码及权重SHA全部核验不变。

**1000/1000硬合法，加权Q1.028385030，平均完整求解0.222522029秒/例**；执行/评分错误均为0。相对E25的Q1.045269650、0.231350916秒，均值点估计同时改善，原Q<1.04、均时<0.25秒联合目标通过。

逐例Q胜/平/负为694/77/229，超额Q(Q−1)下降37.30%、均时下降3.82%。worker聚类配对bootstrap95%的加权Q差为[-0.029110,-0.007885]，时间差为[-0.023232,+0.005216]秒：质量改善有支持，耗时小幅优势尚不稳。P95 Q1.194060→1.145642、最差5%均值1.350253→1.304260，但最大Q2.832076→3.252251；不宣称所有尾部或每例均改善。

[最终报告及配对比较](runs/20261024_final_test1000_seed20261024/analysis.md) · [精确汇总](runs/20261024_final_test1000_seed20261024/summary.json) · [比较JSON](runs/20261024_final_test1000_seed20261024/comparison.json) · [逐例指标CSV](runs/20261024_final_test1000_seed20261024/paired_metrics.csv) · [验收](runs/20261024_final_test1000_seed20261024/audit.json) · [评价入口](final_test.py)。完整坐标各自在原运行保留一份，仅存本机；离线重建配对报告运行本次目录analyze_results.py，不重新推理或评分。

这是MIB平均面积派生任务、沿用原参考分母的本地中性时间Q，不是正式比赛成绩。E42权重在两次测试之前已冻结；本次是用户看到E25结果后发起的既有候选比较，面板已暴露，不称新的独立盲测。剩余900例继续封存。完整计时含公开图特征、CPU推理、解码与全部后处理/救援，加载、JIT初始化、评分及写盘另列。

当前结果呈现采用E42＋常规后处理，E25保留历史参照。以下2026-10-23段落保留当时开发阶段的选择事实。

2026-10-23，用户已自行完成唯一联合AB实验：54轮早停、best51，完整开发代理0.453814较未正则化12层改善3.04%，训练—开发差距缩小34.4%；终末500全硬合法、加权Q1.025860，但P95/最大Q变差。[完成分析与泛化曲线](runs/20261023_rectangle_w96d12_ab_seed0/analysis.md)已保存。本轮只读检查，无续训或重评，旧E39/E41源码与权重保持冻结，旧队列及定时任务不恢复。下文保留原唯一配置和命令。

## 唯一配置

| 项目 | 配置 |
|---|---|
| 主干 | E39 rectangle 96×12，seed0，从头初始化，1,471,539参数 |
| A | 每层FFN的SiLU后、第二个Linear前，Dropout p=0.05；只对隐藏特征作用 |
| B | AdamW weight_decay从1e-4增至3e-4，参数分组方式沿用原控制 |
| 数据增强 | 无；不旋转、镜像、交换坐标轴或改变原始高宽分布 |
| 数据与监督 | 冻结v5 train16000/dev2000，train/dev_monitor各500，MIB平均面积处理及原监督不变 |
| 训练Loss/选模 | 全部保持E39原样，MIB系数0.1；固定dev_monitor500 geometry＋occupancy选best |
| 参照 | 只读复用E39未正则化矩形12层seed0 best49，不重训控制、不重复评价E25 |

假设：联合正则化降低对训练集局部特征组合的依赖，使开发接触/中心误差和整体代理改善。训练—开发差距缩小但开发质量不升，可能只是欠拟合，不算成功。该实验衡量AB组合的效果，不归因到某个单独技巧；不额外安排A-only/B-only、EMA、增强、Loss调整或其他种子。

Dropout为独立无参数模块，保留原`ff.0`/`ff.2`参数键和初始化随机数消耗。因此在相同seed初始化时可与E39逐参数核对；`eval()`完全关闭Dropout，训练/验证监测均在eval模式。不会丢弃硬规格、约束、连接或坐标，也不会增加模型参数量。配置`ffn_dropout`写入checkpoint的model_config，weight_decay写入恢复身份并核对实际优化器，禁止恢复时静默改值。

## 用户启动命令

PowerShell可以从任意当前目录运行，脚本自行切换工作区并使用独立conda环境：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\launch.ps1' -Stage all -Resume
```

只执行这一组AB训练，然后冻结唯一best、完整dev2000原始评价、一次固定开发500例后处理评价、只读生成与E39的配对报告，结束。训练/监测不调用后处理器；CPU正式计时与GPU训练分时。首次`-Resume`没有latest时从头初始化，已有latest时严格恢复，已完成时复用结果。

只训练、不接续评价：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\launch.ps1' -Stage train -Resume
```

训练完成后单独执行评价：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\launch.ps1' -Stage final -Resume
```

加`-DryRun`仅打印命令，不训练、不评价、不创建正式模型/协调目录。

## 预算和恢复

- 单GPU CUDA，batch32；CPU总计四线程，终末2进程×2线程。
- 累计训练最多4小时或200轮；启动/缓存载入/固定监测计入训练墙钟，终末另计。预计预留2～4小时；原未正则化12层约86分钟，但正则化后的停止轮数与速度未实测。
- LR固定`1e-3 → 2.5e-4 → 1e-4 → 5e-5`；第15轮起连续3轮无明显改善降档，3次降档且最低档至少完成两轮后才能早停。沿用原改善阈值、梯度裁剪1、物理项前两轮渐入、优化器及随机状态恢复。
- best负责选模，latest保存优化器/调度器/CPU及CUDA RNG/样本排列与游标/累计预算。Dropout消耗的随机状态也一并恢复；不得从头覆盖已有输出或清零预算。
- 显存/非有限值等异常时保留错误和checkpoint，不自动减batch、切换设备或关闭确定性。

按Ctrl+C请求安全停止，或在另一个PowerShell执行：

```powershell
New-Item -ItemType File -Force -Path 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\runs\STOP' | Out-Null
```

当前batch/病例边界保存后退出。确认退出并决定恢复时：

```powershell
Remove-Item -LiteralPath 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\runs\STOP' -ErrorAction SilentlyContinue
& 'E:\project\ICCAD\2026\problemC\experiments\E42_joint_regularization\launch.ps1' -Stage all -Resume
```

自行创建的模型/协调目录STOP也须在明确恢复时移除；预算停止的完成状态不能通过重启增加4小时。

## 输出与判断

- 模型目录：`runs/20261023_rectangle_w96d12_ab_seed0/`；协调日志：`runs/20261023_rectangle_w96d12_ab_pipeline_seed0/console.log`。
- 模型目录保存config、best/latest、每轮固定训练/开发曲线、full_dev和唯一terminal500。Dropout关闭后的train_monitor与dev_monitor保持相同评价模式。
- `comparison.json`/`comparison.md`与未正则化12层配对，保留预放置占比、E39重叠最高四分位、旧投影25例、旧救援6例及高group12例诊断。子集从旧证据固定，不按新结果选成员。
- 首先看相同面板的开发代理/geometry/occupancy、接触/中心/碰撞和完整dev确认；训练差距缩小不是充分条件。终末报告硬合法、Q、尾部、位移、投影/救援和完整计时，不按终末Q反选checkpoint。
- 当前E39参照：monitor500代理0.471997，full-dev2000代理0.468025；两个范围分开报告。单种子开发结果与一次限时修复不构成独立泛化或正式比赛分数结论，不设固定Q/耗时通过线。

测试封存，不新增教师，不改变v5成员、MIB面积变换、Loss、后处理或E25/baseline。

## 来源与准备检查

独立复制E41已验证的训练/恢复/评价入口及其E39数值实现，重设为12层并加入AB。[provenance.json](provenance.json)记录每个来源和修改后哈希；[model.py](model.py)、[train.py](train.py)、[arms.py](arms.py)给出实际配置，[pipeline.py](pipeline.py)只列一个训练任务。

[准备检查结果](runs/20261023_preparation/results.json)仅使用原来两例train和一例dev：CPU初始化/参数键一致、旧权重严格加载及eval逐元素一致、12个FFN Dropout启停、训练前向随机性及RNG恢复、实际AdamW系数、Loss/缓存身份不变。准备阶段没有训练更新、CUDA执行或求解器调用，检查当时正式运行目录尚未创建。随后用户已完成正式CUDA训练/评价，见顶部完成分析；该正式运行未发生中断恢复，不额外宣称已验证CUDA Dropout断点恢复精确性。
