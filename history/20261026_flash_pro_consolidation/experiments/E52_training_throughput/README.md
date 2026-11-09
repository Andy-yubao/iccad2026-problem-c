# E52：相同模型与统一Loss的训练吞吐优化

2026-10-25，用户要求优先解决训练慢、加快模型研究迭代。本实验独立复制E51 unified_v2，**只优化训练执行路径，不改变模型结构、参数量、Loss定义、数据成员、batch32或学习率/早停规则**。没有启动正式训练，不恢复已完成E51。来源见[provenance.json](provenance.json)。

## 已测结论

E51模型为96宽、12层、1,471,539参数，设备为RTX 5080 Laptop 16GB / PyTorch2.7.1+cu128。模型本身不大，但12轮矩形反馈与两两几何操作产生大量小算子。原实现的一步CPU剖析记录到5,940次CUDA内核启动；这不是GPU计算时间占比，也不等于训练时的平均GPU利用率。

E51总训练墙钟约9,690秒；第30轮监测间隔约3,465秒，其他轮通常100～150秒，监测＋边界保存通常约5秒。存在约55分钟异常长间隔，可能是系统挂起、睡眠或资源长阻塞，日志不能确定原因；不能全部归因于模型参数或算子速度。旧时间与预算不改写。

后续用户已明确确认上述异常属于硬件问题且已解决；后续按正常epoch和E52实测估计速度，不再把异常间隔作为算法瓶颈。加速方法已写入[执行协议](../../docs/protocol.md)。

在同一五个train样本（25/45/70/95/115模块）重复组成batch32的短基准中，原路径与最终优化路径均为FP32和原AdamW、无AMP、确定性算法开启：

| 项目 | 中位耗时 |
|---|---:|
| 原CPU拼batch＋搬运＋训练一步 | 0.171259秒 |
| 数据在GPU后仍普通执行一步 | 0.165621秒 |
| 原CPU拼batch＋搬运 | 0.007971秒 |
| GPU常驻数据选取batch | 0.000550秒 |
| CUDA graph＋GPU常驻数据训练一步 | **0.067089秒** |

最终单步约**2.55倍**；不把它宣称为完整训练加速倍数。基准使用lr=0测量实际AdamW状态开销，不改变模型权重；完整训练尚未运行。初步按500步/轮加固定监测估算约40～50秒/轮；若仍52轮，约35～50分钟，加启动/打包开销。该估计尚未验证，受模块分布、显卡功耗及系统负载影响。

证据：[最终短基准](runs/20261025_profile5/results.json)、[数值与恢复核验](runs/20261025_profile5/verification.json)、[实际训练入口保存/恢复检查](runs/20261025_profile5/runtime_smoke.json)。实际检查只有同一五个train样本，无测试读取。临时重复样本与实现检查权重不作为模型训练成果或验证结果。

## 实现改动

1. [resident.py](resident.py)一次将训练/开发的现成图张量打包到GPU，统一按公开最大模块数padding。样本、shuffle顺序、监督与public-only输入隔离不变；固定dev/train监测仍各500，每轮一次。完整train与dev_monitor500预计常驻约3GiB，实际以启动日志为准；短基准峰值显存约2.15GiB，不含完整数据打包。保留普通模式，不因显存不足静默切模式。
2. [cuda_step.py](cuda_step.py)将FP32模型前向、同一总Loss、反向与梯度裁剪一次捕获后重放，减少每步Python/CUDA调度。每步检查非有限Loss/梯度后再执行原AdamW，保持batch32、clip1、decay3e-4、Dropout .05。预热/捕获不更新权重或优化器，恢复原CUDA随机状态；checkpoint重建图后从已保存状态继续。
3. 缓存Loss中的固定贴边位常量，避免每步从CPU创建GPU常量。两轴max用等价分支保持原第一轴平局规则，避开确定性scatter的同步路径。PyTorch2.7.1通用prod反向含`.item()`取零计数，不能捕获；[ops.py](ops.py)把相同的全局零分支选择留在GPU，并保留原有乘除运算顺序。训练、监测和未来获授权测试共同调用[loss.py](loss.py)，没有第二套选择标准。
4. 监测的标量结果合并传回CPU；保留独立硬约束/几何检查。删除每轮额外七次分项参数梯度诊断，优化总梯度仍完整计算。预算心跳改为每5秒和保存时写入，定期latest最多每30秒一次；epoch边界、best、停止和异常仍立即保存，累计时间不重置。

旧Loss独立副本`reference_loss.py/reference_losses.py/reference_module_loss.py`仅用于工程核验，不参与正式训练选模。架构、优化器数值和随机恢复均通过短检查；不据此声称完整训练轨迹逐位一致或质量一定等同。统一120模块padding的Loss最大差约2.98e-8；同形状下Loss、带Dropout的一步更新及中断重建后下一步均在本次检查中差为0。此前简单面积乘法改写出现约1.29e-5的一步参数差，已弃用，最终保留原PyTorch反向算术，失败探测记录留在profile5下。

## 用户启动

只运行一组seed0、从头训练，默认CUDA graph，不自动启动后处理或评分界面：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E52_training_throughput\launch.ps1' -Stage train -Resume
```

主输出`runs/20261025_rectangle_w96d12_cuda_graph_main_seed0/`；协调日志为同名`_pipeline/console.log`。首次从头，后续相同命令只恢复同一运行；完成后不追加训练。环境沿用`.conda/envs/iccad2026`，单GPU、CPU四线程；无需安装新依赖。训练仍最多4小时/200轮，LR为1e-3→2.5e-4→1e-4→5e-5，第15轮后patience3，最低档至少完整两轮，best与调度均看统一dev500总Loss。本轮没有混入另一个学习率候选。

仅预览：

```powershell
& 'E:\project\ICCAD\2026\problemC\experiments\E52_training_throughput\launch.ps1' -Stage train -Resume -DryRun
```

`-Engine resident_eager`使用GPU常驻数据但不捕获图，`-Engine reference`使用普通拼batch路径，三种引擎使用独立输出目录，不在原目录静默切换或续接。reference是执行模式名称，仍调用同一固定Loss。GPU图模式不支持任意更换batch/模型/Loss后继续旧断点；发生身份不符即停止。

Ctrl+C安全保存，或创建`experiments/E52_training_throughput/runs/STOP`。确认停止后移除自己创建的STOP，再运行同一Resume命令。异常/正常停止保留完整latest，突然终止最多回退到最近约30秒或epoch边界；不重置已登记累计预算。旧E51 checkpoint保留，不复制为E52的初始训练权重。

训练后若需要正式开发后处理，另行运行`-Stage final -Resume`；不自动启动评分服务。默认视觉对照应按用户要求采用reference／当前draft／已接受最优E42 draft，后处理Q只作辅助；不能将复制的旧评分准备脚本当成启动评分的授权。测试继续封存。

## 解释边界与后续

本轮解决执行开销，**没有声称E51 Loss已解决局部冲突问题**。新的Loss研究可从本实验独立复制快速执行路径，在相同预算下更快做对照。先用一次完整运行确认每轮时间和训练行为，再评价新模型效果；不能凭微基准把精度、收敛轮数或最终Q的收益也算进来。

必要复核入口：`python -m experiments.E52_training_throughput.check`只使用五个现有train样本验证Loss、padding、Dropout更新与恢复；`profile`是限时微基准，不是训练。完整源码身份、数据缓存身份和checkpoint恢复规则继续记录；模型、Loss和算法未跨实验私有导入。
