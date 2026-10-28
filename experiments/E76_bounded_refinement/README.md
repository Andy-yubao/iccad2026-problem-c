# E76：冻结E74主干，只学习有界局部修正

2026-10-29，继续用户授权的组合模型研究。E74候选完整保留；E75在3374步触发保护，当前不恢复其失稳权重。E76保留E75的同一固定物理Loss，改用冻结的E74完整主干与新局部修正器，把“保护每一步参数”改为“直接约束模型功能变化”。

## 失败证据

[同一被拒32例训练批次诊断](runs/failure_diagnosis/results.json)显示，源E74在该批次eval平均Loss0.380677、隐藏最大99.85；E75失稳前权重平均Loss16.9744、隐藏最大352.32。最严重的新增训练例17039达到Loss392.51、旧G355.38、contact684.59。E75的图聚合参数确实没变，但head/attention/FFN等仍累计漂移约40%～70%。恢复部分组件能降低异常，却没有单个干预恢复全部行为；不是“固定图参数即可保证稳定”。

诊断的梯度范数是在eval下对所有参数计算，包含实际训练中冻结的图参数，**不等于原训练预裁剪梯度范数**。保存RNG已在被拒Dropout重放之后，诊断不是精确原掩码回放。旧诊断环境源码快照保存在同目录，未运行任何求解器或封存测试。

## 架构与不变量

完整E74 best42的192×12主干全程冻结，并在训练时保持eval模式；源Dropout不再产生训练噪声。基于其预测矩形和相同公开输入，新修正器使用96维节点表示、32维几何pair消息与一个输出头。消息来自公开的相对位置、间隙、尺寸、连接、MIB/分组和固定/预放置标志，不读取自由reference位置。

修正头最后Linear的weight/bias从零开始，初始函数精确等于E74。最终输出为：

`state = frozen_state + tanh(correction) * mask * [0.15 sqrt(a), 0.15 sqrt(a), 0.15]`。

因此每轴中心修正不超过0.15个公开等效边长，log长宽比修正不超过0.15，固定规格/预放置继续由原decoder保持。该约束不保证合法性或全分布稳定；它可能排除必要的大幅重排、小修正器也可能欠拟合，源E74本身仍可能有错误。

优化器只更新新修正器，AdamW、clip1、四档LR与早停保持协议。取消参数相对步长投影：零初始化输出头若按自身零范数限制会近乎冻结；这里由不可变主干和显式输出界约束功能扰动。不是在旧分支偷偷绕过保护。

## 固定Loss与比较

仍是E75的`G + 2 I + 5 E`：真实成对交面积比例I、平方单侧log外框扩大E，原G八项完全保留。训练/固定train-dev500/完整dev使用同一函数；best/LR/早停只看该Loss开发均值，epoch0合格，无渐入。改变了架构和可优化参数域/保护方式，只能先判断组合效果，不能单独归因于“冻结”“消息”或某个系数。

冻结v6前32k、原dev2000/train-dev500、seed0、batch32、FP32保持。train500来自旧16k，这一监测范围不变。E74已完成的raw/Q/100图只读复用，不重评原候选，测试封存。

## 执行与恢复

```powershell
.conda/envs/iccad2026/python.exe -u -B -m experiments.E76_bounded_refinement.preflight
.conda/envs/iccad2026/python.exe -u -B -m experiments.E76_bounded_refinement.pipeline --resume
```

[预检](runs/preflight/results.json)只用同五个训练案例：初始输出与E74相同、源主干在真实两步后逐字不变、训练时源eval、修正幅度上限、冻结通道不变、固定Loss、缓存/测试隔离、CUDA和真实Dropout/Adam精确恢复均检查。首版预检保存在`preflight_v1/`；最终版同步了新架构的参数统计与正则化身份。

最多2小时/200轮，单GPU；冻结主干可能降低训练计算量，但不预先声称端到端加速，实际显存/时间按日志。CPU终末2进程×2线程。输出`runs/20261029_frozen_e74_corrector_seed0/`；创建`runs/STOP`安全停，移除后原命令`--resume`，累计预算和状态不清零。

唯一own-Loss best完成dev2000、一次flash500及同100例reference/current/E42无评分图，与E74结果只读比较。E76为第五个正式候选；全局总账/30分钟跟进仍在[E72 study](../E72_topology_posttrain/study.json)。E42/E50不自动替换。
