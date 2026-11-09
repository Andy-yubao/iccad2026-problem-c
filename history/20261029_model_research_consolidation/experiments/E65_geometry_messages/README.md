# E65：冻结G下的几何消息与匹配容量对照

2026-10-28本阶段两组均正常早停，完整raw2000、flash500及各100图完成。geometry的dev500 G0.351731、raw交面积6.241%，相对E64分别低4.48%／12.66%；feature对照为0.357486／6.495%。几何通道相对对照有小幅晚期开发与重叠收益，但共同早期步数不占优，不能把全部收益归于几何。保留单seed候选，不升级E42。已抽看5张同例图，视觉验收仍待用户；详见[阶段复核](runs/review.md)。长期研究继续独立E66，不做多种子。

本聊天用户已授权后台长训练与每小时自动唤醒推进四条表达/泛化路线，随后明确**不要做多种子确认**。本轮全部seed0，原E42/flash与E64统一协议参照保持冻结，测试封存。[长期路线与预算](long_task.md) · [累计任务状态](long_task.json) · [本阶段状态](state.json)。

## 假设与消融

在原96×12主干每层后增加24维成对消息值，而非只用几何调节注意力分数。消息从归一化节点隐藏状态、当前矩形相对位置/间隙/穿透/形状、固定/预放置标志和公开连接/组关系计算，经有效邻居归一化注意力聚合后残差加入。新分支输出投影零初始化、残差系数0.1；原主干参数及初始化随机状态保持原样。两个变体都从头训练。

| 变体 | 作用 |
|---|---|
| geometry_message | 19维边输入含12维明确几何量、4维公开固定标志、3维关系 |
| feature_message | 参数与执行形状相同，将前12维显式几何输入置零；检验单独增加消息分支的收益 |
| E64 seed0 | 复用已完成的无新消息分支统一G对照，不重复训练 |

隐藏特征仍携带原主干几何，feature_message不是彻底移除几何信息的网络。被屏蔽列缺少数据依赖梯度，所以登记参数量相同不代表有效自由度绝对相同；这是显式几何通道消融，不能泛称所有容量与优化影响均已排除。新分支归一化只作用于新路径，不改变原图聚合路径，不能宣称解决所有历史稳定性问题。

独立模型源码见[model.py](model.py)，原主干字节副本为[base_model.py](base_model.py)。来源与受保护checkpoint哈希见[provenance.json](provenance.json)，预登记比较边界见[plan.json](plan.json)。不跨实验导入私有算法。

## 冻结的训练与评价

E64八项G与实现字节保留：反传、固定train/dev监测、完整开发评价采用同一项/权重/归一化/等例聚合；没有辅助Loss、渐入、occupancy代理或中间监督。best、LR与早停只用固定dev_monitor500 G。v5原train16000成员及逻辑顺序、固定train/dev500，batch32、FP32、AdamW decay3e-4、clip1、FFN dropout.05、原四档LR与15轮保护期、最低档至少两完整轮均沿用E64。max3小时/200完整轮；原E64实际早停预算未耗尽，比较同时报告共同更新数轨迹和各自best，不宣称实际计算完全相同。

两组依次完成唯一best的raw dev2000、原flash dev500、同100例reference/current/E42无评分图及曲线。用户看图评价优先；固定G为主要数值参考，Q辅助；AI抽看不替代用户评价。不按Q重选模型，不自动升级、不读测试。开发面板已自适应暴露，单seed结论不冒称跨种子稳健或独立泛化。

CPU评价2进程×2线程、图片4进程×1线程，总线程预算4。共享训练/开发缓存原位只读复用，GPU常驻输入约2.99GiB，新分支额外激活显存与实际吞吐由预检及正式日志记录。显存不足不静默改变batch或精度。

## 执行与恢复

本阶段只用同五个train成员做短检查：冻结Loss/梯度一致、新分支零初始化与E64逐位等价、几何输入确实影响消息、新分支实际更新、120模块padding、非有限值阻止Adam、CUDA与eager实际更新一致，以及含Dropout的中断恢复逐位一致。证据：[预检结果](runs/20261027_preflight/results.json)。

```powershell
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E65_geometry_messages.preflight
& '.\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E65_geometry_messages.pipeline --resume
```

工作目录为项目根，环境`.conda/envs/iccad2026`。后台使用持续exec会话运行控制器，保持父进程，不使用立即退出的Start-Process父shell。控制器要求当前源码对应通过的预检，源码快照与配置逐运行保存。geometry_message先跑，feature_message后跑，全部seed0。通过完成结果幂等跳过旧训练和已完成评价；源码不匹配则拒绝恢复，不绕过校验。

输出为`runs/20261027_<arm>_seed0/`，训练日志`runs/<arm>_train.log`，终末日志相应`*_final.log`，状态见[state.json](state.json)。训练best及latest分开保存，保留优化器、四档调度、随机状态、样本游标、已见成员与累计预算。安全停止：创建本实验`runs/STOP`；保存边界后停止且不进入下一组。恢复先移除自己创建的STOP，再执行同一pipeline --resume；失败没有结果时依latest恢复，不能改预算/源码后静默继续。

完成后报告：[runs/review.md](runs/review.md)、[runs/review.json](runs/review.json)及learning_curves.png。阶段结果由下一次每小时唤醒读取，写明继续/淘汰理由后准备E66；没有多种子确认阶段。总任务最多10次正式训练、累计18小时训练墙钟、单次3小时/200轮及72小时日历保护，见long_task.json；失败和恢复均计费，耗尽停止新运行并交付现有证据。
