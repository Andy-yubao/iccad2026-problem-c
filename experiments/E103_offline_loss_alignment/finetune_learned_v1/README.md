# E103：第一版权重在E98增强基线上的续训

用户明确授权直接使用第一轮拟合的新Loss续训，初始LR1e-3、不设下降保护、不再限制15轮。本次单独登记并后台训练一个候选，不恢复任何E98旧搜索；既有离线下降门槛未通过的证据仍保留。

唯一父权重为已复制并核验的E98泛化增强best71。精确使用bank1000x5第一轮proposal的physical_coefficients，不舍入或修改近零线长项，不使用canonical-view、二次拟合或bootstrap权重。模型96×12，FFN Dropout .10、残差Dropout .05、50%完整xy交换保留，固定位置/形状与公开约束同步变换。原train16k和固定train/dev各500不变，batch32、FP32、不使用AMP/TF32。

变更目标后重置AdamW动量和age，继承模型、RNG与投影计数；weight_decay=.0003、clip1、模块相对更新界.005不变。新阶段延续父模型完成71轮后的样本/增强seed流；阶段内恢复保留完整状态。

新固定Loss统一用于反传、train/dev监测、best、LR和早停，题目间算术平均。epoch0建立验证基准且可选best；LR依次1e-3、2.5e-4、1e-4、5e-5，每档连续3轮无明显改善即可下降，无保护。最低档连续3轮停滞早停，**没有轮数或时间上限**。best绝对改善门槛1e-5，调度明显改善门槛max(1e-5,0.1%当前档参考Loss)，每次降档重建参考。高LR不保证跳出局部最优，原步长保护可能限制实际参数位移。

新run为`runs/20261104_learned_v1_lr1e3_seed0`，best/latest分别保存。训练结束冻结唯一own-Loss best并报告，不自动做后处理Q或读取测试，不追加其他分支。原模型和所有旧STOP、数据库及离线证据保留。

后台入口launch.ps1，日志本目录stdout.log/stderr.log。写入本目录STOP或新run/STOP，在接受批次边界保存并退出；不自动重试错误。不专门测试断点恢复，仅五例实际1e-3更新短检。
