# E98 best71逐层只读诊断

用户要求先观察中间布局，再讨论网络重设计。本入口只取5个原登记训练成员，不启动训练、搜索、后处理或测试；所有旧STOP保持。

模型为增强96×12 best71，源码使用同实验`generalization_baseline/model.py`，原登记MIB平均面积与公开关系输入忠实保留；不是新的无MIB基线。成员仅按元数据选取最先接近N=21/45/70/95/120的原train成员，不看输出挑题。

通过只读forward hooks记录共享head的13次真实输出、对应96维隐藏特征、12次位置嵌入及注意力/图聚合/FFN分支。原forward不修改；同五例带钩子/不带钩子最终输出逐位一致，参数和原checkpoint未变。CPU FP32 eval、batch1，关闭Dropout；不是GPU训练数值逐位复现或延迟评测。

输出为`runs/20261106_layer_diagnosis5/`，图与交互入口统一放在`visualization/E98_layer_diagnosis/20261106_best71_train5/`。默认同例同ID颜色、全阶段共同坐标尺度，无裁剪；模块可选中追踪中心轨迹。没有评分控件，不自动打开界面。

- [冻结身份](../runs/20261106_layer_diagnosis5/frozen.json)
- [诊断报告](../runs/20261106_layer_diagnosis5/report.md)
- [最终一致性及只读核验](../runs/20261106_layer_diagnosis5/verification.json)
- [逐层观察入口](../../../visualization/E98_layer_diagnosis/20261106_best71_train5/index.html)

运行命令：`python -m experiments.E98_loss_weight_learning.layer_diagnosis.run`。完成结果不覆盖，不重复执行完整同身份观察。
