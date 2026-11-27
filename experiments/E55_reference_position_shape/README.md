2026-10-26已完成：43轮早停／best40，训练30.89分钟；同口径reference开发Loss比E42高6.02%，草稿重叠增加。flash500全硬合法，加权Q1.051852，155胜／35平／310负，未替换基线。[训练结果复核、曲线与同例图集](runs/20261026_reference_position_shape_seed0/review.md)。本轮未续训或启动新实验，测试封存。

# E55：只拟合 reference 的位置与形状

2026-10-26用户在工作区整理后明确授权后台启动单次训练，并在训练结束后立即执行500例检查。此授权仅适用本实验，不恢复旧自主研究、定时任务或E54。预计训练约30–120分钟（上限2小时／200轮），后处理约数分钟，取决于实际机器负载。

## 假设与控制变量

只学习参考中心和形状，观察取消额外几何惩罚能否更好复现参考结构。若拟合仍差，说明既有几何Loss之间的冲突不是唯一障碍；若参考误差下降但原始冲突、尾部或用户草稿评价变差，也不能仅凭较低Loss升级基线。固定单seed0、E42同架构96×12 rectangle、FFN dropout .05；从头随机初始化，不继承旧训练权重。实验是新授权的中心／形状对照，不是恢复已取消E54的多项reference分支。

## 唯一固定Loss

`L = center_local + 0.25 × shape`，沿用原E42这两项的定义与权重。

- center_local：原始预测中心与reference中心分别除以模块自身sqrt(公开面积)，SmoothL1 beta=.1；按可变中心维度取均值。
- shape：原始预测log(w/h)与参考log(w/h)平方误差，按可变形状模块取均值。
- 每布局先聚合，再对布局等权平均。训练反传、train_monitor500、dev_monitor500使用同一项、权重、mask、归一化与聚合；best／LR／早停只看该Loss的dev500均值。全程固定，没有权重渐入或误差容忍带。
- 不计算重叠、边界、MIB、接触、相对位移、参考bbox、紧凑性、空白、拥挤、模块负担或线长额外目标。Loss函数只读取target、mask、公开area。

公共连接／分组／MIB输入仍属于原模型输入；解码固有的面积守恒、坐标截断、固定形状和预放置处理不变。这里关闭的是额外Loss，不是移除架构约束。参考几何来自原任务，MIB平均面积后的精确可行性不由此保证。

## 数据、训练和终末评价

v5 train16000，固定train_monitor500／dev_monitor500，测试封存。原缓存直接引用E42共享缓存，不复制张量。独立复制原快速FP32 CUDA graph引擎；AdamW decay .0003、batch32、clip1，学习率1e-3→2.5e-4→1e-4→5e-5，第15轮起原平台规则下降与早停。AdamW和Dropout仍是原控制条件，不新增几何惩罚。[来源](provenance.json)、[Loss配置](loss_config.py)、[五例检查](runs/20261026_preflight/results.json)。

后台入口：[pipeline.py](pipeline.py)。顺序为训练→冻结唯一Loss-best→原flash同dev500→同100例reference／本轮draft／E42 draft无评分图→基于已存结果报告。不在训练中求解，不根据终末Q重选checkpoint，不打开评分页面，也不运行pro或封存测试。终末500采用2进程×2线程，错误／不可行例保留在主结果中。

输出：`runs/20261026_reference_position_shape_seed0/`；控制状态和控制台日志：`runs/20261026_reference_position_shape_seed0_control/`。最终报告为训练目录下`analysis.md`，完整数值及原始诊断见`analysis.json`、`terminal/results.json`。原始draft质量和后处理质量分开报告。只能比较两模型共同定义的分项，不能横比E42历史总Loss和本轮总Loss。

首次启动或中断后恢复（已完成训练或终末评价会复用，不重复跑）：

```powershell
& 'E:\project\ICCAD\2026\problemC\.conda\envs\iccad2026\python.exe' -u -B -m experiments.E55_reference_position_shape.pipeline --resume
```

需在项目根目录执行。安全停止：在本实验`runs/STOP`或本轮训练目录下创建`STOP`文件，当前保存边界后退出，并停止后续阶段。恢复前移除自己创建的STOP，再用原命令恢复；源码、数据、预算和设备身份检查保持严格，累计预算不清零。后台启动由本轮用户明确授权，不设自动续跑下一轮。
