# E85 E74协同空间修正对照

两组同E74/F81起点、同241,547个可训练参数、同固定Loss。用户于协同组中途取消后训练学习率保护；已完成控制组保留原策略，因此终末差异同时包含调度政策变化，不能单独归因为协同算子。训练前算子直接效果单列。所有统计为已暴露自适应开发证据，测试封存。

|配置|训练分钟|停止/best|dev2000交面积|弱端口稀疏交面积|flash加权Q|最大Q|
|---|---:|---|---:|---:|---:|---:|
|control|38.77|early_stop/23|2.120%|3.309%|1.014964|1.815592|
|cooperative|31.68|early_stop/9|1.942%|3.025%|1.014672|1.812290|

## 协同组减控制组的开发集配对差

all: {"mean_delta": -0.0017801047229375044, "worker_bootstrap95": [-0.0018599525704146684, -0.0017023603320290338], "better_equal_worse": [1916, 0, 84]}
weak_port: {"mean_delta": -0.0024235272366186244, "worker_bootstrap95": [-0.0026577132357644495, -0.0022026985135188996], "better_equal_worse": [315, 0, 9]}
weak_port_sparse: {"mean_delta": -0.0028382786732319273, "worker_bootstrap95": [-0.0031424122838121953, -0.0025199449978083022], "better_equal_worse": [152, 0, 5]}
rich_port_dense: {"mean_delta": -0.0011930992982047803, "worker_bootstrap95": [-0.0013121964089926432, -0.0010826258210180901], "better_equal_worse": [372, 0, 27]}

Q配对：{"weighted_delta": -0.00029229511487174317, "worker_bootstrap95": [-0.001799394664482297, 0.001011636028459564], "better_equal_worse": [171, 154, 175]}

初始与最终分组、无端口模块近邻次序、共同G、完整Loss与失败尾部见[comparison.json](comparison.json)。原始硬合法与后处理合法分别报告，不按Q升级或重选checkpoint。

[control 同100例四阶段图](20261030_P74_F81_control_seed0/four_stage100/index.html)
[cooperative 同100例四阶段图](20261030_P74_F81_cooperative_seed0/four_stage100/index.html)

[完成核查与研究结论](completion/inspection.md)。图册链接在完成核查中修复为相对本报告路径，冻结生成源码不变。
