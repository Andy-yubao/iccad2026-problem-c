# E81小布局器＋加宽多来源修正模型结果

完成27轮，best26，停止原因early_stop，训练阶段64.77分钟。唯一best按每例固定两来源Loss的开发均值选择，未按Q选择。

完整dev2000的部署P42草稿交面积2.491%，原始硬合法0/2000。一次flash500加权Q=1.016242、P95=1.123021、最大=1.825773，硬合法500/500。训练/开发总Loss与两个来源分项必须同时阅读：[曲线](loss_curves.png)，[完整数据](review.json)。不与旧定义总Loss横比。

部署1713086参数，其中独立F241547参数；训练时辅助P74不进入部署计算。由唯一best确定性导出[部署模型](20261029_e42_multisource_seed0/deployment.pt)和[独立修正模型F](20261029_e42_multisource_seed0/correction_F.pt)，父SHA及导出SHA见review.json。

扩大F、两来源训练、联合形状优化和通道压力同时改变，只能先报告组合效果。初始暖启动误差和负面证据保留，测试封存，不自动升级E42/E50。仍待反例、来源框与3至5图深度复核。

[同100例reference／本轮draft／E42图](20261029_e42_multisource_seed0/draft_comparison100/index.html)。
