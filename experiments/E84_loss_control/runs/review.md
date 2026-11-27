# E84同起点G-only目标包控制

与既有E81保持架构、初始化来源、双来源、数据及优化日程一致；每来源训练/监测/选模固定为完整原G，I/E/A只诊断。原F79已学过几何目标，不能称从未受几何监督；本组只检验本轮后训练和匹配选模的目标包。主要同例比较E81见review.json，旧E81不重训/重评。

完成30轮，best29，停止原因early_stop，训练阶段69.91分钟。唯一best按每例固定两来源Loss的开发均值选择，未按Q选择。

完整dev2000的部署P42草稿交面积4.816%，原始硬合法0/2000。一次flash500加权Q=1.017497、P95=1.132741、最大=1.791697，硬合法500/500。训练/开发总Loss与两个来源分项必须同时阅读：[曲线](loss_curves.png)，[完整数据](review.json)。不与旧定义总Loss横比。

部署1713086参数，其中独立F241547参数；训练时辅助P74不进入部署计算。由唯一best确定性导出[部署模型](20261029_same_initialization_G_only_seed0/deployment.pt)和[独立修正模型F](20261029_same_initialization_G_only_seed0/correction_F.pt)，父SHA及导出SHA见review.json。

本组只改变固定训练与匹配选模目标包，架构、两来源、初始化和操作保持E81原样；不能归因于单个I/E/A项，也不能称从未受几何监督。测试封存，不自动升级E42/E50，须继续核对反例、来源框与3至5图。

[同100例参考／布局器／G-only修正／后处理](20261029_same_initialization_G_only_seed0/draft_comparison100/index.html)。
