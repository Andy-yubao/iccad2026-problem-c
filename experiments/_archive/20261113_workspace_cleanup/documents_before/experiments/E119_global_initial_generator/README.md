# E119_global_initial_generator

2026-11-09用户明确授权：初始预测前2层全局Attention+连接/分组聚合，保留原12几何块和.2L4+.8L12。均36k训练，准备检查通过后后台启动；完成后自动固定dev500验证。仅本组seed0，4h/200轮，batch32/FP32/增强/LR调度/AdamW及数值门禁不变。原始面积、无MIB、全模块预测，固定规格由后处理满足。

E114原4+12 best48与既有dev500结果只读复用，不重训；不采用Node26，不读test，不按Q回选权重，不新增图集。每组独立模型/Loss/后处理副本，共享原36k/dev500静态CPU缓存。仅同五训练例必要检查，一次可丢弃受保护更新；正式训练重新seed0。

串行队列由E118_first12_supervision/run_queue.py拥有；E118训练→评价→E119训练→评价。失败/停止不推进、不自动重试。全局STOP/QUEUE_STOP在E118目录，当前实验/具体run/evaluation的STOP也有效；其他旧STOP不动。日志各自launch_logs/train.stdout.log、train.stderr.log及evaluation.stdout.log、evaluation.stderr.log；进度各自runs/global2_seed0/heartbeat.json，评价结果在该run的evaluation_no_mib_dev500。

训练反传、监测、best/LR/早停共用该组固定目标；各自唯一Loss-best冻结后执行原E98.depth_budget.v1，不采用三倍预算；报告草稿Loss/合法性、最终本地Q/硬合法及完整耗时，全部坏例与慢例保留。
