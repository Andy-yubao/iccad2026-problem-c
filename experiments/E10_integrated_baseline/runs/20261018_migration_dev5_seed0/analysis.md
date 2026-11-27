# E10 整合检查

冻结 dev_quality 中 worker 0、24、49、74、99 各1例，CPU、2线程、seed=0。调用新的模型＋后处理入口，官方评分与独立硬检查均5/5通过，平均中性时间Q为1.1505791157。该值仅作五例迁移功能检查，不与开发100例比较，也不用于提出新的质量或速度改善结论。

实际命令、源码哈希、设备、冻结索引哈希见 [config.json](config.json)，最终坐标只存于 [results.json](results.json)。全工作区静态验收见 [workspace.json](workspace.json)。模型与算法源文件继承已有E7/E8，未新增训练、未读取local_test。

迁移时原文件逐一核对大小与SHA-256，映射与哈希见 [manifest.json](../../../../history/20261018_E10_consolidation/manifest.json)。算法实现按依赖复制后保持字节一致；E10增加统一推理入口和本轻量检查入口。原训练和历史评价记录不改写，唯一best/latest本机保存，均不进入Git。
