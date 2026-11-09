# dev500评分导入失败与修复

2026-11-04用户报告error。原流水线于北京时间03:20在dev500首例评分入口退出，原因是隔离目录遗漏官方评分器在模块导入时引用的上游文件，首先报`No module named litetestLoader`。导入短检进一步发现`cost → utils → prime_dataset`依赖；完整本地依赖链已检查并补齐。

已逐字节复制原上游`litetestLoader.py`、`liteLoader.py`、`lite_dataset.py`、`lite_dataset_test.py`、`visualize.py`、`prime_dataset.py`。未改动任何模型、Loss、求解器、官方评分逻辑、数据输入或权重。仅修正未运行的准备脚本，避免将来漏带依赖。controller_sources增加这些文件的SHA校验；原训练sources.json、source_snapshot.zip、冻结选择与导出模型全部保持原样，新增依赖单独打包留证。

原报错时dev结果保存0例，test2000尚未读取。完成导入短检后，从原完整流水线`run.py --resume`继续，自动跳过全部已早停阶段与冻结步骤，从dev500开始，随后执行已授权的2000例重复测试。不重训，不按Q或测试改选模型。

已完成训练：E42正常早停54轮/best51，E74正常早停45轮/best43，F正常早停18轮/best6。冻结模型SHA256：`ca0465e48da7a0e1fc345aee28872e412209ff268edc774b43347c7496748d1a`。

证据：原error/state/日志保留本目录；migration.json和phase2.json记录依赖补全经过；import_check.json记录完整导入及哈希检查通过；恢复进程见bundle的resume20261104.json。后续状态与结果以bundle/state.json、results.json为准。
