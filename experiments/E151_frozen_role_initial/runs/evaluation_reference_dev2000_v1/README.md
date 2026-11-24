# E151 原reference dev2000评价

独立授权的冻结best45终末评价；原随机36k/2k/2k数据，无MIB，原始reference和面积不改。原训练STOP保留且不重启训练。原后处理预算不改。全量2000例正式执行，不做试跑或启动验证。

入口：python -u -B -m experiments.E151_frozen_role_initial.runs.evaluation_reference_dev2000_v1.run --arm e151_best45

结果：output/results.json；进度：output/state.json；日志：background/formal_v2/stdout.log、stderr.log。

2026-11-22进度查询时发现初次后台进程未产生进度。正式执行随后定位到遗留相对导入，已保留background/persistent_launch错误证据并修正入口；模型、权重、成员及预算不变。当前正式执行日志为background/formal_v2/，由持久会话托管。

实时查看：Get-Content -Encoding UTF8 experiments/E151_frozen_role_initial/runs/evaluation_reference_dev2000_v1/background/formal_v2/stdout.log -Tail 10 -Wait

进程托管更新：formal_v2执行已在504例后随托管会话消失，stderr为空且504份结果保留。系统拒绝CREATE_BREAKAWAY_FROM_JOB（证据background/detached_resume_504/launch.json），因此当前由持续执行会话从504例续跑，PID14744；当前日志改为background/held_resume_504/stdout.log及stderr.log。原output/state.json持续记录completed/total。入口、checkpoint、成员、后处理预算和已提交坐标均不改。
