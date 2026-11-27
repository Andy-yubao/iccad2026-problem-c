# E93 两个冻结分支的开发验证

用户2026-11-01授权“运行验证”。按各自固定Loss选出的唯一C best10、G best16，分别复现dev500自身Loss、完整dev2000草稿诊断、一次flash500及固定100例reference/本支/E90草稿图。四进程×每进程两线程；同原E90后处理、计时和计分口径，E90/E92结果只读复用。不同总Loss不可直接比较；同时报告公共C/F诊断值。测试封存，不据Q反选checkpoint或自动续训。

评价算法从E92独立复制；runtime精确保留原E90哈希，provenance.json逐文件记录。移除旧E92研究门槛、study/修正器授权/自动唤醒逻辑。当前授权仅评价与报告。登记见../runs/evaluation_registration.json，源码快照和冻结权重身份见各支qualification/frozen.json及source_snapshot.zip。

入口：`.\.conda\envs\iccad2026\python.exe -u -B -m experiments.E93_matched_posttrain.qualification.pipeline`。串行C→G，各自完成结果复用；只续未完成成员，不重复终末。预算两支累计上限2h（准备预检另计实际费用）。创建`runs/evaluation/STOP`可在当前提交案例结束后停止。不能改变已冻结源码再静默恢复。

预计约10–30分钟，未作本次全量计时；GPU用于批草稿检查，CPU用于固定4进程×2线程flash与4进程绘图。输出：各支`runs/20261101_*_seed0/qualification/`及汇总[runs/evaluation/report.md](../runs/evaluation/report.md)。状态见`runs/evaluation/state.json`、各支heartbeat与stdout/stderr日志。不恢复E92、E91或旧修正控制器。
