"""Close completed E77 and deleted follow-up; preserve all original results."""
import json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
HERE=ROOT/'experiments/E77_compact_refinement'
FOLDER=HERE/'runs/20261029_e76_compact_corrector_seed0'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
utc=datetime.now(timezone.utc).isoformat()
audit=read(HERE/'runs/completion/inspection.json');review=read(HERE/'runs/review.json')
result=read(FOLDER/'results.json');flash=read(FOLDER/'terminal/summary.json')
assert audit['sources_unchanged'] and audit['public_frame_containment500']['violations']==0
assert audit['terminal_hard_legal']==500 and audit['training_seen_count']==32000
text=f'''# E77完成复核与本轮结题

自动跟进`problem-c`已按用户要求删除，E77训练/控制器进程已核实不存在。训练、唯一best的完整dev2000、一次flash500及同100例图全部完成。自动报告仅因NumPy int32无法序列化而失败，已在runs内修复保存；未修改冻结生产源码，未重训、换checkpoint或重评求解器，原错误日志保留。

## 结果

| 项目 | E42 | E74 | E76 | E77 |
|---|---:|---:|---:|---:|
| 完整dev2000草稿交面积/公开面积 | 6.302% | 5.086% | 2.733% | 2.205% |
| flash500加权Q，越低越好 | 1.025860 | 1.022511 | 1.017693 | 1.017483 |
| flash500最差单例Q | 1.927279 | 1.668639 | 1.905367 | 2.163867 |

E77对E42的加权Q差值{audit['paired_weighted_Q500']['delta']:.6f}，按worker配对重抽样95%区间[{audit['paired_weighted_Q500']['worker_resampling95'][0]:.6f}, {audit['paired_weighted_Q500']['worker_resampling95'][1]:.6f}]；302胜/65平/133负。相对E76只改善0.000210，区间[-0.002917,+0.002126]跨零，不能声称进一步稳固Q提升。算术Q均值1.031714也略差于E76的1.031126。均值优势不能替代逐例取舍。

完整dev2000交面积1999例优于E42、1例变差；弱端口324例均值8.724%→3.108%，其中稀疏157例10.237%→3.769%。原始草稿仍0/2000硬合法，flash为500/500，二者必须分开。

## 外框与空白

同500例bbox/A由E76的1.050403回到1.036766，E74为1.036783；500例都满足E74预测框限制，最大外伸仅3.06e-7个公开总面积等效边长，低于事先3e-6的FP32容差。原汇总以1e-9计面积差会把部分浮点差异标为扩大，不能将这解释为实质外扩。相对E74的框均值基本不变，交面积5.161%→2.244%，精确空白/B8.474%→5.692%，两项500例全改善。相对E76，交面积494/500改善、空白500/500改善。

同100例E42直接存档图的bbox/A均值1.040456→1.036564（70例更小、30例更大），交面积6.366%→2.273%，精确空白/B9.933%→5.703%，后两项100例全改善。因此收益没有依赖E76那样的普遍外扩，但也不表示每例都比E42外框小。

实际查看217、399、194三张原图：宏观模块顺序和分区大体保留，部分严重重叠减少；中下部、小模块以及分组边界仍有局部冲突和空隙。原图按manifest哈希核对，没有生成用户评分。这是科研观察，不能替代用户草稿评价。

## Loss、稳定性和预算

完成40轮，按同一固定Loss选择best38；四档LR均实际运行，最低档完成2轮后正常早停。初始train/dev500 Loss {audit['initial_train_dev'][0]:.6f}/{audit['initial_train_dev'][1]:.6f}，best为{audit['best_train_dev'][0]:.6f}/{audit['best_train_dev'][1]:.6f}，改善4.44%/4.03%，相对差距10.13%。train500全部来自旧16k，不能据此声称监测了新增半数的拟合差距。全部32k成员已见过，冻结源241个张量在best/latest逐字等同E76。

预算累计{result['elapsed_seconds']/60:.2f}分钟，其中未知首次启动中断保守计入77.16分钟；本次恢复训练阶段实际测得{audit['measured_resumed_training_stage_seconds']/60:.2f}分钟。用户暂停等待未计入。不能把113.10分钟全部说成有效GPU训练。best/latest、源快照、原预算、重建和恢复证据保留。

## 尾部反例与结论

最差例164（57模块）原始交面积E42为3.832%、E76为1.032%、E77为0.817%，但flash后分组违规由E42的8、E76的2上升到E77的10，总软违规11；E77 Q2.163867，高于E42的1.927279和E76的1.509876。这里减少原始重叠没有转化为最终连接/分组质量。它说明局部几何Loss和固定框不能保证后处理拓扑稳定，不宜用均值掩盖该反例；没有据此追加训练或重新选模。

结论：本轮自适应开发口径上，E77是兼顾更少重叠、紧凑外框和较低flash均值的最强候选；相对E76的主要明确收益是几何紧凑性，而非稳固的进一步Q提升。它不是全指标占优，测试封存且未获得新的用户草稿评价；E42/E50冻结基线不自动替换。六个初始候选已尝试，不再自动启动第七组。自动跟进已删除。

[机器审计](inspection.json) · [完整汇总](../review.json) · [100例reference／E77／E42图](../20261029_e76_compact_corrector_seed0/draft_comparison100/index.html)。
'''
(HERE/'runs/completion/inspection.md').write_text(text,encoding='utf-8')
# Correct generated report wording only, without altering its frozen producer.
report_path=HERE/'runs/review.md';auto=report_path.read_text(encoding='utf-8')
auto=auto.replace(f"训练阶段{result['elapsed_seconds']/60:.2f}分钟。",f"预算累计{result['elapsed_seconds']/60:.2f}分钟（含首次中断保守计入77.16分钟），本次恢复训练阶段{audit['measured_resumed_training_stage_seconds']/60:.2f}分钟。")
auto=auto.replace('本报告为自动汇总，深度失败分析和图像核验由后续唤醒完成；','本报告为保存结果汇总，[深度审计与图像核验已完成](completion/inspection.md)；自动跟进已按用户要求删除，')
report_path.write_text(auto,encoding='utf-8')
state=read(HERE/'state.json');state['original_report_error']=state.get('error');state.pop('error',None)
state['events'].append({'utc':utc,'status':'complete_after_saved_report_recovery','stage':'audited'})
state.update(status='complete_after_saved_report_recovery',stage='audited',updated_utc=utc,
    controller_pid=None,child_pid=None,actual_train_and_controller_processes_absent_verified=True,
    report='runs/completion/inspection.md',automation_status='DELETED',baseline_promoted=False)
write(HERE/'state.json',state)
study_path=ROOT/'experiments/E72_topology_posttrain/study.json';study=read(study_path)
entry=next(v for v in study['run_ledger'] if v['experiment']==HERE.name)
entry.update(status='early_stop_complete_audited',elapsed_seconds=result['elapsed_seconds'],steps=40000,epochs=40,
    best_epoch=38,weighted_Q=flash['weighted_Q'],controller_pid=None,child_pid=None,
    actual_processes_absent_verified=True,terminal_runs=1,baseline_promoted=False,
    measured_resumed_training_stage_seconds=audit['measured_resumed_training_stage_seconds'],
    durable_checkpoint_step=40000,last_verified_step=40000,last_verified_epoch=40)
study.update(status='closed_by_user_after_initial_six_candidates',current_stage='complete_audited',
    next_decision='No automatic follow-up or seventh candidate. E77 retained for human draft review; E42/E50 not automatically replaced.',
    automation_status='DELETED',automation_deleted_utc=utc,
    outcome='E77 improves dev mean Q, raw overlap and compactness versus E42; worst-case Q regresses; no formal baseline promotion or sealed test.')
study['E77_decision']={'report':'experiments/E77_compact_refinement/runs/completion/inspection.md',
    'weighted_Q':flash['weighted_Q'],'weighted_Q_delta_vs_E42':audit['paired_weighted_Q500']['delta'],
    'worker_resampling95':audit['paired_weighted_Q500']['worker_resampling95'],
    'intersection_dev2000':audit['paired_overlap2000']['current'],'raw_hard_legal':0,'flash_hard_legal':500,
    'frame_violations_500':0,'max_Q':flash['max_Q'],'test_read':False,'baseline_promoted':False}
study['cumulative_training_seconds']=sum(v.get('elapsed_seconds',0) for v in study['run_ledger'])
study['last_check']={'utc':utc,'state':'E77_audited_automation_deleted_by_user','experiment':HERE.name,'step':40000}
write(study_path,study)
(HERE/'runs/STOP').write_text('E77 complete and audited. Automatic follow-up deleted at user request. Do not retrain or repeat terminal evaluation.\n',encoding='utf-8')
print('E77 audit saved; study closed; automation deletion recorded.',flush=True)
