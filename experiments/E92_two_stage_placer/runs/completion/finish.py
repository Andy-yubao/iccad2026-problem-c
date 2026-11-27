"""Finalize saved evidence after failed E90 gate. No inference or Q calls."""
import json,sys
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
from experiments.E92_two_stage_placer.io import HERE,ROOT,save,sha
from experiments.E92_two_stage_placer.qualification.common import registration,charge_auxiliary,auxiliary_seconds
from experiments.E92_two_stage_placer.budget import global_charged_seconds

def main():
    out=Path(__file__).resolve().parent;run=HERE/'runs/20261031_two_stage32k_seed0';q=run/'qualification'
    assessment=json.loads((q/'assessment.json').read_text(encoding='utf-8'));assert assessment['status']=='failed'
    reg=registration();baseline=reg['baseline'];actual=assessment['values']
    source=next(n for n in baseline['source_files'] if n.replace('\\','/').endswith('terminal/results.json'))
    old=json.loads((ROOT/source).read_text(encoding='utf-8'))['records'];new=json.loads((q/'terminal/results.json').read_text(encoding='utf-8'))['records']
    assert len(old)==len(new)==500 and all(a['member']==b['member'] and a['n']==b['n'] for a,b in zip(new,old))
    n=np.asarray([r['n'] for r in new]);a=np.asarray([r['metrics']['cost'] for r in new]);b=np.asarray([r['metrics']['cost'] for r in old]);w=np.exp((n-n.max())/12);delta=a-b
    strata=[]
    for low,high in ((20,40),(40,60),(60,80),(80,100),(100,121)):
        mask=(n>=low)&(n<high)
        if mask.any():strata.append({'N_min':low,'N_max':high-1,'count':int(mask.sum()),
            'weight_fraction':float(w[mask].sum()/w.sum()),'E90_weighted_Q':float(np.average(b[mask],weights=w[mask])),
            'E92_weighted_Q':float(np.average(a[mask],weights=w[mask])),'delta_Q':float(np.average(delta[mask],weights=w[mask]))})
    worst=[{'index':int(i),'member':new[i]['member'],'n':int(n[i]),'E90_Q':float(b[i]),'E92_Q':float(a[i]),'delta_Q':float(delta[i])} for i in np.argsort(delta)[-3:][::-1]]
    progress=json.loads((run/'progress.json').read_text(encoding='utf-8'));history=progress['history']
    old_progress=json.loads((ROOT/'experiments/E90_auxiliary_weight_placer/runs/20261031_aux3_seed0/progress.json').read_text(encoding='utf-8'))
    old_loss=min(old_progress['history'],key=lambda r:r['dev_monitor']['total'])['dev_monitor']
    cur_loss=json.loads((q/'dev500_loss.json').read_text(encoding='utf-8'))['losses']
    coefficients={'reference':1.,'overlap_total':.08975162861857286,'task':.3336832108335229}
    components={key:{'E90':old_loss[key],'E92':cur_loss[key],'weighted_delta':(cur_loss[key]-old_loss[key])*weight} for key,weight in coefficients.items()}
    ledger=HERE/'runs/task_auxiliary_budget.json';entries=json.loads(ledger.read_text(encoding='utf-8'))['entries']
    eval_seconds=sum(r['seconds'] for r in entries if r['label']=='frozen qualification all')
    label='heartbeat qualification closure 2026-10-31T17:51:45.035Z'
    if not any(r['label'].startswith(label) for r in entries):
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat('2026-10-31T17:51:45.035+00:00')).total_seconds()
        charge_auxiliary(max(0.,elapsed-eval_seconds),label+'; excludes already charged evaluator wall')
    images=json.loads((q/'draft_comparison100/manifest.json').read_text(encoding='utf-8'))['cases']
    assert len(images)==100 and len({tuple(r['member']) for r in images})==100
    for row in images[:2]:assert sha(q/'draft_comparison100'/row['image'])==row['image_sha256']
    assert sha(run/'checkpoints/F_best.pt')==assessment['checkpoint_sha256']
    results=json.loads((run/'results.json').read_text(encoding='utf-8'))
    analysis={'status':'terminated_layout_gate_failed','assessment':assessment,'same500_strata':strata,
        'wins_losses_ties':[int((delta<-1e-12).sum()),int((delta>1e-12).sum()),int((abs(delta)<=1e-12).sum())],
        'largest_regressions':worst,'dev500_loss_components':components,'training':results,
        'total_auxiliary_seconds':auxiliary_seconds(),'total_new_task_seconds':global_charged_seconds()+json.loads(ledger.read_text(encoding='utf-8'))['charged_seconds']-61407.259365999955,
        'corrector_updates':0,'automation_deleted':True,'terminal_Q_repeated':False,'test_read':False,
        'visual_checks':{'count':2,'indices':[r['index'] for r in images[:2]],'scope':'artifact layout/labels/common axes only; not human feedback or quality ranking'}}
    save(out/'summary.json',analysis)
    save(HERE/'runs/STOP',{'reason':'User requested termination if E90 gate failed; closed after completed qualification','corrector_updates':0,'automation_deleted':True})
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
    study.update(completion_report=(out/'report.md').relative_to(ROOT).as_posix(),auxiliary_charged_seconds=auxiliary_seconds(),total_new_task_charged_seconds=analysis['total_new_task_seconds'])
    save(HERE/'study.json',study)
    table='\n'.join(f'| {r["N_min"]}～{r["N_max"]} | {r["count"]} | {r["E90_weighted_Q"]:.6f} | {r["E92_weighted_Q"]:.6f} | {r["delta_Q"]:+.6f} |' for r in strata)
    report=f'''# E92完成：Loss持平，布局质量未过E90门槛，按约定终止

2026-11-01。**布局阶段未达到预登记质量门槛，因此本任务已终止、每小时自动唤醒已删除，修正网络未设计／未训练，Q<1.01目标未达成。** 所有权重、Adam/RNG断点、原始数据和唯一终末结果保留，未开放测试或更换冻结E42/E50。

## 训练执行

完整32k、batch32、192×12全参数随机初始化。累计2h预算停止，完成55轮、55,262次更新，正在F第17轮；不是早停或收敛。P第20轮首次降LR至2.5e-4，第39轮第二次平台触发时不再降LR，回载P own-best36，切换F Loss并新建Adam。F以epoch0为基准、无保护期，第3／11轮降至1e-4／5e-5。最终选F自身Loss best15（全局第54轮），未用Q换checkpoint。

F best SHA256：`{assessment['checkpoint_sha256']}`。全部711步预算迁移状态和历史费用保留。没有触发数值爆炸保护；这不等于证明所有未监测输入均稳定。

## 固定同例开发500

| 指标 | E90 | E92 |
|---|---:|---:|
| 相同F Loss | {baseline['dev500_F_loss']:.8f} | {actual['dev500_F_loss']:.8f} |
| 加权Q，越低越好 | {baseline['weighted_Q']:.8f} | {actual['weighted_Q']:.8f} |
| 算术Q | 1.03882594 | {actual['mean_Q']:.8f} |
| 最差Q | {baseline['max_Q']:.8f} | {actual['max_Q']:.8f} |
| 后处理硬合法 | 500/500 | 500/500 |
| 模块覆盖并集≤10%达标率，模块等权 | {baseline['module_pass10_pooled']:.2%} | {actual['module_pass10_pooled']:.2%} |
| 每例最差模块覆盖比例P95 | {baseline['worst_module_coverage_p95']:.2%} | {actual['worst_module_coverage_p95']:.2%} |
| 后处理平均秒／例 | 0.27443 | {actual['seconds_mean']:.5f} |

Loss门槛通过，但加权Q、最差Q、覆盖达标率和最差模块覆盖P95四项未通过。[完整门槛](../20261031_two_stage32k_seed0/qualification/assessment.json)未作任何事后调整。两者均为相同flash参数、4进程×2线程；E90只复用旧结果，E92终末只求解一次。耗时只是本次实际观察，不据此宣称算法加速。

同worker配对重抽样5000次：加权ΔQ={assessment['weighted_Q_delta']:+.8f}，95%区间[{assessment['worker_paired_95pct_interval'][0]:+.8f}, {assessment['worker_paired_95pct_interval'][1]:+.8f}]，本次开发对照支持退化。228例改善、206例退化、66例相同，逐例胜数不能代替模块数加权Q。

## 退化集中在较大布局

| 实际模块数N | 例数 | E90加权Q | E92加权Q | 差值 |
|---|---:|---:|---:|---:|
{table}

20～59模块布局有改善，但100～120模块组明显退化，解释了算术均值接近而加权Q变差。该分层是结果解释，不是新选模规则。反例：index329（64模块）Q1.08872→1.65833；index15（106模块）1.03105→1.57683；index99（75模块）1.21958→1.57922。所有失败和尾部均纳入，未筛样本。

## Loss与实际几何仍有错位

同dev500 reference项0.22861459→0.22515589，重叠组合Loss0.44723026→0.48294913，task项0.18084872→0.17928474。reference与task改善抵消了重叠项退化，令总Loss小幅下降，却未带来更好的覆盖尾部和最终Q。这里的重叠Loss不是覆盖面积比例。

完整dev2000的F Loss0.32578574、原始草稿硬合法0/2000、成对交面积／面积均值5.756%、每例最差模块覆盖P95为85.12%、外框／模块面积均值1.02731、精确空白／外框均值8.221%。这是草稿诊断，不能用后处理500/500合法冒充草稿合法。

此轮表明两阶段流程可执行，且已知E91反例没有在监测中重现其数千Loss，但不能证明两阶段本身解决了不稳定：缺少严格匹配的固定Loss日程消融。E90为既有权重续训且部分参数冻结，本轮为32k全参数从头训练，不能将差异只归因于Loss日程。依用户停止规则，本轮不追加训练、改权重、改门槛或另挑Q更好的权重。

## 证据

- [原始资格结果](../20261031_two_stage32k_seed0/qualification/assessment.json)、[完整2000草稿](../20261031_two_stage32k_seed0/qualification/full_dev.json)、[唯一终末500](../20261031_two_stage32k_seed0/qualification/terminal/results.json)
- [同100例reference／E92／E90草稿图，无评分控件](../20261031_two_stage32k_seed0/qualification/draft_comparison100/index.html)。检查了前两图217／194的模块标签、共同尺度和显示完整性；这只是渲染验收，不冒充用户反馈。
- [保存结果分析与预算](summary.json)、[分析入口](finish.py)。准备／评价累计{analysis['total_auxiliary_seconds']:.2f}秒；本任务新增资源保守累计{analysis['total_new_task_seconds']:.2f}秒，未使用条件修正训练额度。自动唤醒删除结果已确认，STOP防止意外恢复。
'''
    (out/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({'status':analysis['status'],'new_weighted_Q':actual['weighted_Q'],'baseline_weighted_Q':baseline['weighted_Q'],'total_auxiliary_seconds':analysis['total_auxiliary_seconds'],'total_new_task_seconds':analysis['total_new_task_seconds'],'report':str(out/'report.md')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
