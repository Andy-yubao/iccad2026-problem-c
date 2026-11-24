"""Compare actual own-best monitoring records after the authorized run completes."""
import csv,json,statistics
from .io import HERE,ROOT,save

COMPONENTS=('total','reference_weighted','position','shape','overlap_weighted','overlap_total',
    'wire_compact_weighted','grouping_weighted','boundary_weighted','public_shape_weighted',
    'public_preplaced_position_weighted','mib_weighted')

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def record(experiment,arm):
    path=experiment/'runs'/f'{arm}_seed0'
    frozen=read(experiment/'frozen_initial.json');progress=read(path/'progress.json')
    selected=next(h for h in progress['history'] if h['epoch']==frozen['best_epoch'])
    return dict(frozen=frozen,result=read(path/'results.json'),panels=read(path/'panels.json'),
        config=read(path/'config.json'),selected=selected)

def compare():
    baseline=record(ROOT/'experiments/E158_role_initial3','e158_initial')
    current=record(HERE,'e159_initial')
    out=HERE/'comparison_e158';out.mkdir(exist_ok=True)
    members={name:baseline['panels'][name]==current['panels'][name] for name in
             ('train_members','train_monitor_members','dev_monitor_members')}
    loss_keys=('reference_coefficient','overlap_scale','effective_overlap_weight','mib_weight',
               'grouping_bottleneck_weight','layer_weights')
    loss_match={k:baseline['config']['identity']['loss'][k]==current['config']['identity']['loss'][k] for k in loss_keys}
    comparable=all(members.values()) and all(loss_match.values())
    summary=dict(status='complete' if comparable else 'incomparable_metadata',same_members=members,
        same_loss_configuration=loss_match,baseline=baseline['frozen'],current=current['frozen'],
        baseline_training=baseline['result'],current_training=current['result'],
        checkpoint_reselection=False,additional_inference_calls=0,postprocessing_calls=0,test_read=False)
    lines=['# E159 与 E158 的冻结最佳点 Loss 比较','',
        '各自按原登记Loss选出的唯一best，不按本次比较重选权重。','']
    if comparable:
        rows=[];paired={}
        for panel,label in [('train_monitor','train500'),('dev_monitor','dev500')]:
            a=baseline['selected'][panel];b=current['selected'][panel]
            for key in COMPONENTS:rows.append(dict(panel=label,component=key,E158=a[key],E159=b[key],delta_E159_minus_E158=b[key]-a[key]))
            differences=[y-x for x,y in zip(a['per_case_total'],b['per_case_total'])]
            paired[label]=dict(count=len(differences),improved=sum(d<-1e-7 for d in differences),
                worsened=sum(d>1e-7 for d in differences),median_delta=statistics.median(differences),
                mean_delta=statistics.mean(differences))
        summary.update(components=rows,paired_per_case=paired)
        with (out/'loss_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        lines+=['| 面板 | 分项 | E158 | E159 | E159−E158 |','|---|---|---:|---:|---:|']
        lines += [f'| {r["panel"]} | {r["component"]} | {r["E158"]:.9f} | {r["E159"]:.9f} | {r["delta_E159_minus_E158"]:+.9f} |' for r in rows]
        lines+=['','差值小于0代表E159较低。position/shape和overlap_total为原始诊断，带weighted后缀为已计系数贡献。']
    else:
        lines+=['成员或登记Loss配置不同，已保留元数据，不输出混合口径的数值优劣。']
    lines+=['','| 项目 | E158 | E159 |','|---|---:|---:|',
        f'| best epoch | {baseline["frozen"]["best_epoch"]} | {current["frozen"]["best_epoch"]} |',
        f'| 完成轮数 | {baseline["result"]["epochs"]} | {current["result"]["epochs"]} |',
        f'| 接受更新数 | {baseline["result"]["steps"]} | {current["result"]["steps"]} |',
        f'| 累计秒数 | {baseline["result"]["elapsed_seconds"]:.3f} | {current["result"]["elapsed_seconds"]:.3f} |','',
        '同预算下完成轮数可能不同；本轮为单seed宽度对照，无后段或后处理质量结论。']
    save(out/'summary.json',summary)
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'[LOSS_COMPARISON] status={summary["status"]} E158_best={baseline["frozen"]["best_epoch"]} E159_best={current["frozen"]["best_epoch"]} report={out/"report.md"}',flush=True)
    return summary

if __name__=='__main__':compare()
