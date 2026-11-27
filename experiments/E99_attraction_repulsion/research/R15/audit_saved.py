"""Read each historical operation record; no inference, force replay or solve."""
import csv,hashlib,json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;EXPERIMENT=HERE.parents[1]
OUT=EXPERIMENT/'runs/research/20261103_R15_historical_operation_audit'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def describe(values):
    x=np.asarray(values,float)
    return dict(count=len(x),mean=float(x.mean()),median=float(np.median(x)),p05=float(np.percentile(x,5)),p95=float(np.percentile(x,95)),max=float(x.max()))

def main():
    if OUT.exists():raise FileExistsError('Historical audit evidence protected')
    old=EXPERIMENT/'runs/20261102_deform_v3_dev100/results.json'
    twenty=EXPERIMENT/'runs/tuning/20261102_230903_relaxed20_move40_shape50_7ed2f2bd4d'
    v3=read(old)['records'];runs={'v3_shape10':[(r['index'],r['deformation']['relaxation']) for r in v3],
        'v3_translate10':[(r['index'],r['translation']['relaxation']) for r in v3],
        'relaxed_shape20':[(i,read(twenty/'cases'/f'{i:04d}.json')['relaxation']) for i in range(100)]}
    summary={};operation_rows=[];cases=[]
    for name,records in runs.items():
        all_steps=[];frozen=[];budget_hits=[];shape_hits=[];near_frozen=[]
        for index,r in records:
            cfg=r['config'];steps=r['trace'];all_steps.extend(steps)
            zero=[t['iteration'] for t in steps if t['translation_safety_scale']<=1e-10]
            small=[t['iteration'] for t in steps if t['translation_safety_scale']<.1]
            if zero:frozen.append(index)
            if small:near_frozen.append(index)
            if r['max_center_path_fraction']>=cfg['total_fraction']-1e-9:budget_hits.append(index)
            if r['max_shape_path_log']>=np.log1p(cfg['shape_total_fraction'])-1e-9:shape_hits.append(index)
            cases.append(dict(run=name,index=index,first_frozen_round=min(zero) if zero else None,
                frozen_rounds=len(zero),path_budget_hit=index in budget_hits,shape_budget_hit=index in shape_hits,
                mean_center_displacement_fraction=r['mean_center_displacement_fraction'],
                max_center_path_fraction=r['max_center_path_fraction'],max_dimension_factor=r['max_dimension_factor']))
            operation_rows.extend(dict(run=name,index=index,**t) for t in steps)
        per_round=[]
        for iteration in range(1,records[0][1]['config']['iterations']+1):
            selected=[t for _,r in records for t in r['trace'] if t['iteration']==iteration]
            if selected:per_round.append(dict(iteration=iteration,case_count=len(selected),
                alpha_median=float(np.median([t['translation_safety_scale'] for t in selected])),
                alpha_under01=sum(t['translation_safety_scale']<.1 for t in selected),
                alpha_zero=sum(t['translation_safety_scale']<=1e-10 for t in selected),
                max_step_fraction_median=float(np.median([t['max_center_step_fraction'] for t in selected]))))
        summary[name]=dict(cases=100,operations=len(all_steps),cases_global_freeze=len(frozen),global_freeze_ids=frozen,
            cases_nearly_frozen=len(near_frozen),cases_path_budget_hit=len(budget_hits),cases_shape_budget_hit=len(shape_hits),
            alpha=describe([t['translation_safety_scale'] for t in all_steps]),
            damping=describe([t['damping'] for t in all_steps]),
            mean_center_displacement_fraction=describe([r['mean_center_displacement_fraction'] for _,r in records]),
            max_dimension_factor=describe([r['max_dimension_factor'] for _,r in records]),
            max_shape_step_log=describe([t['max_shape_step_log'] for t in all_steps]),per_round=per_round)
    OUT.mkdir(parents=True)
    sources={str(old.relative_to(EXPERIMENT)):sha(old),**{str((twenty/'cases'/f'{i:04d}.json').relative_to(EXPERIMENT)):sha(twenty/'cases'/f'{i:04d}.json') for i in range(100)}}
    result=dict(status='complete',summary=summary,cases=cases,source_sha256=sources,all_operations=operation_rows,
        scope='historical exposed dev100 read-only; no new coefficient/solve/evaluation',no_new_solve=True,test_read=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    keys=sorted(set().union(*(r.keys() for r in operation_rows)))
    with (OUT/'operations.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(operation_rows)
    lines=['# 原吸引排斥逐轮历史日志审计','', '全部读取已保存V3形变、V3平移及20轮放宽版本的每次操作记录；不重复求解、不在dev改系数。','',
        '| 版本 | 操作记录 | 出现整图位移冻结的例数 | 触及移动总预算 | 触及形变总预算 | 平均中心位移/短边的中位数 |','|---|---:|---:|---:|---:|---:|']
    for name,s in summary.items():lines.append(f"| {name} | {s['operations']} | {s['cases_global_freeze']}/100 | {s['cases_path_budget_hit']}/100 | {s['cases_shape_budget_hit']}/100 | {s['mean_center_displacement_fraction']['median']:.3%} |")
    lines+=['', '冻结指共用安全缩放alpha<=1e-10，不是所有模块都完成了整理。旧日志只有最大值/汇总，没有限制模块ID、每模块力和剩余预算，因此需要训练来源轨迹重放才能区分真正受力小、局部限幅和全局瓶颈。','', '原式吸引峰值ell/54、排斥饱和ell；这只是公式事实，不能据此直接断言提高引力会提升最终Q。大模块的惯性质量与吸引源强度应分开，并保持成对力相反、有限作用范围与受控位移。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({name:{k:v for k,v in s.items() if k not in ('global_freeze_ids','per_round')} for name,s in summary.items()},indent=2))

if __name__=='__main__':main()
