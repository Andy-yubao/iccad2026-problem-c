"""Saved-result comparisons only; never run a solver or select another checkpoint."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ..io import ROOT,HERE,save
from .common import OUTPUT,registration
from .metrics import module_overlap

def paired(current,previous):
    if len(current)!=500 or len(previous)!=500 or any((a['index'],a['member'],a['n'])!=(b['index'],b['member'],b['n']) for a,b in zip(current,previous)):
        raise ValueError('Paired member mismatch')
    n=np.array([r['n'] for r in current]);w=np.exp((n-n.max())/12)
    delta=np.array([a['metrics']['cost']-b['metrics']['cost'] for a,b in zip(current,previous)])
    workers=sorted({r['member'][0].split('/')[0] for r in current});lookup={v:i for i,v in enumerate(workers)}
    group=np.array([lookup[r['member'][0].split('/')[0]] for r in current])
    top=np.bincount(group,weights=w*delta,minlength=len(workers));bottom=np.bincount(group,weights=w,minlength=len(workers))
    sampled=np.random.default_rng(20261031).integers(0,len(workers),(5000,len(workers)))
    boot=top[sampled].sum(1)/bottom[sampled].sum(1)
    return {'weighted_Q_delta':float(np.average(delta,weights=w)),
            'worker_paired_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),
            'improved':int((delta<0).sum()),'worse':int((delta>0).sum())}

def assess(identity):
    reg=registration();data=json.loads((OUTPUT/'terminal/results.json').read_text(encoding='utf-8'));records=data['records']
    def geometry(r):
        if 'draft_positions' not in r:raise ValueError('Missing draft must not disappear from reporting')
        return module_overlap(np.asarray(r['draft_positions'],dtype=float)[:r['n']])
    with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(geometry,records))
    comparisons={}
    for name,source in reg['previous_results'].items():
        old=json.loads((ROOT/source).read_text(encoding='utf-8'))['records']
        comparisons[name]=paired(records,old)
    result={'status':'completed','arm':identity['arm'],'checkpoint_sha256':identity['checkpoint_sha256'],
            'summary':data['summary'],'module_pass10_pooled':1.-sum(r['N10'] for r in values)/sum(r['n'] for r in records),
            'worst_module_coverage_p95':float(np.percentile([r['max_covered_fraction'] for r in values],95)),
            'worst_module_coverage_max':float(max(r['max_covered_fraction'] for r in values)),
            'dev500_loss':json.loads((OUTPUT/'dev500_loss.json').read_text(encoding='utf-8'))['losses'],
            'comparisons':comparisons,'test_read':False,'next_action':'report only; no automatic training or corrector'}
    save(OUTPUT/'geometry500.json',{'records':[dict(index=r['index'],member=r['member'],**g) for r,g in zip(records,values)],'test_read':False})
    save(OUTPUT/'assessment.json',result)
    return {k:v for k,v in result.items() if k!='dev500_loss'}

def combined():
    folders={arm:HERE/f'runs/20261101_{arm.lower()}_seed0/qualification' for arm in ('C','G')}
    values={a:json.loads((p/'assessment.json').read_text(encoding='utf-8')) for a,p in folders.items()}
    records={a:json.loads((p/'terminal/results.json').read_text(encoding='utf-8'))['records'] for a,p in folders.items()}
    result={'status':'completed','arms':values,'G_minus_C':paired(records['G'],records['C']),
            'caveat':'C trained12 epochs and selected10; G trained19 and selected16 under own fixed Loss early stopping. Actual exposure differs.',
            'test_read':False,'neutral_runtime_development_only':True}
    out=HERE/'runs/evaluation';save(out/'results.json',result)
    lines=['# E93固定best开发验证','', '| 模型 | 加权Q | 平均Q | P95 Q | 最大Q | 合法 | 模块覆盖≤10% | 最差模块覆盖P95 | 平均秒 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for a,v in values.items():
        s=v['summary'];lines.append(f"| {a} | {s['weighted_Q']:.9f} | {s['mean_Q']:.9f} | {s['p95_Q']:.6f} | {s['max_Q']:.6f} | {s['hard_legal']}/500 | {v['module_pass10_pooled']:.3%} | {v['worst_module_coverage_p95']:.3%} | {s['seconds_mean']:.4f} |")
    lines+=['','C/G不同总Loss不直接比较；同口径分项及C/F诊断值见各支dev500_loss.json。完整dev2000包含交面积、精确模块覆盖、外框、空白及约束诊断。',
            '', 'E90/E92只读复用。配对Q差与worker区间见results.json；本结果为自适应开发中性时间评价，测试封存。',
            '', result['caveat'], '', '[C同100图](../20261101_c_seed0/qualification/draft_comparison100/index.html) · [G同100图](../20261101_g_seed0/qualification/draft_comparison100/index.html)']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
