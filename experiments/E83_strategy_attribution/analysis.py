"""Preregistered saved-only comparisons, with worker-paired family intervals."""
import numpy as np
from .io import HERE,ROOT,save,sha
from .prepare import read

OLD={
 'E42':ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0',
 'E74':ROOT/'experiments/E74_collision_32k_posttrain/runs/20261029_e42_topology_seed0',
 'E81':ROOT/'experiments/E81_multisource_refinement/runs/20261029_e42_multisource_seed0'}

def paired(a,b):
    assert [r['member'] for r in a]==[r['member'] for r in b]
    qa=np.array([r['metrics']['cost'] for r in a]);qb=np.array([r['metrics']['cost'] for r in b]);d=qa-qb
    n=np.array([r['n'] for r in a]);w=np.exp((n-n.max())/12);worker=np.array([r['member'][0].split('/')[0] for r in a]);groups=np.unique(worker)
    numer=np.array([np.sum(w[worker==g]*d[worker==g]) for g in groups]);denom=np.array([w[worker==g].sum() for g in groups])
    ids=np.random.default_rng(20261029).integers(0,len(groups),(4000,len(groups)));samples=numer[ids].sum(1)/denom[ids].sum(1)
    family=np.quantile(samples,[.05/(2*3),1-.05/(2*3)]).tolist()
    return {'count':len(a),'weighted_Q_delta':float(np.average(d,weights=w)),'worker95':np.quantile(samples,[.025,.975]).tolist(),
      'worker_Bonferroni_98p333':family,'strong_Q_margin_pass':bool(family[1]<-.001),
      'better_equal_worse':[int((d< -1e-8).sum()),int((abs(d)<=1e-8).sum()),int((d>1e-8).sum())]}

def geometry(p):
    p=np.asarray(p);hi=p[:,:2]+p[:,2:];bbox=float(np.prod(hi.max(0)-p[:,:2].min(0)))
    xs=np.unique(np.r_[p[:,0],hi[:,0]]);union=0.
    for l,r in zip(xs[:-1],xs[1:]):
        ix=(p[:,0]<r)&(hi[:,0]>l)
        intervals=sorted(zip(p[ix,1],hi[ix,1]));height=0.;end=-np.inf
        for low,high in intervals:height+=max(0.,high-max(low,end));end=max(end,high)
        union+=(r-l)*height
    return {'bbox':bbox,'union':float(union),'empty_over_bbox':float((bbox-union)/bbox)}

def main():
    out=HERE/'runs/analysis.json'
    if out.exists():return
    plan=read(HERE/'plan.json');paths={**OLD,**{a['name']:HERE/'runs'/a['name'] for a in plan['arms']}}
    terminals={n:read(p/'terminal/results.json')['records'] for n,p in paths.items()}
    raw={n:read(p/'full_dev.json')['records'] for n,p in paths.items() if n!='E42'}
    raw['E42']=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/forwards.json')['models']['E42']
    features=read(ROOT/'experiments/E69_case_loss_statistics/runs/20261028_dev2000/features.json')['records'];members=[r['member'] for r in features]
    n=np.array([r['n'] for r in features]);pin=np.array([r['pin_connected_fraction'] for r in features]);density=np.array([r['connection_density'] for r in features]);sparse=np.array([density[i]<=np.median(density[n==n[i]]) for i in range(len(n))])
    groups={'all':n>0,'low_pin':pin<=.3,'low_pin_sparse':(pin<=.3)&sparse};lookup={tuple(m):i for i,m in enumerate(members)}
    result={'status':'complete_pending_visual_audit','training_updates':0,'test_read':False,'baseline_promoted':False,
        'comparisons':{},'models':{},'limits':plan['limits'],'method':plan['statistics']}
    for c in plan['primary_comparisons']:
        a,b=c['current'],c['baseline'];result['comparisons'][a+'_vs_'+b]=paired(terminals[a],terminals[b])
    # Source area/frame identity from already audited frozen source inference.
    audit=ROOT/'experiments/E81_multisource_refinement/runs/20261029_placer_corrector_audit'
    source_meta=read(audit/'source_metrics.json');source_arrays=np.load(audit/'source_drafts.npz')
    source_lookup={tuple(m):i for i,m in enumerate(source_meta['members'])}
    for a in plan['arms']:
        name=a['name'];rr=raw[name];tt=terminals[name];assert [r['member'] for r in rr]==members
        overlap=np.array([r['overlap_area_fraction'] for r in rr]);subgroups={}
        for group,ix in groups.items():
            panel=[r for r in tt if ix[lookup[tuple(r['member'])]]]
            weights=np.exp((np.array([r['n'] for r in panel])-max(r['n'] for r in panel))/12)
            subgroups[group]={'raw_count':int(ix.sum()),'raw_overlap':float(overlap[ix].mean()),'panel_count':len(panel),
                'panel_weighted_Q':float(np.average([r['metrics']['cost'] for r in panel],weights=weights))}
        maximum=0.;contain_count=0;geometry_rows=[]
        for r in tt:
            i=source_lookup[tuple(r['member'])];source=source_arrays['E42' if a['placer']=='P42' else 'E74'][i,:r['n']]
            p=np.array(r['draft_positions']);lo=source[:,:2].min(0);hi=(source[:,:2]+source[:,2:]).max(0)
            deviation=max(float((lo-p[:,:2]).max()),float((p[:,:2]+p[:,2:]-hi).max()),0.)/np.sqrt(source_meta['models']['E42'][i]['required_area'])
            maximum=max(maximum,deviation);contain_count+=deviation<=3e-6;geometry_rows.append(geometry(p))
        result['models'][name]={'summary':read(paths[name]/'terminal/results.json')['summary'],'raw_overlap':float(overlap.mean()),
          'raw_hard_legal':sum(r['raw_hard_legal'] for r in rr),'groups':subgroups,
          'frame_containment':{'count':contain_count,'maximum_normalized_excursion':maximum,'tolerance':3e-6},
          'draft_mean_empty_over_bbox':float(np.mean([r['empty_over_bbox'] for r in geometry_rows])),
          'worst5_Q':[{'index':r['index'],'n':r['n'],'Q':r['metrics']['cost'],'rescue':r.get('rescue_used')} for r in sorted(tt,key=lambda r:r['metrics']['cost'])[-5:][::-1]]}
    result['evidence_sha256']={p.relative_to(ROOT).as_posix():sha(p) for p in [*[p/'terminal/results.json' for p in paths.values()],audit/'source_drafts.npz',audit/'source_metrics.json']}
    save(out,result)
    text='# E83策略归因：冻结模块干预\n\n只读配对分析；0训练更新，测试封存，基线未升级。该结果区分部署模块作用，不是重训消融。\n\n'
    text+='| 新组合 | 草稿交面积 | Q | P95 | 最大Q | 救援 | 框内例数 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for name,v in result['models'].items():
        s=v['summary'];text+=f"| {name} | {v['raw_overlap']:.3%} | {s['weighted_Q']:.6f} | {s['p95_Q']:.6f} | {s['max_Q']:.6f} | {s['rescue_cases']} | {v['frame_containment']['count']}/500 |\n"
    text+='\n预登记主对照：差值为前者减后者，负数较好。按来源worker配对重抽4000次；同时报告95%及三对照Bonferroni约98.333%区间，后者上界低于−.001才通过预设实用Q门槛，仍须检查草稿/合法/尾部取舍。\n\n'
    for name,v in result['comparisons'].items():text+=f"- {name}: ΔQ={v['weighted_Q_delta']:+.6f}；95%={v['worker95']}；族区间={v['worker_Bonferroni_98p333']}；胜平负={v['better_equal_worse']}。\n"
    text+='\n局限：旁路已学残差属于部署干预，不等于无残差重训；来源更换也更换预测框，不是规模单因果。统计只涵盖当前开发案例来源，不覆盖训练种子不确定性或自适应选模偏差。详细弱端口/稀疏、精确空白、框、尾部见[analysis.json](analysis.json)。[四阶段图](index.html)。完成后须实际查看3至5图，再形成研究判断。\n'
    (HERE/'runs/report.md').write_text(text,encoding='utf-8');print(text,flush=True)

if __name__=='__main__':main()
