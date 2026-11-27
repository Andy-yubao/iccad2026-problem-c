"""Consume saved evidence only: common axes, model switch, fixed train5/worst5."""
import json
from pathlib import Path
import numpy as np
from matplotlib import colormaps
from datasets.paths import ROOT
from visualization.layouts import plot_case

FOLDER=ROOT/'visualization/E117_node26/20261109_vs_E114'
INLINE=Path('C:/Users/yuche/.codex/visualizations/2026/11/09/01a1be1b-4138-7f32-a467-0f3055120961/node26-e114-comparison.html')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def failures(rectangles,constraints):
    q=np.asarray(rectangles);lo=q[:,:2].min(0);hi=(q[:,:2]+q[:,2:]).max(0)
    d=np.stack((abs(q[:,0]-lo[0]),abs(q[:,0]+q[:,2]-hi[0]),abs(q[:,1]+q[:,3]-hi[1]),abs(q[:,1]-lo[1])),1)
    return [[bit for bit,distance in zip((1,2,4,8),row) if flags[4]&bit and distance>=1e-6] for row,flags in zip(d,constraints)]

def main():
    owners={'e114':ROOT/'experiments/E114_supervision_placement','node26':ROOT/'experiments/E117_node26'}
    runs={k:p/('runs/staged_seed0' if k=='e114' else 'runs/node26_seed0') for k,p in owners.items()}
    result={k:read(p/'evaluation_no_mib_dev500/results.json') for k,p in runs.items()}
    evidence={k:read(FOLDER/('e114_evidence.json' if k=='e114' else 'node26_evidence.json')) for k in owners}
    assert (owners['e114']/'evaluation_staged/budget.json').read_bytes()==(owners['node26']/'evaluation/budget.json').read_bytes()
    assert (owners['e114']/'evaluation_staged/solver.py').read_bytes()==(owners['node26']/'evaluation/solver.py').read_bytes()
    cases=[]
    for a,b in zip(evidence['e114']['cases'],evidence['node26']['cases']):
        assert a['member']==b['member'] and a['n']==b['n'] and a['group']==b['group']
        np.testing.assert_allclose(a['reference'],b['reference'],rtol=1e-6,atol=1e-5)
        constraints=[list(f[:2])+[0]+list(f[3:]) for f in a['constraints']]
        assert constraints==[list(f[:2])+[0]+list(f[3:]) for f in b['constraints']]
        models={}
        for key,case in [('e114',a),('node26',b)]:
            assert len(case['layers'])==12
            models[key]={n:case[n] for n in ('layers','final','coverage','Q','boundary','grouping')}
            models[key]['boundary_status']=dict(layers=[failures(p,constraints) for p in case['layers']],final=failures(case['final'],constraints))
        all_rect=np.concatenate([a['reference'],*[p for c in (a,b) for p in [c['final'],*c['layers']]]])
        lo=all_rect[:,:2].min(0);hi=(all_rect[:,:2]+all_rect[:,2:]).max(0);mid=(lo+hi)/2;span=max(hi-lo)*1.08
        groups=sorted({f[3] for f in constraints if f[3]>0})
        cases.append(dict(group=a['group'],label=a['label'],member=a['member'],n=a['n'],reference=a['reference'],constraints=constraints,
            evaluation_index=a['evaluation_index'],models=models,bounds=[float(mid[0]-span/2),float(mid[0]+span/2),float(mid[1]-span/2),float(mid[1]+span/2)],
            reference_boundary=failures(a['reference'],constraints),
            group_colors={str(g):[int(round(x*255)) for x in colormaps['Set3'](i%12)[:3]] for i,g in enumerate(groups)}))
    steps={k:[json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()] for k,p in runs.items()}
    medians={k:float(np.median([r['seconds_per_step'] for r in rows])*1000) for k,rows in steps.items()}
    metrics=[]
    def metric(label,key,fmt):metrics.append(dict(label=label,**{k:fmt(result[k][key]) for k in owners}))
    metric('末层 dev Loss','common_final_dev_loss',lambda x:f'{x:.6f}')
    metric('4+12 选模 Loss','own_selection_loss',lambda x:f'{x:.6f}')
    metric('最终混合 Q ↓','mixed_Q',lambda x:f'{x:.6f}')
    metric('草稿重复覆盖 ↓','raw_repeated_coverage',lambda x:f'{x*100:.3f}%')
    metric('完整耗时 · 均值','mean_complete_seconds',lambda x:f'{x*1000:.1f} ms')
    metric('完整耗时 · P95','p95_complete_seconds',lambda x:f'{x*1000:.1f} ms')
    metrics.append(dict(label='训练更新 · 中位数',**{k:f'{medians[k]:.1f} ms' for k in owners}))
    metric('最终硬合法','hard_legal',lambda x:f'{x} / 500')
    data=dict(models={k:dict(label='E114 4+12' if k=='e114' else 'Node26 4+12',best_epoch=result[k]['best_epoch'],checkpoint_sha256=result[k]['checkpoint_sha256']) for k in owners},metrics=metrics,cases=cases)
    save(FOLDER/'data.json',data)
    def rounded(v):
        if isinstance(v,float):return round(v,5)
        if isinstance(v,list):return [rounded(x) for x in v]
        if isinstance(v,dict):return {k:rounded(x) for k,x in v.items()}
        return v
    template=(ROOT/'visualization/node26_comparison_template.html').read_text(encoding='utf-8')
    fragment=template.replace('__DATA__',json.dumps(rounded(data),ensure_ascii=False,separators=(',',':')).replace('<','\\u003c'))
    assert len(fragment.encode('utf-8'))<1_000_000
    INLINE.write_text(fragment,encoding='utf-8')
    (FOLDER/'viewer-fragment.html').write_text(fragment,encoding='utf-8')
    shell='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>E114 / Node26</title><style>:root{--font-size-base:15px;--foreground:#182331;--background:#fff;--muted:#e5e7eb;--muted-foreground:#526174;--border:#b6c1cc;--red:#c8223a;--green:#15803d}body{font:15px/1.5 system-ui;margin:24px}h2,h3,p{margin:0}.viz-controls,.viz-row{display:flex;align-items:center;flex-wrap:wrap;gap:16px}.form-label{display:grid;gap:5px}select{font:inherit;padding:6px;max-width:100%}.form-range{flex:1;min-width:160px}.form-check{display:flex;align-items:center;gap:6px}.table{width:100%;border-collapse:collapse}.table th,.table td{padding:5px 8px;border-bottom:1px solid var(--border);font-weight:400}.text-end{text-align:right}.table-responsive{overflow-x:auto}.text-small{font-size:12px}.tabular-nums{font-variant-numeric:tabular-nums}</style>'
    (FOLDER/'index.html').write_text(shell+fragment+'</html>',encoding='utf-8')
    (FOLDER/'syntax-check.js').write_text(fragment.split('<script>',1)[1].split('</script>',1)[0],encoding='utf-8')
    ar=[read(p) for p in sorted((runs['e114']/'evaluation_no_mib_dev500/cases').glob('*.json'))]
    br=[read(p) for p in sorted((runs['node26']/'evaluation_no_mib_dev500/cases').glob('*.json'))]
    assert len(ar)==len(br)==500 and all(x['member']==y['member'] for x,y in zip(ar,br))
    differences=np.array([y['metrics']['cost']-x['metrics']['cost'] for x,y in zip(ar,br)])
    summary=dict(results=result,training_update_median_ms=medians,paired_500=dict(improved=int((differences< -1e-12).sum()),worse=int((differences>1e-12).sum()),tied=int((abs(differences)<=1e-12).sum())),
        postprocessor='identical E98.depth_budget.v1; NOT the separate E1143x budget',test_read=False,
        evidence_counts={k:{n:v for n,v in e.items() if n.endswith('_count') or n.endswith('_reused')} for k,e in evidence.items()},inline_bytes=len(fragment.encode('utf-8')))
    save(FOLDER/'comparison.json',summary)
    report=['# Node26 与 E114 4+12 对比','',
        '两组均为seed0、36k、12层64维4头、.2L4+.8L12、batch32/FP32、原面积/无MIB/全模块预测；原固定dev500与相同E98.depth_budget.v1后处理。各自唯一Loss-best先冻结，未按Q重选。这里比较的是原预算，不是E114三倍预算结果。','',
        '| 指标 | E114 best48 | Node26 best64 |','|---|---:|---:|']
    for row in metrics:report.append(f"| {row['label']} | {row['e114']} | {row['node26']} |")
    for label,key in [('加权Q','weighted_Q'),('平均Q','mean_Q'),('最大Q','max_Q'),('P95 Q','p95_Q')]:report.append(f'| {label} | {result["e114"][key]:.6f} | {result["node26"][key]:.6f} |')
    report += ['| 草稿硬合法 | 0/500 | 0/500 |','| 行回退 | 0 | 0 |','| 超过250ms | 119/500 | 138/500 |',
        '| 参数 | 612307 | 647155 |','| 完成epoch / best | 49 / 48 | 64 / 64 |','| 更新步数 | 55125 | 72000 |',
        f'| 累计训练分钟 | {result["e114"]["training_seconds"]/60:.2f} | {result["node26"]["training_seconds"]/60:.2f} |','',
        '本次Node26末层Loss高4.154%，混合Q高0.007951616（0.766%）；训练更新中位数低38.82%，但完整逐例均时高19.86%。逐例Q：147例改善、211例变差、142例相同。草稿重复覆盖增加1.495个百分点。','',
        '结论：本次节点几何Q/K方案支持训练更新更快的工程观察，尚未保持E114草稿和最终质量。推理/后处理端没有观察到整体加速。E114研究基线与E101交付不变。','',
        '限制：单seed与反复曝光dev；近似架构对照但参数不等，实际训练epoch/步数不等；运行时段不同，耗时不是同场锁定环境的因果测速。累计训练时间含数据准备/缓存及已记录中断预算，不能当作纯网络算力。两组early_stop只表示已登记dev调度的平台；Node26未观察到train平台，best64仍在末轮，不能称充分拟合或擅自续训。','',
        '展示：同五个固定训练成员＋按Node26最终Q降序/index破同分的最差五个dev成员；两模型共同坐标轴、模块编号与G颜色，1–12真实层输出，右栏固定各自最终解。E114复用五训练例与两个dev逐层缓存，仅补三个dev；Node26复用五训练例层缓存，补五dev逐层推理，仅五训练例新增后处理。全部dev最终坐标复用既有500评价，无新增整套评价/test读取。','',
        '[交互对比](index.html) · [完整对比数据](comparison.json)','']
    (FOLDER/'report.md').write_text('\n'.join(report),encoding='utf-8')
    for key in ('e114','node26'):
        c=cases[5];v=c['models'][key]
        variants={}
        for name,positions,status in [('reference',c['reference'],c['reference_boundary']),('raw',v['layers'][-1],v['boundary_status']['layers'][-1]),('final',v['final'],v['boundary_status']['final'])]:
            variants[name]=dict(positions=positions,soft_details=dict(boundary=[dict(block=i,failed_bits=bits) for i,bits in enumerate(status)]))
        plot_case(dict(panel=key,case_id=str(c['member']),constraints=c['constraints'],axis_limits=c['bounds'],variants=variants,
            caption='Same member, module IDs, grouping colors and axes across both models and all twelve layers.'),FOLDER/f'{key}_worst01.png',
            stages=(('reference','Reference'),('raw',key+' layer12'),('final',key+' postprocessed')),show_metrics=False)
    print('Complete:',INLINE,'bytes',summary['inline_bytes'],flush=True)

if __name__=='__main__':main()
