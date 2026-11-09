"""Fixed500 saved-result geometry, matched previous results and report; no Q selection."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from ..io import ROOT,HERE,save,sha
from .common import OUTPUT,registration,sources
from .metrics import module_overlap,union_area

def paired(current,previous):
    if len(current)!=500 or len(previous)!=500 or any((a['index'],a['member'],a['n'])!=(b['index'],b['member'],b['n']) for a,b in zip(current,previous)):
        raise ValueError('Paired members differ')
    n=np.array([r['n'] for r in current]);w=np.exp((n-n.max())/12);delta=np.array([a['metrics']['cost']-b['metrics']['cost'] for a,b in zip(current,previous)])
    workers=sorted({r['member'][0].split('/')[0] for r in current});lookup={v:i for i,v in enumerate(workers)};group=np.array([lookup[r['member'][0].split('/')[0]] for r in current])
    numerator=np.bincount(group,weights=w*delta,minlength=len(workers));denominator=np.bincount(group,weights=w,minlength=len(workers))
    sampled=np.random.default_rng(20261031).integers(0,len(workers),(5000,len(workers)));boot=numerator[sampled].sum(1)/denominator[sampled].sum(1)
    return {'weighted_Q_delta':float(np.average(delta,weights=w)),'worker_paired_95pct_interval':np.quantile(boot,[.025,.975]).tolist(),
            'improved':int((delta<0).sum()),'worse':int((delta>0).sum())}

def draft_geometry(r):
    if 'draft_positions' not in r:raise ValueError('Missing draft must not disappear from geometry aggregation')
    p=np.asarray(r['draft_positions'],float)[:r['n']];bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    return {**module_overlap(p),'empty_fraction_of_bbox':max(0.,1.-union_area(p)/bbox),
            'overlap_area_fraction':r['draft_diagnostics']['overlap_area_fraction']}

def assess(identity):
    reg=registration();data=json.loads((OUTPUT/'terminal/results.json').read_text(encoding='utf-8'));records=data['records']
    if len(records)!=500 or not all(r['torch_threads']==2 for r in records):raise ValueError('Thread/count drift')
    with ThreadPoolExecutor(max_workers=4) as pool:values=list(pool.map(draft_geometry,records))
    compare={}
    for name,path in reg['previous_results'].items():compare[name]=paired(records,json.loads((ROOT/path).read_text(encoding='utf-8'))['records'])
    monitor=json.loads((OUTPUT/'dev500_loss.json').read_text(encoding='utf-8'))['losses']
    result={'status':'completed','checkpoint_sha256':identity['checkpoint_sha256'],'best_F_epoch':identity['best_epoch'],'summary':data['summary'],
            'module_pass10_pooled':1.-sum(r['N10'] for r in values)/sum(r['n'] for r in records),
            'all_modules_pass10_count':sum(r['all_modules_pass10'] for r in values),
            'worst_module_coverage_p95':float(np.percentile([r['max_covered_fraction'] for r in values],95)),
            'worst_module_coverage_max':float(max(r['max_covered_fraction'] for r in values)),
            'raw500_geometry_mean':{k:float(np.mean([r[k] for r in values])) for k in ('bbox_over_A','empty_fraction_of_bbox','overlap_area_fraction')},
            'dev500_loss':{k:v for k,v in monitor.items() if k not in ('per_case_total','per_case_output_max')},
            'comparisons':compare,'test_read':False,'neutral_runtime_development_only':True,'next_action':'report only; no Q-based reselect or further training'}
    save(OUTPUT/'geometry500.json',{'records':[dict(index=r['index'],member=r['member'],**v) for r,v in zip(records,values)],'test_read':False})
    gallery=OUTPUT/'draft_comparison100';manifest=json.loads((gallery/'manifest.json').read_text(encoding='utf-8'))
    if len(manifest['cases'])!=100 or any(sha(gallery/r['image'])!=r['image_sha256'] for r in manifest['cases']):raise ValueError('Gallery integrity failure')
    if json.loads((OUTPUT/'frozen.json').read_text(encoding='utf-8'))['sources']!=sources() or sha(ROOT/identity['checkpoint'])!=identity['checkpoint_sha256']:raise ValueError('Frozen model/source drift')
    result['integrity']={'verified_images':100,'worker_pids':sorted({r['worker_pid'] for r in records}),'threads_per_worker':2,'source_checkpoint_unchanged':True}
    save(OUTPUT/'assessment.json',result)
    s=data['summary'];lines=['# E94后训练LR重置1e-3：固定best500例验证','',
        f"F best{identity['best_epoch']}，dev500固定Loss {monitor['total']:.9f}。加权Q {s['weighted_Q']:.9f}，算术Q {s['mean_Q']:.9f}，硬合法{s['hard_legal']}/500。",
        '',f"Q P95 {s['p95_Q']:.6f}，最大{s['max_Q']:.6f}，最差5%均值{s['worst5pct_mean_Q']:.6f}；平均{s['seconds_mean']:.4f}秒，P95耗时{s['seconds_p95']:.4f}秒。",
        '',f"模块覆盖≤10%达标率{result['module_pass10_pooled']:.3%}，全部模块达标{result['all_modules_pass10_count']}/500；最差模块覆盖P95 {result['worst_module_coverage_p95']:.3%}，最大{result['worst_module_coverage_max']:.3%}。",
        '', '| 旧结果（只读） | E94−旧加权Q | worker配对95%区间 |','|---|---:|---|']
    for name,v in compare.items():lines.append(f"| {name} | {v['weighted_Q_delta']:+.9f} | {v['worker_paired_95pct_interval']} |")
    lines+=['', '草稿交面积比例、外框/模块面积、精确空白/外框：'+json.dumps(result['raw500_geometry_mean'],ensure_ascii=False),
            '', '本次同时改变整套LR日程及训练上限；实际更新轮数须与E93 C分别报告，不将终末Q用于反选checkpoint。不宣称统计等价或正式未见比赛成绩，测试封存。',
            '', '[固定100例reference/E94/E90草稿图](draft_comparison100/index.html)；分项、约束与完整证据见assessment.json和terminal/results.json。']
    (OUTPUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
