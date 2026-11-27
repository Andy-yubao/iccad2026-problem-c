"""Analyze and render saved full500 results. Never run inference or a solver."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json,csv,argparse
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'experiments/E50_anytime_quality/runs/20261025_full500_parallel_v3'
OUT=SOURCE.parent/'20261025_full500_review'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def quant(x):
    x=[v for v in x if v is not None]
    return {'count':len(x),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'max':float(max(x))} if x else {'count':0}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');args=ap.parse_args()
    assert read(SOURCE/'execution.json')=={'complete':True,'count':500,'requested':500}
    from datasets.reference_layouts import ReferenceLayouts
    from experiments.E50_anytime_quality.archive import Archive
    from experiments.E50_anytime_quality.backend.frozen.packing import prepare_problem
    data=ReferenceLayouts('dev_monitor500')
    old={r['index']:r for r in read(ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0/terminal/results.json')['records']}
    OUT.mkdir(exist_ok=True);table=[];frontiers=[];rows={}
    for i in range(500):
        r=read(SOURCE/'cases'/f'{i:04d}.json');rows[i]=r
        item=data[i];assert list(item['member'])==r['member']==old[i]['member']
        p=prepare_problem(*item['case'].solver_inputs);arc=Archive(p,np.asarray(r['draft']),0,None)
        cs=r['solver']['candidates'];s=next(c for c in cs if c['id']==r['solver']['selected_id'])
        od=arc.distance(np.asarray(old[i]['positions']));d=s['distance'];hit=[c for c in cs if c['metrics']['cost']<1.03]
        first=min(hit,key=lambda c:c['checked_seconds']) if hit else None
        best=min(cs,key=lambda c:c['metrics']['cost'])
        traces=r['solver'].get('worker_stats',{});ex=traces.get('explore',{}).get('trace')
        steps=[t for t in (ex or []) if 'status' in t]
        raw=np.asarray(r['draft']);pos=np.asarray(s['positions']);free=~p.preplaced
        local=np.linalg.norm(pos[:,:2]+pos[:,2:]/2-raw[:,:2]-raw[:,2:]/2,axis=1)/np.sqrt(p.areas)
        shapes=np.maximum((pos[:,2]/pos[:,3])/(raw[:,2]/raw[:,3]),(raw[:,2]/raw[:,3])/(pos[:,2]/pos[:,3]))
        row={'index':i,'n':r['n'],'old_Q':r['baseline_Q'],'new_Q':r['selected_metrics']['cost'],
             'delta_Q':r['selected_metrics']['cost']-r['baseline_Q'],'old_D':od['D'],'new_D':d['D'],
             'mean_movement_module_units':d['mean_center_local'],'p95_movement_module_units':d['p95_center_local'],
             'mean_movement_chip_diagonal':d['mean_center_over_chip_diagonal'],
             'shape_log_change':d['mean_abs_log_aspect_change'],'near_relation_change':d['near_relation_change_fraction'],
             'fraction_moved_over_one_module':float((local[free]>1).mean()) if free.any() else 0.,
             'fraction_aspect_changed_over20pct':float((shapes[arc.shape_free]>1.2).mean()) if arc.shape_free.any() else 0.,
             'full_seconds':r['full_seconds'],'candidate_count':len(cs),
             'first_legal_seconds':min(c['checked_seconds'] for c in cs),
             'first_good_seconds':first['checked_seconds'] if first else None,
             'first_good_D':first['distance']['D'] if first else None,
             'min_found_good_D':min((c['distance']['D'] for c in hit),default=None),
             'best_found_Q':best['metrics']['cost'],'best_found_id':best['id'],'selected_id':s['id'],
             'baseline_fresh_Q':next((c['metrics']['cost'] for c in cs if c['id']==r['solver']['baseline_id']),None),
             'selected_source':s['source'],'selected_worker':s['worker'],
             'explore_trace_present':ex is not None,
             'recorded_explore_solves':len(steps) if ex is not None else None,
             'recorded_explore_infeasible':sum('Infeasible' in t['status'] for t in steps) if ex is not None else None,
             'recorded_relation_repairs':sum(t.get('phase')=='infeasible_relation_repair' for t in ex) if ex is not None else None,
             'worker_aborts':len(r['solver']['worker_aborts']),
             'new_area_gap':r['selected_metrics']['area_gap'],'new_wire_gap':r['selected_metrics']['hpwl_gap'],
             'new_soft':r['selected_metrics']['total_soft_violations'],
             'offline_diagnostics_seconds':r['offline_diagnostics_seconds']}
        table.append(row)
        values=np.array([[c['metrics']['cost'],c['distance']['D'],c['checked_seconds']] for c in cs]);front=[]
        for j,c in enumerate(cs):
            if not np.any(np.all(values<=values[j],axis=1)&np.any(values<values[j],axis=1)):
                front.append({'id':c['id'],'Q':c['metrics']['cost'],'D':c['distance']['D'],'checked_seconds':c['checked_seconds']})
        frontiers.append({'index':i,'frontier':front})
        if (i+1)%100==0:print(f'Analyzed {i+1}/500 saved cases',flush=True)
    with (OUT/'paired_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    q=np.array([r['new_Q'] for r in table]);b=np.array([r['old_Q'] for r in table]);n=np.array([r['n'] for r in table]);weights=np.exp((n-n.max())/12)
    worst=sorted(table,key=lambda r:(-r['new_Q'],r['index']))[:10];reserved={r['index'] for r in worst}
    improved=sorted([r for r in table if r['index'] not in reserved and r['new_Q']<1.03],key=lambda r:(r['delta_Q'],r['index']))[:5]
    reserved|={r['index'] for r in improved}
    pool=[r['index'] for r in table if r['new_Q']<1.03 and r['index'] not in reserved]
    normal=np.random.default_rng(20261025).choice(pool,size=20,replace=False).tolist()
    groups=[{'key':'normal','label':'正常结果随机20例','indices':normal},
            {'key':'improved','label':'常规差、新结果好5例','indices':[r['index'] for r in improved]},
            {'key':'worst','label':'新结果Q最差10例','indices':[r['index'] for r in worst]}]
    assert len(set(i for g in groups for i in g['indices']))==35
    regressions=[r for r in table if r['delta_Q']>1e-8]
    result={**read(SOURCE/'summary.json'),'baseline_mean_Q':float(b.mean()),'baseline_p95_Q':float(np.quantile(b,.95)),
            'baseline_max_Q':float(b.max()),'baseline_Q_lt1_03':int(sum(b<1.03)),
            'win_tie_loss':[int(sum(q<b-1e-8)),int(sum(abs(q-b)<=1e-8)),int(sum(q>b+1e-8))],
            'max_regression':max(r['delta_Q'] for r in regressions),
            'regressions_with_nonworse_candidate':sum(r['best_found_Q']<=r['old_Q']+1e-8 for r in regressions),
            'poor_selected_but_good_found':sum(r['new_Q']>=1.03 and r['best_found_Q']<1.03 for r in table),
            'all_with_good_found':sum(r['best_found_Q']<1.03 for r in table),
            'closer_than_ordinary':sum(r['new_D']<r['old_D']-1e-9 for r in table),
            'old_D':quant([r['old_D'] for r in table]),'new_D':quant([r['new_D'] for r in table]),
            'mean_movement_module_units':quant([r['mean_movement_module_units'] for r in table]),
            'p95_movement_module_units':quant([r['p95_movement_module_units'] for r in table]),
            'mean_movement_chip_diagonal':quant([r['mean_movement_chip_diagonal'] for r in table]),
            'shape_log_change':quant([r['shape_log_change'] for r in table]),
            'near_relation_change':quant([r['near_relation_change'] for r in table]),
            'first_legal_seconds':quant([r['first_legal_seconds'] for r in table]),
            'first_good_seconds':quant([r['first_good_seconds'] for r in table]),
            'min_found_good_D':quant([r['min_found_good_D'] for r in table]),
            'first_good_within':{str(t):sum(r['first_good_seconds'] is not None and r['first_good_seconds']<=t for r in table) for t in (.25,.5,1,2,4)},
            'good_found_and_minD_le':{str(t):sum(r['min_found_good_D'] is not None and r['min_found_good_D']<=t for r in table) for t in (.1,.2,.3,.5)},
            'candidate_count':sum(r['candidate_count'] for r in table),
            'trace_missing':sum(not r['explore_trace_present'] for r in table),
            'total_offline_diagnostics_seconds':sum(r['offline_diagnostics_seconds'] for r in table),
            'groups':groups,'regressions':regressions,'worst10':worst,'improved5':improved,
            'selection_seed':20261025,'solver_or_inference_calls':0,
            'selection_policy':'worst10 by selected Q; top5 Q reductions among remaining Q<1.03; random20 among remaining Q<1.03',
            'scope':'adaptive frozen development500; found witnesses only, no global optimality or intrinsic difficulty claim'}
    save(OUT/'analysis.json',result);save(OUT/'frontiers.json',frontiers)
    manifest={'title':'E50：500例结果与35例布局对照','note':'左：原草稿｜中：常规后处理｜右：E50。三列坐标尺度和模块编号完全一致。正常组只代表Q<1.03子集；差例组按新Q排序。','groups':groups,'cases':[]}
    if args.render:
        from visualization.layouts import plot_case,geometry_details
        from visualization.saved_gallery import write_gallery
        (OUT/'images').mkdir(exist_ok=True)
        by_index={r['index']:r for r in table}
        for g in groups:
            for i in g['indices']:
                r=rows[i];s=next(c for c in r['solver']['candidates'] if c['id']==r['solver']['selected_id']);c=np.asarray(data[i]['case'].solver_inputs[5]).tolist();t=by_index[i]
                variants={'raw':{'positions':r['draft']},'old':{'positions':old[i]['positions'],'metrics':old[i]['metrics']},'new':{'positions':s['positions'],'metrics':s['metrics']}}
                for v in variants.values():v['soft_details']=geometry_details(v['positions'],c)
                dest=OUT/'images'/f'case_{i:04d}.png'
                plot=plot_case({'panel':g['key'],'case_id':f'dev index{i}','constraints':c,'variants':variants,
                    'caption':f"Same IDs and scale. Draft movement: mean {t['mean_movement_module_units']:.2f}, P95 {t['p95_movement_module_units']:.2f} module units; D {t['new_D']:.3f}."},dest,
                    stages=(('raw','Frozen E42 draft'),('old','Ordinary postprocessing'),('new','E50 selected')))
                manifest['cases'].append({'id':i,'group':g['key'],'image':dest.relative_to(OUT).as_posix(),'axes_bounds':plot['axes_bounds'],'metrics':t})
                print(f"Rendered {len(manifest['cases'])}/35 index{i}",flush=True)
        save(OUT/'manifest.json',manifest);write_gallery(OUT,manifest)
    print(json.dumps({k:v for k,v in result.items() if k not in ('regressions','worst10','improved5')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
