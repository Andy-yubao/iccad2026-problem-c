"""Read saved solver traces and scores; no solver, scoring or repair invocation."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import itertools
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
PATHS={'E139':ROOT/'experiments/E139_delayed_overlap/runs/e119_delayed_overlap_seed0', 'E140':HERE.parent}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def trace_features(row):
    main=[r for r in row['trace'] if r['phase'] in ('shape','extension_shape','early_projection','project','periodic_project')]
    shape=[r for r in main if r['phase'] in ('shape','extension_shape')]
    def flatten(trace):
        output=[]
        for stage in trace:
            output.append(stage)
            output.extend(flatten(stage.get('trace',[])))
        return output
    all_stages=flatten(row['trace'])
    all_shape=[r for r in all_stages if r['phase'] in ('shape','extension_shape')]
    all_project=[r for r in all_stages if r['phase'] in ('project','early_projection','periodic_project')]
    good=[i for i,r in enumerate(main) if r.get('legal')]
    return dict(main_legal_candidates=sum(bool(r.get('legal')) for r in main),
        main_shape_legal=sum(bool(r.get('legal')) for r in shape),main_shape_calls=len(shape),
        main_all_rejected=not bool(good),first_main_legal_index=good[0]+1 if good else None,
        main_projection_infeasible=sum(r.get('status')=='PrimalInfeasible' for r in main if 'project' in r['phase']),
        main_projection_calls=sum('project' in r['phase'] for r in main),
        rescue_seeds=row['work']['rescue_seeds'],packed_refine_count=sum(r['phase']=='full_packed_refine' for r in row['trace']),
        all_shape_calls=len(all_shape),all_shape_iterations=sum(r.get('iterations',0) for r in all_shape),
        all_projection_calls=len(all_project),all_projection_infeasible=sum(r.get('status')=='PrimalInfeasible' for r in all_project),
        polish_accepted=sum(r.get('accepted',0) for r in row['trace'] if r['phase']=='certified_rigid_moves'))


def main():
    rows={m:[read(p/f'evaluation_no_mib_dev500/cases/{i:04d}.json') for i in range(500)] for m,p in PATHS.items()}
    assert all(a['member']==b['member'] for a,b in zip(rows['E139'],rows['E140']))
    ns=np.array([r['n'] for r in rows['E139']]);weight=np.exp((ns-ns.max())/12);weight=.5*(weight/weight.sum()+1/500)
    q={m:np.array([r['metrics']['cost'] for r in rs]) for m,rs in rows.items()}
    records=[]
    for i in range(500):
        rec=dict(index=i,member=rows['E139'][i]['member'],n=int(ns[i]),Q_delta=float(q['E140'][i]-q['E139'][i]),
            mixed_Q_contribution=float(weight[i]*(q['E140'][i]-q['E139'][i])),models={})
        for model in PATHS:
            r=rows[model][i]
            rec['models'][model]=dict(trace=trace_features(r),raw_metrics=r['raw_metrics'],final_metrics=r['metrics'],
                hpwl_repair_factor=r['metrics']['hpwl_total']/r['raw_metrics']['hpwl_total'],
                bbox_repair_factor=r['metrics']['bbox_area']/r['raw_metrics']['bbox_area'])
        records.append(rec)
    save(HERE/'postprocessing_cases.json',records)
    result=dict(models={},route_groups={},tail_sensitivity={},score_attribution={})
    for model in PATHS:
        r=[v['models'][model]['trace'] for v in records]
        keys=[k for k in r[0] if k!='first_main_legal_index']
        result['models'][model]={k:dict(sum=float(sum(v[k] for v in r)),mean=float(np.mean([v[k] for v in r]))) for k in keys}
        result['models'][model]['rescue_cases']=sum(v['rescue_seeds']>0 for v in r)
        result['models'][model]['projection_infeasible_cases']=sum(v['all_projection_infeasible']>0 for v in r)
    for label,mask in [('both_main_success',np.array([not r['models']['E139']['trace']['main_all_rejected'] and not r['models']['E140']['trace']['main_all_rejected'] for r in records])),
        ('new_E140_main_failure',np.array([not r['models']['E139']['trace']['main_all_rejected'] and r['models']['E140']['trace']['main_all_rejected'] for r in records])),
        ('E140_fixed_main_failure',np.array([r['models']['E139']['trace']['main_all_rejected'] and not r['models']['E140']['trace']['main_all_rejected'] for r in records])),
        ('both_main_failure',np.array([r['models']['E139']['trace']['main_all_rejected'] and r['models']['E140']['trace']['main_all_rejected'] for r in records]))]:
        dq=q['E140']-q['E139']
        result['route_groups'][label]=dict(cases=int(mask.sum()),indices=np.flatnonzero(mask).tolist(),
            mean_Q_delta=float(dq[mask].mean()) if mask.any() else None,
            mixed_Q_contribution=float(weight[mask]@dq[mask]))
    def vector(model):
        return np.array([[max(0,r['metrics']['hpwl_gap']),max(0,r['metrics']['area_gap']),
            r['metrics']['grouping_violations']/max(1,r['metrics']['max_possible_violations']),
            r['metrics']['boundary_violations']/max(1,r['metrics']['max_possible_violations'])] for r in rows[model]])
    def formula(x):return np.minimum((1+.5*(x[:,0]+x[:,1]))*np.exp(2*(x[:,2]+x[:,3])),9.999999)
    a,b=vector('E139'),vector('E140');attr=np.zeros_like(a)
    np.testing.assert_allclose(formula(a),q['E139'],atol=1e-12,rtol=0);np.testing.assert_allclose(formula(b),q['E140'],atol=1e-12,rtol=0)
    for order in itertools.permutations(range(4)):
        v=a.copy();last=formula(v)
        for j in order:
            v[:,j]=b[:,j];now=formula(v);attr[:,j]+=(now-last)/24;last=now
    np.testing.assert_allclose(attr.sum(1),q['E140']-q['E139'],atol=1e-12,rtol=0)
    result['score_attribution']={k:dict(mean_Q=float(attr[:,j].mean()),mixed_Q=float(weight@attr[:,j])) for j,k in enumerate(('hpwl','area','grouping','boundary'))}
    for label,excluded in [('all',[]),('without_dev358',[358])]:
        mask=~np.isin(np.arange(500),excluded);w=np.exp((ns[mask]-ns[mask].max())/12);w=.5*(w/w.sum()+1/mask.sum())
        x,y=w@q['E139'][mask],w@q['E140'][mask]
        result['tail_sensitivity'][label]=dict(count=int(mask.sum()),E139=float(x),E140=float(y),delta=float(y-x))
    order=np.argsort([r['mixed_Q_contribution'] for r in records])
    result['largest_worse']=[records[int(i)] for i in order[-8:][::-1]]
    result['largest_better']=[records[int(i)] for i in order[:8]]
    result['saved_score_reconstruction_exact']=True
    result['new_postprocessing_calls']=0
    save(HERE/'postprocessing_summary.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('largest_worse','largest_better')},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
