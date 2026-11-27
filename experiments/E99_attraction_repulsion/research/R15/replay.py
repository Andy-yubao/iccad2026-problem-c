"""Registered training-source per-module trajectory audit, no planner solves."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,csv,datetime,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from datasets.paths import ROOT
from datasets.reference_layouts import registry
from ...tune import process_lock
from ...diagnostics import measures
from ...deformation import preprocess as original_preprocess
from ...tuning_config import Parameters
from .operator import preprocess,variants,MODULE_COLUMNS,ROUND_COLUMNS

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;EXPERIMENT=RESEARCH.parent
PLAN=json.loads((HERE/'replay_plan.json').read_text(encoding='utf-8'))
CONTROL=EXPERIMENT/'runs/research/20261102_frozen_C3_train500_control'
OUT=EXPERIMENT/PLAN['output'];PREFLIGHT=EXPERIMENT/PLAN['preflight_output']


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp')
    t.write_text(json.dumps(x,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(5):
        try:t.replace(p);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.02*(attempt+1))

def sources():
    paths=[HERE/s for s in ('operator.py','replay.py','replay_plan.json')]
    paths += [EXPERIMENT/s for s in ('deformation.py','relaxation.py','tuning_config.py','diagnostics.py','tune.py','planner_config.json')]
    paths += list(EXPERIMENT.joinpath('planner').rglob('*.py'))
    paths += [ROOT/s for s in ('datasets/paths.py','datasets/reference_layouts.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}

def check_planner():
    old=read(CONTROL/'registration.json')['identity']['sources'];now=sources()
    selected={k:v for k,v in old.items() if k.startswith('experiments/E99_attraction_repulsion/planner/') or k=='experiments/E99_attraction_repulsion/planner_config.json'}
    assert selected and all(now[k]==v for k,v in selected.items()),'Original full planner changed'
    return selected

def cached(index):
    r=read(CONTROL/'cases'/f'{index:04d}.json');p=CONTROL/'coordinates'/f'{index:04d}.npz';assert sha(p)==r['coordinates_sha256']
    with np.load(p) as a:
        inputs=(r['n'],*[a[k].copy() for k in ('areas','b2b','p2b','pins','constraints','specs')]);draft=a['positions'].copy()
    return draft,inputs,r

def stopped(out):return (RESEARCH/'STOP').exists() or (out/'STOP').exists()

def register(out):
    identity=dict(plan=PLAN,sources=sources(),planner_frozen=check_planner(),control_registration_sha256=sha(CONTROL/'registration.json'),
        registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),checkpoint_sha256=read(CONTROL/'registration.json')['identity']['checkpoint_sha256'],
        configs={k:__import__('dataclasses').asdict(v) for k,v in variants().items()})
    if (out/'registration.json').exists():assert read(out/'registration.json')['identity']==identity
    else:
        out.mkdir(parents=True,exist_ok=True);save(out/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in identity['sources']:z.write(ROOT/p,p)
    return identity

def preflight():
    if PREFLIGHT.exists():raise FileExistsError('Existing preflight protected')
    identity=register(PREFLIGHT);records=[]
    with process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1);draft,inputs,_=cached(0);begin=time.perf_counter();preprocess(draft,inputs,capture=True);cold=time.perf_counter()-begin
        for index in PLAN['preflight']['indices']:
            draft,inputs,r=cached(index)
            expected,old=original_preprocess(draft,inputs,config=Parameters(),return_history=True)
            for name,cfg in variants().items():
                final,log=preprocess(draft,inputs,cfg,capture=True);plain,_=preprocess(draft,inputs,cfg,capture=False)
                np.testing.assert_array_equal(final,plain)
                if name=='baseline20':
                    np.testing.assert_allclose(final,expected,rtol=2e-12,atol=2e-10)
                    np.testing.assert_allclose(log['history'],old['history'],rtol=2e-12,atol=2e-10)
                records.append(dict(index=index,member=r['member'],arm=name,baseline_max_abs_difference=float(np.abs(final-expected).max()) if name=='baseline20' else None,
                    capture_and_production_bit_exact=True,operator_stats={k:v for k,v in log.items() if not isinstance(v,np.ndarray)}))
    assert sources()==identity['sources'];save(PREFLIGHT/'results.json',dict(status='passed',unique_cases=5,records=records,native_cold_seconds=cold,source_sha256=identity['sources'],test_read=False))
    print(json.dumps(dict(status='passed',unique_cases=5,cold_seconds=cold,baseline_differences=[r['baseline_max_abs_difference'] for r in records if r['arm']=='baseline20'])))

def diagnose(draft,inputs,log):
    x=log['history'];m=log['module_log'];rounds=log['round_log'];n=len(draft);local=draft[:,2:].min(1);area=draft[:,2:].prod(1);mass=area/np.median(area)
    c=np.asarray(inputs[5]);movable=c[:,1]==0;soft=~(c[:,:2]!=0).any(1)
    i,j=np.triu_indices(n,1);lo=x[:,:,:2];hi=lo+x[:,:,2:]
    ov=np.minimum(hi[:,i],hi[:,j])-np.maximum(lo[:,i],lo[:,j]);inter=np.maximum(ov,0).prod(2).sum(1)/area.sum()
    bbox=(hi.max(1)-lo.min(1)).prod(1);steps=[]
    previous_path=np.vstack((np.zeros((1,n)),m[:-1,:,12]));remaining=log['config']['total_fraction']*local[None]-previous_path
    raw=np.linalg.norm(m[:,:,6:8],axis=2);actual=np.linalg.norm(m[:,:,10:12],axis=2)
    eps=256*np.finfo(float).eps*np.maximum(1,np.abs(draft).max(1))
    active_remaining=(remaining>eps[None])&(raw>eps[None])&movable[None]
    harmful_freeze=(rounds[:,4]<=1e-10)&active_remaining.any(1)
    for t in range(len(rounds)):
        lim=int(rounds[t,6]);old_lim=int(rounds[t,5])
        steps.append(dict(iteration=t+1,attractive_pairs=int(rounds[t,0]),repulsive_pairs=int(rounds[t,1]),damping=int(rounds[t,2]),
            old_common_scale=float(rounds[t,3]),actual_common_scale=float(rounds[t,4]),limiter_id=lim,
            limiter_mass=float(mass[lim]) if lim>=0 else None,old_limiter_id=old_lim,
            excluded_exhausted=int(rounds[t,7]),shape_clipped=int(rounds[t,8]),coordinate_clipped=int(rounds[t,9]),
            harmful_global_freeze=bool(harmful_freeze[t]),other_mobile_budget_mean_use=float((previous_path[t,active_remaining[t]]/(log['config']['total_fraction']*local[active_remaining[t]])).mean()) if active_remaining[t].any() else None,
            mean_actual_step_fraction=float((actual[t,movable]/local[movable]).mean()) if movable.any() else 0.,
            mean_raw_step_fraction=float((raw[t,movable]/local[movable]).mean()) if movable.any() else 0.,
            overlap_before=float(inter[t]),overlap_after=float(inter[t+1]),bbox_before=float(bbox[t]),bbox_after=float(bbox[t+1]),
            near_diagonal_force_fraction=float(rounds[t,11]/max(rounds[t,12],1e-30))))
    force=m[:,:,1:3]+m[:,:,3:5];force_norm=np.linalg.norm(force,axis=2)
    quartiles=[];bins=np.digitize(area,np.quantile(area,[.25,.5,.75]),right=True)
    for group in range(4):
        ids=(bins==group)&movable
        quartiles.append(dict(quartile=group+1,count=int(ids.sum()),mean_mass=float(mass[ids].mean()) if ids.any() else None,
            mean_center_path_fraction=float((m[-1,ids,12]/local[ids]).mean()) if ids.any() else None,
            mean_attractive_force=float(np.linalg.norm(m[:,ids,1:3],axis=2).mean()) if ids.any() else None))
    endcenter=x[-1,:,:2]+x[-1,:,2:]/2;startcenter=draft[:,:2]+draft[:,2:]/2
    final=measures(x[-1]);paths=m[-1,:,12]/local;qpaths=m[-1,:,17]
    return dict(rounds=steps,final_geometry=final,mass_quartiles=quartiles,
        harmful_global_freeze_rounds=int(harmful_freeze.sum()),old_scale_zero_rounds=int((rounds[:,3]<=1e-10).sum()),
        mean_center_displacement_fraction=float((np.linalg.norm(endcenter-startcenter,axis=1)/local)[movable].mean()) if movable.any() else 0.,
        mean_center_path_fraction=float(paths[movable].mean()) if movable.any() else 0.,
        path_budget_hit_modules=int((movable&(paths>=log['config']['total_fraction']-1e-9)).sum()),
        movable_modules=int(movable.sum()),shape_budget_hit_modules=int((soft&(qpaths>=np.log1p(log['config']['shape_total_fraction'])-1e-9)).sum()),free_shape_modules=int(soft.sum()),
        force_cancellation_ratio=float(force_norm.sum()/max(m[:,:,5].sum(),1e-30)),
        attraction_pair_magnitude_fraction=float(m[:,:,20].sum()/max((m[:,:,20]+m[:,:,21]).sum(),1e-30)),
        mean_shape_step_fraction_of_cap=float(np.abs(m[:,:,16])[:,soft].mean()/np.log1p(log['config']['shape_step_fraction'])) if soft.any() else 0.)

def summarize(records):
    out={}
    for arm in variants():
        rows=[r for r in records if r['arm']==arm];d=[r['diagnostics'] for r in rows];timevalues=np.array([r['operator_stats']['seconds'] for r in rows])
        out[arm]=dict(count=len(rows),cases_with_harmful_global_freeze=sum(x['harmful_global_freeze_rounds']>0 for x in d),
            total_harmful_global_freeze_rounds=sum(x['harmful_global_freeze_rounds'] for x in d),
            mean_raw_overlap=float(np.mean([x['final_geometry']['pair_intersection_over_area_sum'] for x in d])),
            mean_exact_void_fraction=float(np.mean([x['final_geometry']['exact_void_fraction'] for x in d])),
            mean_below10_fraction=float(np.mean([x['final_geometry']['fraction_strictly_below_10pct'] for x in d])),
            mean_worst_module_coverage=float(np.mean([x['final_geometry']['worst_module_coverage'] for x in d])),
            mean_center_displacement_fraction=float(np.mean([x['mean_center_displacement_fraction'] for x in d])),
            path_budget_hit_module_fraction=sum(x['path_budget_hit_modules'] for x in d)/max(1,sum(x['movable_modules'] for x in d)),
            shape_budget_hit_module_fraction=sum(x['shape_budget_hit_modules'] for x in d)/max(1,sum(x['free_shape_modules'] for x in d)),
            mean_attraction_force_fraction=float(np.mean([x['attraction_pair_magnitude_fraction'] for x in d])),
            operator_seconds=dict(mean=float(timevalues.mean()),p95=float(np.percentile(timevalues,95)),max=float(timevalues.max())))
    return out

def run(resume=False):
    pre=read(PREFLIGHT/'results.json');assert pre['status']=='passed' and pre['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Use explicit --resume')
    identity=register(OUT);records=[];begin=time.perf_counter();members=registry()['sets']['train_monitor500']['records']
    with process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1);warm_draft,warm_inputs,_=cached(0);cold=time.perf_counter();preprocess(warm_draft,warm_inputs,capture=True);cold=time.perf_counter()-cold
        for index in range(500):
            draft,inputs,source=cached(index);assert source['member']==members[index]['member']
            for arm,cfg in variants().items():
                if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),total=2000,test_read=False));return
                path=OUT/'cases'/arm/f'{index:04d}.json';coords=OUT/'trajectories'/arm/f'{index:04d}.npz'
                if path.exists():
                    row=read(path);assert row['member']==source['member'] and sha(coords)==row['trajectory_sha256'];records.append(row);continue
                final,log=preprocess(draft,inputs,cfg,capture=True);d=diagnose(draft,inputs,log)
                coords.parent.mkdir(parents=True,exist_ok=True);temp=coords.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,history=log['history'],module_log=log['module_log'],round_log=log['round_log'])
                temp.replace(coords)
                row=dict(index=index,member=source['member'],arm=arm,n=int(inputs[0]),operator_stats={k:v for k,v in log.items() if not isinstance(v,np.ndarray)},
                    diagnostics=d,trajectory_sha256=sha(coords),input_coordinate_source=str((CONTROL/'coordinates'/f'{index:04d}.npz').relative_to(ROOT)),input_coordinate_sha256=source['coordinates_sha256'])
                save(path,row);records.append(row)
                save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=2000,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            if (index+1)%25==0:print('R15 trajectory cases',index+1,'/500',flush=True)
        assert sources()==identity['sources'];summary=summarize(records)
        result=dict(status='complete',summary=summary,records=records,cold_native_seconds=cold,module_columns=MODULE_COLUMNS,round_columns=ROUND_COLUMNS,
            source_sha256=identity['sources'],no_planner_solves=True,no_new_training=True,test_read=False)
        save(OUT/'results.json',result);save(OUT/'state.json',dict(status='complete',completed=2000,total=2000,test_read=False))
        table=[dict(index=r['index'],arm=r['arm'],**t) for r in records for t in r['diagnostics']['rounds']]
        with (OUT/'operations.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
        study=read(RESEARCH/'study.json');study.update(status='R15_trajectory_complete_register_full_planner_controls_next',current_run=None,active_pid=None)
        study['R15_trajectory_audit']=dict(run=PLAN['output'],summary=summary);save(RESEARCH/'study.json',study)
        print(json.dumps(dict(status='complete',summary=summary,no_planner_solves=True,test_read=False),indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run(args.resume)
    except BaseException as e:
        out=PREFLIGHT if args.preflight else OUT
        save(out/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),test_read=False));raise
