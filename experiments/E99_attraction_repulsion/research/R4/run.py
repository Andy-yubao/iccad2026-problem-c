"""Source-bound R4 matched training study; cached C3 timing is diagnostic only."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import argparse,datetime,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from ...tune import process_lock
from ...planner.canonical_native import warmup
from .bounded import optimize

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;EXPERIMENT=RESEARCH.parent
PLAN=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
REUSED=EXPERIMENT/'runs/research/20261103_R3_train500_matched'
CONTROL=EXPERIMENT/'runs/research/20261102_frozen_C3_train500_control'
OUT=EXPERIMENT/'runs/research/20261103_R4_train500_matched'
PREFLIGHT=EXPERIMENT/'runs/research/20261103_R4_preflight5'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(5):
        try:temp.replace(p);return
        except PermissionError:
            if attempt==4:raise
            time.sleep(.02*(attempt+1))

def sources():
    paths=[*HERE.rglob('*.py'),HERE/'plan.json',*EXPERIMENT.joinpath('planner').rglob('*.py'),
        EXPERIMENT/'relaxation.py',EXPERIMENT/'tune.py',
        *[ROOT/s for s in ('scoring/checks.py','scoring/evaluation.py','scoring/official.py','datasets/reference_layouts.py')]]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}

def cached(index):
    record=read(CONTROL/'cases'/f'{index:04d}.json');path=CONTROL/'coordinates'/f'{index:04d}.npz'
    assert sha(path)==record['coordinates_sha256']
    with np.load(path) as a:
        inputs=(record['n'],*[a[k].copy() for k in ('areas','b2b','p2b','pins','constraints','specs')])
        draft=a['positions'].copy()
    return draft,inputs,record

def hard(final,inputs):
    h=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
    assert h['is_feasible'] and np.isfinite(final).all() and (final[:,:2]>=0).all()
    return h

def stopped(out):return (RESEARCH/'STOP').exists() or (out/'STOP').exists()

def register(out):
    identity=dict(plan=PLAN,sources=sources(),control_registration_sha256=sha(CONTROL/'registration.json'),
        reused_control_result_sha256=sha(REUSED/'results.json'),registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),timing_scope='cached_C3_to_final_CPU_only_not_acceptance')
    if (out/'registration.json').exists():assert read(out/'registration.json')['identity']==identity
    else:
        out.mkdir(parents=True,exist_ok=True)
        save(out/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in identity['sources']:z.write(ROOT/p,p)
    return identity


def preflight():
    if PREFLIGHT.exists():raise FileExistsError('Preflight evidence protected')
    identity=register(PREFLIGHT);rows=[]
    with process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1);warmup()
        for index in PLAN['implementation_panel']:
            if stopped(PREFLIGHT):raise RuntimeError('STOP')
            draft,inputs,_=cached(index)
            for arm in PLAN['arms']:
                if arm=='conservative_control':
                    source=next(r for r in read(EXPERIMENT/'runs/research/20261103_R3_preflight5/results.json')['records'] if r['index']==index and r['arm']=='raw_hard_control')
                    rows.append(dict(source,arm=arm,reused_control=True));continue
                final,trace,prefix=optimize(draft,inputs,PLAN['backend'],arm=='convex_area');h=hard(final,inputs)
                rows.append(dict(index=index,arm=arm,n=len(final),hard=h,trace=trace))
    assert sources()==identity['sources']
    result=dict(status='passed',unique_cases=5,records=rows,source_sha256=identity['sources'],test_read=False)
    save(PREFLIGHT/'results.json',result)
    print(json.dumps(dict(status='passed',records=[dict(index=r['index'],arm=r['arm'],seconds=r['trace']['seconds'],row_retained=r['trace']['row_retained']) for r in rows]),indent=2))


def summary(records,arm):
    rows=[r for r in records if r['arm']==arm];walls=np.array([r['geometry_and_hard_seconds'] for r in rows]);q=[r['metrics']['cost'] for r in rows]
    return dict(count=len(rows),weighted_Q=official_evaluator().compute_total_score(q,[r['n'] for r in rows]),
        hard_legal=sum(r['metrics']['is_feasible'] and r['hard']['is_feasible'] for r in rows),
        row_fallbacks=sum(r['trace']['row_retained'] for r in rows),CPU_geometry_hard_mean_seconds=float(walls.mean()),
        CPU_geometry_hard_p95_seconds=float(np.percentile(walls,95)),CPU_geometry_hard_max_seconds=float(walls.max()),
        cases_CPU_geometry_at_or_over220ms=int((walls>=.220).sum()),mean_shape_calls=float(np.mean([sum(t['phase']=='shape' for t in r['trace']['trace']) for r in rows])),
        mean_boundary_violations=float(np.mean([r['metrics']['boundary_violations'] for r in rows])),mean_grouping_violations=float(np.mean([r['metrics']['grouping_violations'] for r in rows])))


def run(resume):
    assert read(CONTROL/'results.json')['status']=='complete'
    assert read(EXPERIMENT/'runs/research/20261103_R3_train500_matched/results.json')['status']=='complete'
    pre=read(PREFLIGHT/'results.json');assert pre['status']=='passed' and pre['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Use explicit --resume')
    identity=register(OUT);data=ReferenceLayouts('train_monitor500');records=[];begin=time.perf_counter()
    with process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1);warmup()
        for index in range(500):
            for arm in PLAN['arms']:
                if arm=='conservative_control':
                    source=REUSED/'cases/raw_hard_control'/f'{index:04d}.json'
                    record=read(source);assert record['member']==data.records[index]['member']
                    record=dict(record,arm=arm,reused_control=True,reused_record=str(source.relative_to(ROOT)),reused_record_sha256=sha(source),timing_scope='historical R3 CPU timing read-only reuse, not newly paired timing')
                    save(OUT/'cases'/arm/f'{index:04d}.json',record);records.append(record);continue
                if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),test_read=False));return
                path=OUT/'cases'/arm/f'{index:04d}.json';coords=OUT/'coordinates'/arm/f'{index:04d}.npz'
                if path.exists():
                    r=read(path);assert r['member']==data.records[index]['member'] and sha(coords)==r['coordinates_sha256'];records.append(r);continue
                draft,inputs,control=cached(index);row=data[index]
                assert control['member']==list(row['member'])
                start=time.perf_counter();final,trace,prefix=optimize(draft,inputs,PLAN['backend'],arm=='convex_area');h=hard(final,inputs);wall=time.perf_counter()-start
                scored=evaluate_case(row['case'],final,wall)
                coords.parent.mkdir(parents=True,exist_ok=True);temp=coords.with_suffix('.tmp')
                with temp.open('wb') as f:np.savez_compressed(f,prefix=prefix,final=final)
                temp.replace(coords)
                r=dict(index=index,member=list(row['member']),arm=arm,n=int(inputs[0]),geometry_and_hard_seconds=wall,
                    metrics=scored['metrics'],hard=h,trace=trace,coordinates_sha256=sha(coords),control_Q=control['metrics']['cost'],
                    timing_scope='CPU_geometry_only_cached_frozen_C3_not_acceptance')
                save(path,r);records.append(r)
                save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=1000,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            if (index+1)%25==0:print('R4 paired',index+1,'/500',flush=True)
        assert sources()==identity['sources']
        summaries={arm:summary(records,arm) for arm in PLAN['arms']}
        a=np.array([r['metrics']['cost'] for r in records if r['arm']==PLAN['arms'][0]])
        b=np.array([r['metrics']['cost'] for r in records if r['arm']==PLAN['arms'][1]])
        result=dict(status='complete',panel='train_monitor500',summaries=summaries,organized_vs_control=dict(better=int((b<a-1e-10).sum()),equal=int((np.abs(b-a)<=1e-10).sum()),worse=int((b>a+1e-10).sum())),records=records,test_read=False,timing_scope='CPU cached C3 geometry only; no formal dev acceptance')
        save(OUT/'results.json',result);save(OUT/'state.json',dict(status='complete',completed=1000,test_read=False))
        study=read(RESEARCH/'study.json');study.update(status='R4_training_matched_complete_review_next_method',current_run=None,active_pid=None)
        study['candidate_ledger'].append(dict(candidate='R4',plan='research/R4/plan.json',run=str(OUT.relative_to(EXPERIMENT)),status='train_complete_no_dev',summaries=summaries));save(RESEARCH/'study.json',study)
        print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run(args.resume)
    except BaseException as e:
        target=PREFLIGHT if args.preflight else OUT
        save(target/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),test_read=False));raise
