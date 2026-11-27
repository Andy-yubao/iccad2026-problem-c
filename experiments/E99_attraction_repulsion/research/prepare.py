"""Registered frozen C3 preflight and reusable train500 full-planner control."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,datetime,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts,registry,public_hard_reference
from datasets.public_member_inputs import public_member_inputs
from scoring.evaluation import evaluate_case
from scoring.checks import independent_hard_checks
from scoring.official import official_evaluator
from ..planner.solver import optimize
from ..planner.canonical_native import warmup
from ..tune import process_lock
from .deployment import FrozenC3,CHECKPOINT_SHA

HERE=Path(__file__).resolve().parent;EXPERIMENT=HERE.parent
PLAN=json.loads((HERE/'prepare_plan.json').read_text(encoding='utf-8'))
OUT=EXPERIMENT/PLAN['output'];PREFLIGHT=EXPERIMENT/PLAN['preflight_output']


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix('.tmp')
    temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(p)
def setup():
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def stopped(out):return (HERE/'STOP').exists() or (out/'STOP').exists()
def sources():
    paths=[Path(__file__),HERE/'deployment.py',HERE/'prepare_plan.json',EXPERIMENT/'planner_config.json',EXPERIMENT/'tune.py',
        *HERE.joinpath('frozen_c3').rglob('*.py'),*EXPERIMENT.joinpath('planner').rglob('*.py'),
        *[ROOT/p for p in ('datasets/paths.py','datasets/reference_layouts.py','datasets/public_member_inputs.py','scoring/checks.py','scoring/evaluation.py','scoring/official.py')]]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
def state_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def hard(final,inputs):
    c=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]))
    if not c['is_feasible'] or not np.isfinite(final).all() or (final[:,:2]<0).any():raise AssertionError('Invalid returned geometry')
    return c


@torch.inference_mode()
def preflight():
    PREFLIGHT.mkdir(parents=True,exist_ok=False);setup();rows=[]
    with threadpool_limits(limits=1):
        entries=registry()['sets']['train']['records'];cases=[public_member_inputs(entries[i],'train') for i in PLAN['preflight']['indices']]
        model=FrozenC3(cases[0]);before=state_hash(model.model)
        for i,inputs in zip(PLAN['preflight']['indices'],cases):
            if stopped(PREFLIGHT):raise RuntimeError('STOP')
            expected=model.eager(inputs);torch.cuda.synchronize();start=time.perf_counter();p=model.predict(inputs);wall=time.perf_counter()-start
            if not np.array_equal(p,expected):raise AssertionError(('Eager/capture mismatch',i,float(np.abs(p-expected).max())))
            c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1);pre=c[:,1]!=0
            assert np.isfinite(p).all() and (p[:,2:]>0).all() and (p[:,:2]>=0).all()
            np.testing.assert_array_equal(p[locked,2:],s[locked,2:]);np.testing.assert_array_equal(p[pre],s[pre])
            rows.append(dict(train_index=i,member=entries[i]['member'],n=int(inputs[0]),public_to_draft_seconds=wall,bit_exact=True))
        after=state_hash(model.model);assert before==after and all(not p.requires_grad for p in model.model.parameters())
        result=dict(status='passed',records=rows,cold_seconds=model.cold_seconds,source_sha256=sources(),checkpoint_sha256=CHECKPOINT_SHA,
            frozen_state_sha256=before,model_state_unchanged=True,optimizer_or_training=False,test_read=False)
        save(PREFLIGHT/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))


@torch.inference_mode()
def prepare(resume=False):
    check=read(PREFLIGHT/'results.json');assert check['status']=='passed' and check['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Existing evidence protected; explicit --resume required')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'cases').mkdir(exist_ok=True);(OUT/'coordinates').mkdir(exist_ok=True)
    setup();options=read(EXPERIMENT/'planner_config.json');data=ReferenceLayouts('train_monitor500');begin=time.perf_counter();records=[]
    identity=dict(plan=PLAN,sources=sources(),members=[r['member'] for r in data.records],registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),
                  paths_sha256=sha(ROOT/'protocol/paths.json'),checkpoint_sha256=CHECKPOINT_SHA)
    if (OUT/'registration.json').exists():assert read(OUT/'registration.json')['identity']==identity
    else:
        save(OUT/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(OUT/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for name in identity['sources']:z.write(ROOT/name,name)
    with process_lock(HERE/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1)
        if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=0,test_read=False));return
        first=data[0];model=FrozenC3(first['case'].solver_inputs);native=time.perf_counter();warmup();native=time.perf_counter()-native
        for index in range(500):
            if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),test_read=False));return
            path=OUT/'cases'/f'{index:04d}.json';coords=OUT/'coordinates'/f'{index:04d}.npz'
            if path.exists():
                r=read(path);assert r['member']==data.records[index]['member'] and sha(coords)==r['coordinates_sha256'];records.append(r);continue
            row=data[index];inputs=row['case'].solver_inputs
            torch.cuda.synchronize();start=time.perf_counter();draft=model.predict(inputs);draft_seconds=time.perf_counter()-start
            final,trace=optimize(draft,inputs,**options);h=hard(final,inputs);wall=time.perf_counter()-start
            scored=evaluate_case(row['case'],final,wall)
            tmp=coords.with_suffix('.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,positions=draft,final=final,areas=np.asarray(inputs[1]),b2b=np.asarray(inputs[2]),p2b=np.asarray(inputs[3]),pins=np.asarray(inputs[4]),constraints=np.asarray(inputs[5]),specs=np.asarray(inputs[6]))
            tmp.replace(coords)
            r=dict(index=index,member=list(row['member']),n=int(inputs[0]),draft_seconds=draft_seconds,complete_seconds=wall,
                repair_seconds=trace['seconds'],metrics=scored['metrics'],hard=h,trace=trace,coordinates_sha256=sha(coords))
            save(path,r);records.append(r)
            save(OUT/'state.json',dict(status='running',stage='frozen_C3_train500_control',pid=os.getpid(),completed=len(records),total=500,
                elapsed_seconds=time.perf_counter()-begin,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),test_read=False))
            if (index+1)%25==0:print('TRAIN CONTROL',index+1,'/500',flush=True)
        assert sources()==identity['sources']
        values=np.array([r['complete_seconds'] for r in records]);q=[r['metrics']['cost'] for r in records]
        result=dict(status='complete',finished=True,count=500,panel='train_monitor500',weighted_Q=official_evaluator().compute_total_score(q,[r['n'] for r in records]),
            hard_legal=500,complete_mean_seconds=float(values.mean()),complete_p95_seconds=float(np.percentile(values,95)),complete_max_seconds=float(values.max()),
            cases_at_or_over250ms=int((values>=.25).sum()),cold_model_graph_seconds=model.cold_seconds,cold_native_seconds=native,
            source_sha256=identity['sources'],records=records,elapsed_seconds=time.perf_counter()-begin,test_read=False,no_training=True)
        save(OUT/'results.json',result);save(OUT/'state.json',dict(status='complete',stage='exited',pid=os.getpid(),completed=500,total=500,test_read=False))
        study=read(HERE/'study.json');study.update(status='training_control_complete_next_method_registration',current_run=None,training_control=PLAN['output']);save(HERE/'study.json',study)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else prepare(args.resume)
    except BaseException as e:
        target=PREFLIGHT if args.preflight else OUT
        save(target/'error.json',dict(error=repr(e),pid=os.getpid(),no_automatic_retry=True,test_read=False));raise
