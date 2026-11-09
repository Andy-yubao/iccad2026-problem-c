"""One frozen current-best candidate, fresh complete C3 dev500 acceptance."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['MPLBACKEND']='Agg'
import argparse,datetime,json,time,zipfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import ReferenceLayouts,registry,public_hard_reference
from datasets.public_member_inputs import public_member_inputs
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from ...planner.solver import optimize
from ...planner.canonical_native import warmup
from ..deployment import FrozenC3,CHECKPOINT_SHA
from ..R15.operator import preprocess,variants
from ..R16 import replay
from .public_inputs import PublicShard,build_public
from .gallery import draw,render,PRIOR

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;EXPERIMENT=RESEARCH.parent;ROOT=replay.ROOT
read=replay.read;save=replay.save;sha=replay.sha
PLAN=read(HERE/'plan.json');CFG=variants()['attraction_equal_slope'];assert asdict(CFG)==PLAN['configuration']
assert PLAN['checkpoint_sha256']==CHECKPOINT_SHA
OUT=EXPERIMENT/PLAN['output'];PREFLIGHT=EXPERIMENT/PLAN['preflight_output']
CONTROL=EXPERIMENT/'runs/research/20261103_R15_AR_original_planner_train500'


def sources():
    result=replay.sources()
    paths=[HERE/p for p in ('run.py','public_inputs.py','gallery.py','plan.json')]
    paths+=[RESEARCH/'R16/replay.py',RESEARCH/'deployment.py',*RESEARCH.joinpath('frozen_c3').rglob('*.py')]
    paths+=[ROOT/p for p in ('datasets/public_member_inputs.py','datasets/worksets.py','datasets/cases.py','scoring/checks.py','scoring/evaluation.py','scoring/official.py','visualization/layouts.py')]
    result.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths});return result


def setup():
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def model_hash(model):
    import hashlib
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def register(out,role):
    assert sha(ROOT/PLAN['checkpoint'])==CHECKPOINT_SHA
    selected=read(CONTROL/'registration.json')['identity']['sources'];now=sources()
    frozen={p:h for p,h in selected.items() if p.startswith('experiments/E99_attraction_repulsion/planner/') or p in ('experiments/E99_attraction_repulsion/planner_config.json','experiments/E99_attraction_repulsion/research/R15/operator.py')}
    assert frozen and all(now[p]==h for p,h in frozen.items()),'Selected original numerical sources changed'
    identity=dict(plan=PLAN,sources=now,config=asdict(CFG),role=role,members=[r['member'] for r in registry()['sets'][role]['records']],
        checkpoint_sha256=CHECKPOINT_SHA,selected_training_registration_sha256=sha(CONTROL/'registration.json'),selected_training_results_sha256=sha(CONTROL/'results.json'),
        registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),paths_sha256=sha(ROOT/'protocol/paths.json'),prior_V3_config_sha256=sha(PRIOR),original_planner=replay.check_planner())
    if (out/'registration.json').exists():assert read(out/'registration.json')['identity']==identity
    else:
        out.mkdir(parents=True,exist_ok=True);save(out/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in now:z.write(ROOT/p,p)
    return identity


def stopped(out):return (RESEARCH/'STOP').exists() or (out/'STOP').exists()


def hard(final,inputs):
    h=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
    assert h['is_feasible'] and np.isfinite(final).all() and (final[:,:2]>=0).all()
    return h


def warm_model(cache,record):
    raw,io=cache.load(record);public=build_public(raw,record);model=FrozenC3(public)
    start=time.perf_counter();draft=model.predict(public);preprocess(draft,public,CFG,capture=False);warmup();native=time.perf_counter()-start
    return model,dict(model_load_capture_seconds=model.cold_seconds,AR_native_warmup_and_one_draft_seconds=native,first_shard_loading_seconds=io,GPU=torch.cuda.get_device_name(),precision='FP32',batch=1)


def calculate(index,data,cache,model,out,options):
    metadata=data.records[index];raw,io=cache.load(metadata)
    torch.cuda.synchronize();start=time.perf_counter()
    inputs=build_public(raw,metadata);public_end=time.perf_counter()
    draft=model.predict(inputs);draft_end=time.perf_counter()
    prefix,stats=preprocess(draft,inputs,CFG,capture=False);AR_end=time.perf_counter()
    final,trace=optimize(prefix,inputs,**options);planner_end=time.perf_counter()
    h=hard(final,inputs);torch.cuda.synchronize();end=time.perf_counter();wall=end-start
    # References and denominators first accessed only after complete request returns.
    row=data[index];assert list(row['member'])==metadata['member']
    score=evaluate_case(row['case'],final,wall);assert score['metrics']['is_feasible']
    coords=out/'coordinates'/f'{index:04d}.npz';coords.parent.mkdir(parents=True,exist_ok=True);tmp=coords.with_suffix('.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,raw=draft,prefix=prefix,final=final,**{k:np.asarray(v) for k,v in zip(('areas','b2b','p2b','pins','constraints','specs'),inputs[1:])})
    tmp.replace(coords)
    record=dict(index=index,member=metadata['member'],n=int(inputs[0]),metrics=score['metrics'],hard=h,complete_seconds=wall,
        stages_seconds=dict(public_construction=public_end-start,public_features_C3_transfers=draft_end-public_end,AR=AR_end-draft_end,planner=planner_end-AR_end,hard_and_final_sync=end-planner_end),
        excluded_shard_loading_seconds=io,preprocessor_stats=stats,planner_trace=trace,coordinate_source=coords.relative_to(ROOT).as_posix(),coordinates_sha256=sha(coords),
        timing_scope='fresh public extraction/masking/area transform -> features/C3 GPU batch1 -> AR -> full original planner/fallback -> public-only hard checks; IO/load/warmup separate')
    save(out/'cases'/f'{index:04d}.json',record);return record


@torch.inference_mode()
def preflight():
    if PREFLIGHT.exists():raise FileExistsError('Preflight evidence protected')
    setup();identity=register(PREFLIGHT,'train_monitor500');data=ReferenceLayouts('train_monitor500');cache=PublicShard('train');checks=[];records=[];options=read(EXPERIMENT/'planner_config.json')
    with replay.process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        model,cold=warm_model(cache,data.records[0]);before=model_hash(model.model)
        for index in PLAN['preflight']['indices']:
            if stopped(PREFLIGHT):raise RuntimeError('STOP')
            meta=data.records[index];raw,_=cache.load(meta);inputs=build_public(raw,meta);existing=public_member_inputs(meta,'train')
            for a,b in zip(inputs,existing):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
            eager=model.eager(inputs);draft=model.predict(inputs);np.testing.assert_array_equal(draft,eager)
            cached,_,_=replay.cached(index);np.testing.assert_array_equal(draft,cached)
            prefix,_=preprocess(draft,inputs,CFG,capture=False)
            with np.load(EXPERIMENT/'runs/research/20261103_R15_train500_trajectory_audit/trajectories/attraction_equal_slope'/f'{index:04d}.npz') as a:np.testing.assert_array_equal(prefix,a['history'][-1])
            record=calculate(index,data,cache,model,PREFLIGHT,options);records.append(record)
            checks.append(dict(index=index,public_constructor_exact=True,C3_eager_capture_and_saved_draft_bit_exact=True,AR_saved_trajectory_bit_exact=True,all_hard_legal=True))
        after=model_hash(model.model);assert before==after and all(not p.requires_grad for p in model.model.parameters())
        draw(records[0],PREFLIGHT/'preview',records[0]['index'],'train500')
    assert sources()==identity['sources']
    save(PREFLIGHT/'results.json',dict(status='passed',unique_cases=5,checks=checks,records=records,cold=cold,frozen_model_state_sha256=before,source_sha256=identity['sources'],test_read=False,new_training=False))
    print(json.dumps(dict(status='passed',unique_cases=5,public_builder_exact=True,C3_and_AR_saved_evidence_bit_exact=True,all_hard_legal=True,cold=cold),indent=2))


def summarize(records):
    t=np.array([r['complete_seconds'] for r in records]);q=[r['metrics']['cost'] for r in records];n=[r['n'] for r in records]
    Q=official_evaluator().compute_total_score(q,n);legal=sum(r['hard']['is_feasible'] and r['metrics']['is_feasible'] for r in records)
    return dict(count=len(records),weighted_neutral_Q=Q,hard_legal=legal,complete_seconds=dict(mean=float(t.mean()),p95=float(np.percentile(t,95)),max=float(t.max())),
        cases_at_or_over250ms=int((t>=.25).sum()),slow_indices=[r['index'] for r in records if r['complete_seconds']>=.25],
        stages_mean_seconds={k:float(np.mean([r['stages_seconds'][k] for r in records])) for k in records[0]['stages_seconds']},
        mean_excluded_shard_loading_seconds=float(np.mean([r['excluded_shard_loading_seconds'] for r in records])),
        mean_planner_shape_calls=float(np.mean([sum(t['phase']=='shape' for t in r['planner_trace']['trace']) for r in records])),
        deep_packed_refine_calls=sum(t['phase']=='full_packed_refine' for r in records for t in r['planner_trace']['trace']),
        strict_acceptance_pass=bool(len(records)==500 and legal==500 and Q<1.01 and (t<.25).all()))


@torch.inference_mode()
def run(resume=False):
    pre=read(PREFLIGHT/'results.json');assert pre['status']=='passed' and pre['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Explicit identity-bound --resume required')
    setup();identity=register(OUT,'dev_monitor500');data=ReferenceLayouts('dev_monitor500');assert len(data)==500;cache=PublicShard('dev');records=[];options=read(EXPERIMENT/'planner_config.json');begin=time.perf_counter()
    with replay.process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=0,total=500));return
        model,cold=warm_model(cache,data.records[0]);before=model_hash(model.model)
        for index in range(500):
            if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),total=500));return
            path=OUT/'cases'/f'{index:04d}.json'
            if path.exists():record=read(path)
            else:record=calculate(index,data,cache,model,OUT,options)
            assert record['member']==data.records[index]['member'] and sha(ROOT/record['coordinate_source'])==record['coordinates_sha256']
            records.append(record);save(OUT/'state.json',dict(status='running',pid=os.getpid(),stage='complete_dev500',completed=len(records),total=500,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            if (index+1)%25==0:print('R23 complete frozen dev',index+1,'/500',flush=True)
        assert sources()==identity['sources'] and before==model_hash(model.model)
        assert all(not p.requires_grad for p in model.model.parameters()) and sha(ROOT/PLAN['checkpoint'])==CHECKPOINT_SHA
        summary=summarize(records)
        save(OUT/'results.json',dict(status='complete',panel='dev_monitor500',summary=summary,cold=cold,records=records,source_sha256=identity['sources'],frozen_model_state_sha256=before,new_training=False,test_read=False,original_planner_unchanged=True,config_frozen_before_dev=True))
        save(OUT/'state.json',dict(status='numerical_complete_gallery_pending',pid=os.getpid(),completed=500,total=500,test_read=False))
        lines=['# Frozen retained AR: fresh complete C3 dev500 acceptance','',f"Weighted neutralQ={summary['weighted_neutral_Q']:.12f}, hard legal={summary['hard_legal']}/500.",
            '',f"Complete per-request mean/P95/max seconds: {summary['complete_seconds']}; >=250ms: {summary['cases_at_or_over250ms']}/500. Strict acceptance: {summary['strict_acceptance_pass']}.",
            '', 'Timer includes fresh public extraction/cloning/masking/mean-area transform, C3 features/padding/transfers/GPU/output restoration, current retained AR, all original planner/fallback, necessary public-only hard checks and final CUDA sync. Shard disk loading and model/native warmup are separate; no batch amortization. All500 bad/slow cases retained.',
            '', 'One candidate frozen from training before this dev run; no coefficient sweep or checkpoint reselection. Model unchanged, original planner unchanged, test sealed. Current100 four-stage gallery: [open saved gallery](four_stage100/index.html).']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        study=read(RESEARCH/'study.json');study.update(status='R23_complete_dev500_done_gallery_running',current_run=PLAN['output'],active_pid=os.getpid())
        study['R23_acceptance']=dict(plan='research/R23/plan.json',run=PLAN['output'],summary=summary,goal_complete=False);save(RESEARCH/'study.json',study)
        print(json.dumps(dict(status='numerical_complete',summary=summary,cold=cold),indent=2))
        rendered=render(OUT,records)
        save(OUT/'state.json',dict(status='complete' if rendered else 'numerical_complete_gallery_stopped',completed=500,total=500,gallery_count=100 if rendered else read(OUT/'four_stage100/state.json')['completed'],test_read=False))
        study=read(RESEARCH/'study.json');study.update(status='R23_acceptance_and_gallery_complete_review' if rendered else 'R23_acceptance_complete_gallery_stopped',current_run=None,active_pid=None);study['R23_acceptance']['gallery_complete']=rendered;save(RESEARCH/'study.json',study)
        print(json.dumps(dict(status='complete',strict_acceptance_pass=summary['strict_acceptance_pass'],gallery_complete=rendered)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run(args.resume)
    except BaseException as error:
        out=PREFLIGHT if args.preflight else OUT
        save(out/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(error),pid=os.getpid(),no_automatic_retry=True,test_read=False));raise
