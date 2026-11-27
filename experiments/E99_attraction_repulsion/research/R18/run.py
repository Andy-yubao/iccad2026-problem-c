"""Single preregistered rectangle-face AR control with unchanged original planner."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,datetime,json,time,zipfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
from threadpoolctl import threadpool_limits
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from ...planner.solver import optimize
from ...planner.canonical_native import warmup
from ..R16 import replay
from ..R15.operator import preprocess as old_preprocess,variants as old_variants
from .operator import preprocess,Config,variants,MODULE_COLUMNS,ROUND_COLUMNS

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;EXPERIMENT=RESEARCH.parent;ROOT=replay.ROOT
read=replay.read;save=replay.save;sha=replay.sha
PLAN=read(HERE/'candidate_plan.json');CFG=variants()['rectangle_faces']
assert asdict(CFG)==PLAN['configuration']
OUT=EXPERIMENT/PLAN['output'];PREFLIGHT=EXPERIMENT/PLAN['preflight_output']
CONTROL=EXPERIMENT/'runs/research/20261103_R15_AR_original_planner_train500'


def sources():
    result=replay.sources()
    paths=[HERE/p for p in ('run.py','operator.py','candidate_plan.json','operator_provenance.json')]
    paths+=[RESEARCH/'R16/replay.py']
    paths+=[ROOT/p for p in ('scoring/checks.py','scoring/evaluation.py','scoring/official.py')]
    result.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    return result


def register(out):
    assert PLAN['test_read'] is False
    assert read(EXPERIMENT/'runs/research/20261103_R17_saved_shape_direction_audit/results.json')['status']=='complete'
    identity=dict(plan=PLAN,config=asdict(CFG),sources=sources(),original_planner=replay.check_planner(),
        source_C3_registration_sha256=sha(replay.CONTROL/'registration.json'),
        source_C3_results_sha256=sha(replay.CONTROL/'results.json'),
        matched_AR_control_registration_sha256=sha(CONTROL/'registration.json'),matched_AR_control_results_sha256=sha(CONTROL/'results.json'),
        registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),parent_R17_results_sha256=sha(EXPERIMENT/'runs/research/20261103_R17_rectangle_face_train500/results.json'))
    if (out/'registration.json').exists():assert read(out/'registration.json')['identity']==identity
    else:
        out.mkdir(parents=True,exist_ok=True)
        save(out/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in identity['sources']:z.write(ROOT/p,p)
    return identity


def stopped(out):return (RESEARCH/'STOP').exists() or (out/'STOP').exists()


def hard(final,inputs):
    h=independent_hard_checks(final.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
    assert h['is_feasible'] and np.isfinite(final).all() and (final[:,:2]>=0).all()
    return h


def calculate(index,row,out,options):
    draft,inputs,original=replay.cached(index);assert original['member']==list(row['member'])
    control_path=CONTROL/'cases/attraction_equal_slope'/f'{index:04d}.json';control=read(control_path)
    assert control['member']==original['member'] and control['n']==original['n']
    # Every operation is captured outside production timing, once per new case.
    expected,log=preprocess(draft,inputs,CFG,capture=True)
    diagnostics=replay.diagnose(draft,inputs,log)
    begin=time.perf_counter();prefix,stats=preprocess(draft,inputs,CFG,capture=False)
    final,trace=optimize(prefix,inputs,**options);h=hard(final,inputs);wall=time.perf_counter()-begin
    np.testing.assert_array_equal(expected,prefix)
    score=evaluate_case(row['case'],final,wall)
    assert score['metrics']['is_feasible']
    coords=out/'coordinates'/f'{index:04d}.npz';coords.parent.mkdir(parents=True,exist_ok=True);temp=coords.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,history=log['history'],module_log=log['module_log'],round_log=log['round_log'],final=final)
    temp.replace(coords)
    record=dict(index=index,member=original['member'],n=original['n'],config=asdict(CFG),metrics=score['metrics'],hard=h,
        processing_seconds=wall,preprocessor_stats=stats,planner_trace=trace,diagnostics=diagnostics,
        coordinate_source=coords.relative_to(ROOT).as_posix(),coordinates_sha256=sha(coords),
        AR_control_Q=control['metrics']['cost'],control_Q=original['metrics']['cost'],
        AR_control_record_source=control_path.relative_to(ROOT).as_posix(),AR_control_record_sha256=sha(control_path),
        timing_scope='CPU_cached_draft_AR_original_planner_hard_only; no_C3_complete_latency_claim')
    save(out/'cases'/f'{index:04d}.json',record)
    return record


def compile_warm():
    draft,inputs,_=replay.cached(0);begin=time.perf_counter()
    preprocess(draft,inputs,CFG,capture=True);preprocess(draft,inputs,CFG,capture=False);warmup()
    return time.perf_counter()-begin


def preflight():
    if PREFLIGHT.exists():raise FileExistsError('Preflight evidence protected')
    identity=register(PREFLIGHT);data=ReferenceLayouts('train_monitor500');options=read(EXPERIMENT/'planner_config.json');records=[];checks=[]
    with replay.process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        cold=compile_warm()
        for index in PLAN['preflight']['indices']:
            if stopped(PREFLIGHT):raise RuntimeError('STOP')
            draft,inputs,_=replay.cached(index)
            previous,old=old_preprocess(draft,inputs,old_variants()['attraction_equal_slope'],capture=True)
            same,new=preprocess(draft,inputs,Config(exclude_exhausted=True,attraction_strength=1.),capture=True)
            np.testing.assert_array_equal(new['history'],old['history'])
            np.testing.assert_array_equal(same,previous)
            records.append(calculate(index,data[index],PREFLIGHT,options))
            checks.append(dict(index=index,unchanged_branch_history_bit_exact=True,candidate_capture_production_bit_exact=True))
    assert sources()==identity['sources']
    save(PREFLIGHT/'results.json',dict(status='passed',unique_cases=5,checks=checks,records=records,cold_seconds=cold,source_sha256=identity['sources'],test_read=False))
    print(json.dumps(dict(status='passed',unique_cases=5,unchanged_branch_bit_exact=True,all_hard_legal=True,cold_seconds=cold)))


def summary(records):
    q=np.array([r['metrics']['cost'] for r in records]);old=np.array([r['AR_control_Q'] for r in records]);wall=np.array([r['processing_seconds'] for r in records]);n=[r['n'] for r in records]
    score=official_evaluator().compute_total_score
    return dict(count=len(records),weighted_Q=score(q.tolist(),n),matched_AR_control_weighted_Q=score(old.tolist(),n),
        original_no_AR_weighted_Q=score([r['control_Q'] for r in records],n),hard_legal=sum(r['hard']['is_feasible'] and r['metrics']['is_feasible'] for r in records),
        compared_to_matched_AR=dict(better=int((q<old-1e-10).sum()),equal=int((np.abs(q-old)<=1e-10).sum()),worse=int((q>old+1e-10).sum())),
        processing_seconds=dict(mean=float(wall.mean()),p95=float(np.percentile(wall,95)),max=float(wall.max())),CPU_processing_over250ms=int((wall>=.25).sum()),
        preprocessor_mean_seconds=float(np.mean([r['preprocessor_stats']['seconds'] for r in records])),
        mean_planner_shape_calls=float(np.mean([sum(t['phase']=='shape' for t in r['planner_trace']['trace']) for r in records])),
        deep_packed_refine_calls=sum(t['phase']=='full_packed_refine' for r in records for t in r['planner_trace']['trace']),
        mean_raw_overlap=float(np.mean([r['diagnostics']['final_geometry']['pair_intersection_over_area_sum'] for r in records])),
        mean_boundary_violations=float(np.mean([r['metrics']['boundary_violations'] for r in records])),mean_grouping_violations=float(np.mean([r['metrics']['grouping_violations'] for r in records])))


def run(resume=False):
    pre=read(PREFLIGHT/'results.json');assert pre['status']=='passed' and pre['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Explicit source-bound --resume required')
    identity=register(OUT);data=ReferenceLayouts('train_monitor500');options=read(EXPERIMENT/'planner_config.json');records=[];begin=time.perf_counter()
    with replay.process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        cold=compile_warm()
        for index in range(500):
            if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),total=500));return
            path=OUT/'cases'/f'{index:04d}.json'
            if path.exists():record=read(path)
            elif index in PLAN['preflight']['indices']:
                original=PREFLIGHT/'cases'/f'{index:04d}.json';record=dict(read(original),reused_same_identity_preflight=True,preflight_record_sha256=sha(original))
                save(path,record)
            else:record=calculate(index,data[index],OUT,options)
            assert record['member']==data.records[index]['member'] and sha(ROOT/record['coordinate_source'])==record['coordinates_sha256']
            records.append(record)
            save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=500,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            if (index+1)%25==0:print('R18 rectangle face',index+1,'/500',flush=True)
        assert sources()==identity['sources'];s=summary(records)
        save(OUT/'results.json',dict(status='complete',summary=s,records=records,source_sha256=identity['sources'],module_columns=MODULE_COLUMNS,round_columns=ROUND_COLUMNS,
            cold_seconds=cold,fresh_solves=495,reused_same_identity_preflight=5,original_planner_unchanged=True,new_training=False,test_read=False))
        save(OUT/'state.json',dict(status='complete',completed=500,total=500,test_read=False))
        study=read(RESEARCH/'study.json');study.update(status='R18_rectangle_face_train500_complete_review',current_run=None,active_pid=None)
        study['R18_candidate']=dict(plan='research/R18/candidate_plan.json',run=PLAN['output'],summary=s);save(RESEARCH/'study.json',study)
        lines=['# Rectangle contact-face sign-only shape law: matched train500 control','',f"Candidate weightedQ={s['weighted_Q']:.12f}; prior matchedAR={s['matched_AR_control_weighted_Q']:.12f}; original noAR={s['original_no_AR_weighted_Q']:.12f}.",
            '',f"Hard legal {s['hard_legal']}/500. Paired {s['compared_to_matched_AR']}. CPU cached-draft mean/P95/max: {s['processing_seconds']} seconds; >=250ms: {s['CPU_processing_over250ms']}/500.",
            '', 'Only pair strain direction changed, old coefficient magnitude retained. All center forces, initial-area mobility,20 rounds,5%/40% movement,1.03/1.5 shape allowances, original full planner/config unchanged. All member results retained;5 identical preflight results reused read-only. Full trajectories and each-operation logs saved.',
            '', 'Training-source development only; no new C3 inference in timing, no complete-latency acceptance or dev quality claim; test sealed.']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(dict(status='complete',summary=s),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run(args.resume)
    except BaseException as error:
        out=PREFLIGHT if args.preflight else OUT
        save(out/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(error),pid=os.getpid(),no_automatic_retry=True,test_read=False));raise
