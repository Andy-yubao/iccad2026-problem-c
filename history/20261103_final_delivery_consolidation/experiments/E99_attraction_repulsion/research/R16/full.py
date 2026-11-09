"""Full original-planner training controls with a fixed pure AR prefix."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import argparse,datetime,json,time,zipfile
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from datasets.paths import ROOT
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from ...tune import process_lock
from ...planner.solver import optimize
from ...planner.canonical_native import warmup
from .operator import preprocess,variants
from . import replay

HERE=Path(__file__).resolve().parent;RESEARCH=HERE.parent;EXPERIMENT=RESEARCH.parent
PLAN=replay.read(HERE/'full_plan.json');CONTROL=replay.CONTROL;TRAJECTORIES=replay.OUT
OUT=EXPERIMENT/PLAN['output'];PREFLIGHT=EXPERIMENT/PLAN['preflight_output']
AR_CONTROL=EXPERIMENT/'runs/research/20261103_R15_AR_original_planner_train500'
read=replay.read;sha=replay.sha;save=replay.save


def sources():
    current=replay.sources()
    for p in [Path(__file__),HERE/'full_plan.json',*[ROOT/s for s in ('scoring/checks.py','scoring/evaluation.py','scoring/official.py')]]:
        current[p.relative_to(ROOT).as_posix()]=sha(p)
    return current

def register(out):
    assert read(TRAJECTORIES/'results.json')['status']=='complete'
    identity=dict(plan=PLAN,sources=sources(),planner_frozen=replay.check_planner(),
        trajectory_registration_sha256=sha(TRAJECTORIES/'registration.json'),trajectory_results_sha256=sha(TRAJECTORIES/'results.json'),
        baseline_registration_sha256=sha(CONTROL/'registration.json'),baseline_results_sha256=sha(CONTROL/'results.json'),
        AR_control_registration_sha256=sha(AR_CONTROL/'registration.json'),AR_control_results_sha256=sha(AR_CONTROL/'results.json'),
        registry_sha256=sha(ROOT/'protocol/reference_workset_v5.json'),configs=read(TRAJECTORIES/'registration.json')['identity']['configs'])
    if (out/'registration.json').exists():assert read(out/'registration.json')['identity']==identity
    else:
        out.mkdir(parents=True,exist_ok=True);save(out/'registration.json',dict(identity=identity,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        with zipfile.ZipFile(out/'source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in identity['sources']:z.write(ROOT/p,p)
    return identity

def hard(x,inputs):
    h=independent_hard_checks(x.tolist(),inputs[1],inputs[5],public_hard_reference(inputs[1],inputs[5],inputs[6]))
    assert h['is_feasible'] and np.isfinite(x).all() and (x[:,:2]>=0).all()
    return h

def calculate(index,arm,row,options,out):
    draft,inputs,old=replay.cached(index);assert old['member']==list(row['member'])
    control_arm=PLAN['matched_controls'][arm];control_path=AR_CONTROL/'cases'/control_arm/f'{index:04d}.json';AR_control=read(control_path)
    assert AR_control['member']==old['member'] and AR_control['n']==old['n']
    diag=read(TRAJECTORIES/'cases'/arm/f'{index:04d}.json');path=TRAJECTORIES/'trajectories'/arm/f'{index:04d}.npz'
    assert sha(path)==diag['trajectory_sha256']
    with np.load(path) as a:expected=a['history'][-1].copy()
    start=time.perf_counter();prefix,ps=preprocess(draft,inputs,variants()[arm],capture=False)
    final,trace=optimize(prefix,inputs,**options);h=hard(final,inputs);wall=time.perf_counter()-start
    np.testing.assert_array_equal(prefix,expected)
    scored=evaluate_case(row['case'],final,wall)
    coords=out/'coordinates'/arm/f'{index:04d}.npz';coords.parent.mkdir(parents=True,exist_ok=True);temp=coords.with_suffix('.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,final=final)
    temp.replace(coords)
    result=dict(index=index,member=old['member'],n=int(inputs[0]),arm=arm,metrics=scored['metrics'],hard=h,
        processing_seconds=wall,preprocessor_stats=ps,planner_trace=trace,coordinates_sha256=sha(coords),
        coordinate_source=str(coords.relative_to(ROOT)),trajectory_source=str(path.relative_to(ROOT)),trajectory_sha256=diag['trajectory_sha256'],
        control_Q=old['metrics']['cost'],AR_control_Q=AR_control['metrics']['cost'],AR_control_arm=control_arm,AR_control_record_source=str(control_path.relative_to(ROOT)),AR_control_record_sha256=sha(control_path),control_complete_seconds=old['complete_seconds'],control_draft_seconds=old['draft_seconds'],
        timing_scope='fresh_CPU_AR_original_planner_hard_checks; cached_C3_no_complete_latency_claim',
        raw_geometry=diag['diagnostics']['final_geometry'])
    save(out/'cases'/arm/f'{index:04d}.json',result)
    return result

def stopped(out):return (RESEARCH/'STOP').exists() or (out/'STOP').exists()

def preflight():
    if PREFLIGHT.exists():raise FileExistsError('Preflight protected')
    identity=register(PREFLIGHT);data=ReferenceLayouts('train_monitor500');options=read(EXPERIMENT/'planner_config.json');records=[]
    with process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1);draft,inputs,_=replay.cached(0);cold=time.perf_counter();preprocess(draft,inputs,capture=False);warmup();cold=time.perf_counter()-cold
        for index in PLAN['preflight']['indices']:
            row=data[index]
            for arm in PLAN['arms']:
                if stopped(PREFLIGHT):raise RuntimeError('STOP')
                records.append(calculate(index,arm,row,options,PREFLIGHT))
    assert sources()==identity['sources']
    save(PREFLIGHT/'results.json',dict(status='passed',unique_cases=5,records=records,source_sha256=identity['sources'],cold_seconds=cold,test_read=False))
    print(json.dumps(dict(status='passed',unique_cases=5,original_planner_arm_cases=len(records),all_hard_legal=True,cold_seconds=cold)))

def summary(records):
    output={}
    for arm in PLAN['arms']:
        rows=[r for r in records if r['arm']==arm];q=np.array([r['metrics']['cost'] for r in rows]);old=np.array([r['control_Q'] for r in rows]);wall=np.array([r['processing_seconds'] for r in rows])
        output[arm]=dict(count=len(rows),weighted_Q=official_evaluator().compute_total_score(q.tolist(),[r['n'] for r in rows]),
            hard_legal=sum(r['metrics']['is_feasible'] and r['hard']['is_feasible'] for r in rows),
            matched_AR_control_arm=PLAN['matched_controls'][arm],matched_AR_control_weighted_Q=official_evaluator().compute_total_score([r['AR_control_Q'] for r in rows],[r['n'] for r in rows]),
            compared_to_matched_AR=dict(better=int(sum(r['metrics']['cost']<r['AR_control_Q']-1e-10 for r in rows)),equal=int(sum(abs(r['metrics']['cost']-r['AR_control_Q'])<=1e-10 for r in rows)),worse=int(sum(r['metrics']['cost']>r['AR_control_Q']+1e-10 for r in rows))),
            compared_to_original_no_AR=dict(better=int((q<old-1e-10).sum()),equal=int((np.abs(q-old)<=1e-10).sum()),worse=int((q>old+1e-10).sum())),
            processing_seconds=dict(mean=float(wall.mean()),p95=float(np.percentile(wall,95)),max=float(wall.max())),
            CPU_processing_over250ms=int((wall>=.25).sum()),preprocessor_mean_seconds=float(np.mean([r['preprocessor_stats']['seconds'] for r in rows])),
            mean_planner_shape_calls=float(np.mean([sum(t['phase']=='shape' for t in r['planner_trace']['trace']) for r in rows])),
            deep_packed_refine_calls=sum(t['phase']=='full_packed_refine' for r in rows for t in r['planner_trace']['trace']),
            mean_boundary_violations=float(np.mean([r['metrics']['boundary_violations'] for r in rows])),
            mean_grouping_violations=float(np.mean([r['metrics']['grouping_violations'] for r in rows])),
            mean_mib_violations=float(np.mean([r['metrics']['mib_violations'] for r in rows])))
    return output

def run(resume=False):
    pre=read(PREFLIGHT/'results.json');assert pre['status']=='passed' and pre['source_sha256']==sources()
    if OUT.exists() and not resume:raise FileExistsError('Explicit --resume required')
    identity=register(OUT);data=ReferenceLayouts('train_monitor500');options=read(EXPERIMENT/'planner_config.json');records=[];begin=time.perf_counter()
    pre_ids=set(PLAN['preflight']['indices'])
    with process_lock(RESEARCH/'.numerical.lock'),threadpool_limits(limits=1):
        import numba
        numba.set_num_threads(1);draft,inputs,_=replay.cached(0);cold=time.perf_counter();preprocess(draft,inputs,capture=False);warmup();cold=time.perf_counter()-cold
        for index in range(500):
            row=None
            order=PLAN['arms'][index%4:]+PLAN['arms'][:index%4]
            for arm in order:
                if stopped(OUT):save(OUT/'state.json',dict(status='stopped',completed=len(records),total=2000,test_read=False));return
                existing=OUT/'cases'/arm/f'{index:04d}.json'
                if existing.exists():
                    record=read(existing);assert record['member']==data.records[index]['member'] and sha(ROOT/record['coordinate_source'])==record['coordinates_sha256']
                elif index in pre_ids:
                    path=PREFLIGHT/'cases'/arm/f'{index:04d}.json';record=read(path)
                    assert record['member']==data.records[index]['member'] and sha(ROOT/record['coordinate_source'])==record['coordinates_sha256']
                    record=dict(record,reused_same_identity_preflight=True,preflight_record_source=str(path.relative_to(ROOT)),preflight_record_sha256=sha(path));save(existing,record)
                else:
                    if row is None:row=data[index]
                    record=calculate(index,arm,row,options,OUT)
                records.append(record)
                save(OUT/'state.json',dict(status='running',pid=os.getpid(),completed=len(records),total=2000,elapsed_seconds=time.perf_counter()-begin,test_read=False))
            if (index+1)%25==0:print('R16 original planner',index+1,'/500 x4',flush=True)
        assert sources()==identity['sources'];summaries=summary(records);control=read(CONTROL/'results.json')
        result=dict(status='complete',panel='train_monitor500',original_no_AR_weighted_Q=control['weighted_Q'],summary=summaries,
            records=records,fresh_new_arm_solves=sum(not r.get('reused_same_identity_preflight',False) for r in records),
            reused_same_identity_preflight_cases=sum(r.get('reused_same_identity_preflight',False) for r in records),
            cold_native_seconds=cold,original_planner_unchanged=True,source_sha256=identity['sources'],test_read=False,new_training=False,
            timing_scope='CPU cached-draft processing, no C3 inference; historical no-AR timing unpaired; not complete250ms acceptance')
        save(OUT/'results.json',result);save(OUT/'state.json',dict(status='complete',completed=2000,total=2000,test_read=False))
        study=read(RESEARCH/'study.json');study.update(status='R16_full_original_planner_controls_complete_review',current_run=None,active_pid=None)
        study['R16_full_controls']=dict(run=PLAN['output'],summary=summaries,original_no_AR_weighted_Q=control['weighted_Q']);save(RESEARCH/'study.json',study)
        lines=['# 纯吸引排斥前置＋原完整后处理：训练来源控制','',f"原无AR完整控制Q={control['weighted_Q']:.9f}。相同C3/public输入、原planner与配置逐字节冻结；每臂500成员全部保留。",'',
            '| 方案 | 加权Q | 硬合法 | CPU处理均时ms | 最大ms | 原规划shape调用均数 |','|---|---:|---:|---:|---:|---:|']
        for arm,s in summaries.items():lines.append(f"| {arm} | {s['weighted_Q']:.9f} | {s['hard_legal']}/500 | {1000*s['processing_seconds']['mean']:.3f} | {1000*s['processing_seconds']['max']:.3f} | {s['mean_planner_shape_calls']:.3f} |")
        lines+=['', '| 方案 | 对应R15控制 | 控制Q | 加权Q变化 | 逐例改善/相同/变差 |', '|---|---|---:|---:|---|']
        for arm,s in summaries.items():
            paired=s['compared_to_matched_AR']
            lines.append(f"| {arm} | {s['matched_AR_control_arm']} | {s['matched_AR_control_weighted_Q']:.9f} | {s['weighted_Q']-s['matched_AR_control_weighted_Q']:+.9f} | {paired['better']}/{paired['equal']}/{paired['worse']} |")
        lines+=['', '仅算子参数/预算耗尽处理不同，原后处理器未改。20条同身份实施记录只读复用，其余1980臂例本阶段新求解；原无AR控制质量只读复用，历史计时不配对。CPU阶段未包含新C3推理，不能作为完整250ms目标验收。','', '这些是训练来源机制开发，不是dev500或独立泛化成绩；测试封存。']
        (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','source_sha256')},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    try:preflight() if args.preflight else run(args.resume)
    except BaseException as e:
        out=PREFLIGHT if args.preflight else OUT
        save(out/('error_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json'),dict(error=repr(e),pid=os.getpid(),test_read=False));raise
