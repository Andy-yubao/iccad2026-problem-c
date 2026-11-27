"""Frozen development evaluation with independent official checks and resumable evidence."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['NUMBA_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from .model import GeometryModel
from .inference import predict
from .canonical_native import warmup
from .solver import optimize,Archive
from .backend.frozen.packing import prepare_problem
from .backend.frozen.feasible import row_control
from .backend.frozen.solver import hard_valid

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
MODEL=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0'
CHECKPOINT=MODEL/'checkpoints/best.pt';CHECKPOINT_SHA='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
GATE=[0,50,164,443,21]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)


def load_record(path):
    value=read(path)
    if 'reused_from' in value:
        original=(ROOT/value['reused_from']).resolve()
        if not original.is_relative_to(HERE/'runs') or sha(original)!=value['source_sha256']:raise ValueError('Reused record changed')
        return read(original)
    return value


def synthetic_gate():
    c=np.array([[0,1,0,0,0],[1,0,0,0,0],[0,0,0,0,0]])
    specs=np.array([[0.,0.,2.,2.],[-1.,-1.,3.,1.],[-1.,-1.,-1.,-1.]])
    inputs=(3,np.array([4.,3.,2.]),np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),c,specs)
    p=prepare_problem(*inputs);x=row_control(p);assert hard_valid(x,p)
    archive=Archive(p,x,time.perf_counter(),None);assert archive.accept(x,'valid') is not None
    bad=x.copy();bad[1,2:]=[1.,3.];assert archive.accept(bad,'rotated_fixed') is None
    bad=x.copy();bad[0,0]+=.01;assert archive.accept(bad,'moved_preplaced') is None
    bad=x.copy();bad[2,0]=-.01;assert archive.accept(bad,'negative') is None
    hidden=specs.copy();hidden[2]=[999,777,10,10];hidden[1,:2]=[888,555]
    masked=prepare_problem(*inputs[:-1],hidden);assert np.all(masked.specs[2]==-1) and np.all(masked.specs[1,:2]==-1)
    solved,stats=optimize(x,inputs,seconds=.4);assert hard_valid(solved,p)
    assert np.array_equal(solved[0],specs[0]) and np.array_equal(solved[1,2:],specs[1,2:])
    return {'passed':True,'nonsquare_fixed':True,'preplaced':True,'negative_rejected':True,
            'private_fields_ignored':True,'real_controller_on_synthetic':True,'seconds':stats['seconds']}


def summarize(rows,output,stage):
    n=np.array([r['n'] for r in rows]);q=np.array([r['selected_metrics']['cost'] for r in rows]);t=np.array([r['full_solve_seconds'] for r in rows])
    allq=np.array([min([c['metrics']['cost'] for c in r['solver']['candidates']],default=10.) for r in rows])
    distance=np.array([r['solver']['candidates'][r['solver']['selected_id']]['distance']['D'] for r in rows])
    targetrows=[]
    for r in rows:
        hit=[c for c in r['solver']['candidates'] if c['metrics']['cost']<1.03 and not c['shelf_fallback']]
        targetrows.append({'index':r['index'],'selected_target_met':r['selected_metrics']['cost']<1.03,
            'candidate_target_met':bool(hit),'first_Q_target_full_seconds':min((c['score_complete_seconds'] for c in hit),default=None),
            'min_found_D_for_Q_target':min((c['distance']['D'] for c in hit),default=None)})
    result={'stage':stage,'count':len(rows),'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in rows),
        'selected_Q_lt1_03':int(sum(q<1.03)),'all_selected_Q_lt1_03':bool((q<1.03).all()),
        'weighted_Q':float(np.average(q,weights=np.exp((n-n.max())/12))),
        'mean_Q':float(q.mean()),'p95_Q':float(np.quantile(q,.95)),'max_Q':float(q.max()),
        'worst5_mean_Q':float(np.sort(q)[-max(1,int(np.ceil(.05*len(q)))):].mean()),
        'posthoc_best_candidate_Q_lt1_03':int(sum(allq<1.03)),
        'posthoc_best_candidate_max_Q':float(allq.max()),
        'mean_full_seconds':float(t.mean()),'max_full_seconds':float(t.max()),'over5s_count':int(sum(t>5.)),
        'shelf_selected':sum(r['solver']['shelf_selected'] for r in rows),'mean_selected_D':float(distance.mean()),
        'candidate_count_total':sum(len(r['solver']['candidates']) for r in rows),
        'selected_source_counts':{s:sum(r['solver']['candidates'][r['solver']['selected_id']]['source']==s for r in rows)
            for s in sorted({r['solver']['candidates'][r['solver']['selected_id']]['source'] for r in rows})},
        'baseline_weighted_Q_same_cases':float(np.average([r['baseline']['Q'] for r in rows],weights=np.exp((n-n.max())/12))),
        'per_case_target_diagnostics':targetrows,'test_read':False,'formal_contest_score':False,
        'scope':'development, input-only online selection; posthoc candidateQ is diagnosis only, not deployed selection',
        'ready_for_screen100':stage=='gate' and len(rows)==5 and all(r['selected_metrics']['is_feasible'] for r in rows) and not any(r['solver']['shelf_selected'] for r in rows)}
    save(output/'summary.json',result);return result


def run(args):
    output=args.output.resolve()
    if not output.is_relative_to(HERE/'runs'):raise ValueError('Output must remain in E48/runs')
    if args.stage=='screen' and not read(args.gate/'summary.json')['ready_for_screen100']:raise RuntimeError('Five-case legality/fallback gate not passed')
    torch.manual_seed(0);torch.set_num_threads(4);torch.set_num_interop_threads(1)
    dataset=ReferenceLayouts('dev_monitor500')
    indices=GATE if args.stage=='gate' else GATE+random.Random(20261024).sample([i for i in range(500) if i not in GATE],95)
    sourcefiles=sorted(HERE.rglob('*.py'));sourcefiles=[p for p in sourcefiles if 'runs' not in p.parts]
    sources={str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sourcefiles}
    if args.stage=='screen':
        gate_identity=read(args.gate/'config.json')
        for key,value in gate_identity['source_sha256'].items():
            if key.endswith('/run.py'):continue  # harness-only reuse addition; search kernels must match exactly
            if sources.get(key)!=value:raise ValueError('Screen search differs from checked gate: '+key)
    config={'stage':args.stage,'indices':indices,'members':[dataset.records[i] for i in indices],
        'selection':'fixed hard-case gate' if args.stage=='gate' else 'fixed5 hard cases plus seed20261024 uniform95 of remaining495; targeted development panel',
        'source_sha256':sources,'checkpoint_sha256':sha(CHECKPOINT),'registry_sha256':sha(ROOT/'protocol/reference_workset_v5.json'),
        'split_sha256':sha(ROOT/'protocol/split_v1.json'),'threads':4,'case_budget_seconds':5.,'seed':20261024,
        'timing':'public graph features + CPU inference + all optimizer branches/checks/official candidate scoring + selection; setup and disk IO separate',
        'observer_return_used':False,'online_Q_or_reference_denominators_used':False,
        'command':subprocess.list2cmdline(sys.argv),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
        'evaluation_budget_seconds':900.,'test_read':False}
    assert config['checkpoint_sha256']==CHECKPOINT_SHA
    output.mkdir(parents=True,exist_ok=True);(output/'cases').mkdir(exist_ok=True)
    if (output/'config.json').exists():
        old=read(output/'config.json')
        for k in ('source_sha256','indices','checkpoint_sha256','registry_sha256','split_sha256'):
            if old[k]!=config[k]:raise ValueError('Resume identity changed: '+k)
        if not args.resume:raise FileExistsError('Use --resume for this existing run')
    else:
        save(output/'config.json',config)
        with zipfile.ZipFile(output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in sourcefiles:z.write(p,p.relative_to(ROOT))
    setup=time.perf_counter();warmup();official_evaluator()
    state=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'],strict=True)
    check=synthetic_gate()
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
    except ImportError:
        pools={'available':False,'reason':'optional threadpoolctl is not installed; thread caps are explicit in environment/Torch/solver settings'}
    save(output/'environment.json',{'setup_seconds':time.perf_counter()-setup,'synthetic_gate':check,'torch_threads':torch.get_num_threads(),
        'threadpools':pools,'numba_threads_env':os.environ['NUMBA_NUM_THREADS'],'python':sys.version})
    baseline=read(MODEL/'terminal/results.json')['records'];baseline={r['index']:r for r in baseline}
    records=[];started=time.perf_counter()
    for ordinal,index in enumerate(indices):
        file=output/'cases'/f'{index:04d}.json'
        if file.exists():records.append(load_record(file));continue
        if args.stage=='screen' and index in GATE:
            original=args.gate/'cases'/f'{index:04d}.json'
            save(file,{'index':index,'reused_from':original.resolve().relative_to(ROOT).as_posix(),'source_sha256':sha(original)})
            records.append(read(original));continue
        if (HERE/'runs/STOP').exists() or (output/'STOP').exists():break
        if time.perf_counter()-started>900:break
        item=dataset[index];case=item['case'];case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
        begin=time.perf_counter();draft,inference_seconds=predict(model,case.solver_inputs)
        def observer(positions,row):
            result=evaluate_case(case,positions.tolist(),time.perf_counter()-begin)
            if not result['metrics']['is_feasible']:raise AssertionError('Candidate passed internal but failed independent official hard checks')
            row['metrics']=result['metrics'];row['independent_hard_checks']=result['independent_hard_checks']
            row['score_complete_seconds']=time.perf_counter()-begin
        final,stats=optimize(draft,case.solver_inputs,seconds=max(.1,5.-(time.perf_counter()-begin)-.02),observer=observer)
        elapsed=time.perf_counter()-begin;selected=stats['candidates'][stats['selected_id']]
        assert np.array_equal(final,np.asarray(selected['positions']))
        r={'index':index,'member':list(item['member']),'n':case.block_count,'raw_positions':draft.tolist(),
            'inference_seconds':inference_seconds,'full_solve_seconds':elapsed,'selected_metrics':selected['metrics'],
            'solver':stats,'baseline':{'source':'E42 frozen terminal500, existing record; different thread/concurrency timing not compared',
                'Q':baseline[index]['metrics']['cost'],'member':baseline[index]['member']}}
        assert r['member']==r['baseline']['member'];save(file,r);records.append(r)
        summary=summarize(records,output,args.stage);save(output/'progress.json',{'done':len(records),'total':len(indices),'last_index':index,'updated_utc':datetime.now(timezone.utc).isoformat()})
        print(f"{ordinal+1}/{len(indices)} index{index} n{case.block_count} Q{selected['metrics']['cost']:.6f} D{selected['distance']['D']:.3f} {elapsed:.3f}s candidates{len(stats['candidates'])} source{selected['source']}",flush=True)
        if not selected['metrics']['is_feasible']:break
    summary=summarize(records,output,args.stage)
    save(output/'execution.json',{'complete':len(records)==len(indices),'seconds_this_invocation':time.perf_counter()-started,'count':len(records)})
    print(json.dumps({k:v for k,v in summary.items() if k!='per_case_target_diagnostics'},ensure_ascii=True),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--stage',choices=('gate','screen'),default='gate')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--resume',action='store_true')
    ap.add_argument('--gate',type=Path,default=HERE/'runs/20261024_gate5_v4')
    run(ap.parse_args())
