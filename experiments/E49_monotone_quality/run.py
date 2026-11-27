"""Versioned E49 development gates and panels; Q is an offline observer only."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='4'
os.environ['NUMBA_NUM_THREADS']='1'
import argparse,json,hashlib,sys,time,zipfile
from pathlib import Path
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts,public_hard_reference
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from scoring.checks import independent_hard_checks
from .model import GeometryModel
from .inference import predict
from .canonical_native import warmup
from .baseline.canonical_native import warmup as baseline_warmup
from .monotone import optimize

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
MODEL=ROOT/'experiments/E42_joint_regularization/runs/20261023_rectangle_w96d12_ab_seed0';CHECKPOINT=MODEL/'checkpoints/best.pt'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,p)
def summarize(records):
    q=np.array([r['selected_metrics']['cost'] for r in records]);b=np.array([r['baseline_Q'] for r in records]);n=np.array([r['n'] for r in records]);t=np.array([r['full_seconds'] for r in records])
    w=np.exp((n-n.max())/12);weighted=float(np.average(q,weights=w));base=float(np.average(b,weights=w))
    return {'count':len(records),'hard_legal':sum(r['selected_metrics']['is_feasible'] for r in records),
        'weighted_Q':weighted,'baseline_weighted_Q':base,'excess_Q_reduction_fraction':(base-weighted)/max(base-1,1e-12),
        'mean_Q':float(q.mean()),'p95_Q':float(np.quantile(q,.95)),'max_Q':float(q.max()),
        'regression_indices':[r['index'] for r in records if r['selected_metrics']['cost']>r['baseline_Q']+1e-8],
        'over1_2_indices':[r['index'] for r in records if r['selected_metrics']['cost']>1.2],
        'Q_lt1_03':int(sum(q<1.03)),'mean_seconds':float(t.mean()),'max_seconds':float(t.max()),'over5s':int(sum(t>5.)),
        'passed':len(records)==100 and bool((q<=b+1e-8).all()) and sum(q>1.2)<=1 and weighted<=1+.7*(base-1) and bool((t<=5.).all()),
        'scope':'adaptive development; historical comparison and fresh ordinary certificate separately recorded'}
def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--indices',default='164,443,246,435,21');ap.add_argument('--screen',action='store_true')
    ap.add_argument('--profile',default='v1');ap.add_argument('--resume',action='store_true');ap.add_argument('--guarded',action='store_true');args=ap.parse_args()
    out=args.output.resolve();assert out.is_relative_to(HERE/'runs');out.mkdir(parents=True,exist_ok=True);(out/'cases').mkdir(exist_ok=True)
    ids=read(ROOT/'experiments/E48_quality_potential/runs/20261024_screen100_frozen_v4/config.json')['indices'] if args.screen else [int(x) for x in args.indices.split(',')]
    data=ReferenceLayouts('dev_monitor500');files=[p for p in HERE.rglob('*.py') if 'runs' not in p.relative_to(HERE).parts]
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    config={'indices':ids,'profile':args.profile,'source_sha256':sources,'checkpoint_sha256':sha(CHECKPOINT),
        'registry_sha256':sha(ROOT/'protocol/reference_workset_v5.json'),'command':sys.argv,'threads':4,'case_seconds':5.,'test_read':False,
        'guarded':args.guarded,'inference_threads':2,'candidate_Q':'offline, outside solve; selected official scoring inside solve timer'}
    assert config['checkpoint_sha256']=='12878108e9a58b8c195b314ba6586686c2a6e2796c1f4b6443e102b4db4224a6'
    if (out/'config.json').exists():
        assert args.resume,'Use --resume for existing output';old=read(out/'config.json')
        for key in ('source_sha256','indices','checkpoint_sha256','registry_sha256','profile'):assert old[key]==config[key],key
    else:
        save(out/'config.json',config)
        with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.relative_to(ROOT))
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0)
    setup=time.perf_counter();warmup();baseline_warmup();official_evaluator()
    checkpoint=torch.load(CHECKPOINT,map_location='cpu',weights_only=False);model=GeometryModel(**checkpoint['model_config']).eval();model.load_state_dict(checkpoint['model'])
    save(out/'environment.json',{'setup_seconds':time.perf_counter()-setup,'torch_threads':torch.get_num_threads()})
    baseline={r['index']:r for r in read(MODEL/'terminal/results.json')['records']};records=[]
    resident=None
    if args.guarded:
        from .resident import ResidentPotential
        resident=ResidentPotential(CHECKPOINT);resident.ensure_ready()
    for index in ids:
        file=out/'cases'/f'{index:04d}.json'
        if file.exists():records.append(read(file));continue
        if (out/'STOP').exists() or (HERE/'runs/STOP').exists():break
        item=data[index];case=item['case'];assert list(item['member'])==baseline[index]['member']
        case.reference_positions=public_hard_reference(case.solver_inputs[1],case.solver_inputs[5],case.solver_inputs[6])
        begin=time.perf_counter()
        def observe(x,row):
            checked=independent_hard_checks(x.tolist(),case.solver_inputs[1],case.solver_inputs[5],case.reference_positions)
            assert checked['is_feasible'];row['independent_checks']=checked;row['checked_seconds']=time.perf_counter()-begin
        if resident is not None:
            positions,stats=resident.solve(case.solver_inputs,case=case,profile=args.profile)
            draft=np.asarray(stats['raw_positions']);inference=stats['inference_seconds'];elapsed=stats['seconds']
            choice=next(c for c in stats['candidates'] if c['id']==stats['selected_id'])
        else:
            draft,inference=predict(model,case.solver_inputs)
            positions,stats=optimize(draft,case.solver_inputs,seconds=max(.1,5-(time.perf_counter()-begin)-.2),observer=observe,profile=args.profile)
            choice=stats['candidates'][stats['selected_id']]
            choice['metrics']=evaluate_case(case,positions.tolist(),time.perf_counter()-begin)['metrics'];elapsed=time.perf_counter()-begin
        offline=time.perf_counter()
        for candidate in stats['candidates']:
            if 'metrics' not in candidate:candidate['metrics']=evaluate_case(case,candidate['positions'],elapsed)['metrics']
            if 'soft_violations' in candidate:
                assert sum(candidate['soft_violations'])==candidate['metrics']['total_soft_violations'],'Public soft-count mismatch'
        row={'index':index,'member':list(item['member']),'n':case.block_count,'draft':draft.tolist(),'inference_seconds':inference,
            'full_seconds':elapsed,'offline_diagnostics_seconds':time.perf_counter()-offline,'selected_metrics':choice['metrics'],'baseline_Q':baseline[index]['metrics']['cost'],'solver':stats}
        save(file,row);records.append(row);save(out/'summary.json',summarize(records))
        fresh=stats['candidates'][stats['baseline_id']]['metrics']['cost'] if stats['baseline_id'] is not None else 10.
        print(f"{len(records)}/{len(ids)} index{index} Q{choice['metrics']['cost']:.6f} old{row['baseline_Q']:.6f} fresh{fresh:.6f} {elapsed:.3f}s candidates{len(stats['candidates'])} safe{stats['safe_candidates']}",flush=True)
    if resident is not None:resident.close()
    save(out/'execution.json',{'complete':len(records)==len(ids),'count':len(records),'requested':len(ids)})
    if records:print(summarize(records),flush=True)
if __name__=='__main__':main()
