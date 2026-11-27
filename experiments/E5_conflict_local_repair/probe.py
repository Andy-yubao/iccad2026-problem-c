"""E5 same-initialization, equal-incremental-budget input-only repair probe."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from datasets.cases import load_frozen_cases
from datasets.paths import resolved_data_paths
from scoring.evaluation import evaluate_case, summarize_variants
from experiments.conflict_local_repair.geometry import Objective, is_legal, reinsertion_candidates
from experiments.conflict_local_repair.packing import prepare_problem


def conflict_scores(p, positions):
    """Input-only wire hotspots plus offending boundary/group components."""
    centers = positions[:,:2] + positions[:,2:]/2
    scores = (p.adjacency * np.abs(centers[:,None]-centers[None]).sum(axis=2)).sum(axis=1)
    scores /= max(p.total_wire_weight*p.scale, 1e-9)
    lo, hi = positions[:,:2].min(axis=0), (positions[:,:2]+positions[:,2:]).max(axis=0)
    touch = np.c_[abs(positions[:,0]-lo[0])<1e-6,
                  abs(positions[:,0]+positions[:,2]-hi[0])<1e-6,
                  abs(positions[:,1]+positions[:,3]-hi[1])<1e-6,
                  abs(positions[:,1]-lo[1])<1e-6]
    required = (p.constraints[:,4,None] & np.array([1,2,4,8])) != 0
    scores += np.any(required & ~touch,axis=1)
    for group in np.unique(p.constraints[:,3]):
        if group <= 0: continue
        ids = np.flatnonzero(p.constraints[:,3]==group)
        rect = positions[ids]
        ov = np.minimum((rect[:,:2]+rect[:,2:])[:,None],(rect[:,:2]+rect[:,2:])[None])-np.maximum(rect[:,None,:2],rect[None,:,:2])
        adjacent = ((ov[:,:,0]>=0)&(ov[:,:,1]>0))|((ov[:,:,0]>0)&(ov[:,:,1]>=0))
        visited = {0}; pending = [0]
        while pending:
            new = set(np.flatnonzero(adjacent[pending.pop()])).difference(visited)
            pending.extend(new); visited.update(new)
        if len(visited)!=len(ids): scores[ids] += 1
    return scores


def repair(inputs, initial, strategy, seconds=3, seed=0):
    """External rectangles and anchors never change under local variants."""
    start = time.monotonic(); deadline = start + seconds
    p = prepare_problem(*inputs)
    pos = np.asarray(initial,dtype=float).copy()
    if not is_legal(pos,p): raise ValueError('Probe requires a hard-legal frozen initialization')
    objective = Objective(p)
    before = objective(pos)
    rng = np.random.default_rng(seed)
    movable = np.flatnonzero(~p.preplaced)
    k = min(12,len(movable))
    if strategy == 'conflict': region = movable[np.argsort(-conflict_scores(p,pos)[movable],kind='stable')[:k]]
    elif strategy == 'random': region = rng.choice(movable,k,replace=False)
    elif strategy == 'global': region = movable
    else: raise ValueError(strategy)
    selected_seconds = time.monotonic()-start
    current_score = before
    attempts=accepted=checks=0
    # Coordinate reinsertion is a deliberately conservative local reconstruction.
    while len(region) and time.monotonic()<deadline-.03:
        i = int(rng.choice(region))
        wh = pos[i,2:].copy()
        candidates = reinsertion_candidates(pos,i,wh,p,rng,True)
        old = pos[i].copy()
        best_score,best_rect = current_score,old
        for rect in candidates:
            if time.monotonic()>=deadline-.02: break
            pos[i]=rect
            value=objective(pos)
            if value<best_score:
                checks+=1
                if is_legal(pos,p): best_score,best_rect=value,rect.copy()
        pos[i]=best_rect
        attempts+=1
        if best_score<current_score: accepted+=1; current_score=best_score
    outside = np.setdiff1d(np.arange(p.n),region)
    unchanged = bool(np.array_equal(pos[outside],np.asarray(initial)[outside]))
    if not unchanged or not is_legal(pos,p): raise AssertionError('Repair violated frozen context/global legality')
    elapsed=time.monotonic()-start
    return pos, {'solve_seconds':elapsed,'selection_seconds':selected_seconds,'region':region.tolist(),
                 'selected_block_count':len(region),'actually_changed_block_count':int(np.any(pos!=initial,axis=1).sum()),
                 'outside_unchanged':unchanged,'attempts':attempts,'accepted':accepted,'global_hard_checks':checks+1,
                 'initial_surrogate':before,'final_surrogate':current_score,
                 'surrogate_improvement_per_second':(before-current_score)/max(elapsed,1e-9),
                 'budget_overrun':elapsed>seconds,'fallback':False,
                 'scope':'coordinate-only reinsertion; fixed shapes; no joint region removal or learned reconstruction'}


def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial',type=Path,default=ROOT/'experiments/baseline/runs/stage1_dev_run1/results.json')
    parser.add_argument('--initial-variant',default='search')
    parser.add_argument('--seconds',type=float,default=3)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if not 0<args.seconds<=15: parser.error('Probe is limited to 0..15 seconds/variant/case')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    sources=[*Path(__file__).parent.glob('*.py'),*(ROOT/'datasets').glob('*.py'),*(ROOT/'scoring').glob('*.py'),ROOT/'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py']
    config={'command':[sys.executable,*sys.argv],'created_utc':datetime.now(timezone.utc).isoformat(),
            'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            'source_recovery':'source_snapshot.zip','initial':str(args.initial.resolve()),
            'initial_sha256':hashlib.sha256(args.initial.read_bytes()).hexdigest(),
            'initial_variant':args.initial_variant,'seed':args.seed,'seconds':args.seconds,'cpu_threads':1,
            'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
            'resolved_data_paths':resolved_data_paths(),
            'split_sha256':hashlib.sha256((ROOT/'protocol/split_v1.json').read_bytes()).hexdigest(),
            'upstream':json.loads((ROOT/'third_party/upstream_manifest.json').read_text())['commit'],
            'budget_scope':'equal incremental seconds; historical initialization cost excluded for all; not end-to-end 15s comparison',
            'hypothesis':'Conflict targeting beats random same-size targeting and global feasible search from identical initialization.',
            'learning_gate':'Need weighted Q gain plus >=3/5 same-direction improvements against random/global; no automatic training.'}
    write(args.output/'config.json',config)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sources: z.write(p,p.relative_to(ROOT))
    frozen=json.loads(args.initial.read_text(encoding='utf-8'))
    initial_by_id={c['case_id']:c['variants'][args.initial_variant]['positions'] for c in frozen['cases']}
    records=[]; strategies=['initial','conflict','random','global']
    results={'status':'running','cases':records,'scope':config['budget_scope']}
    for case in load_frozen_cases('dev'):
        initial=np.array(initial_by_id[case.case_id],dtype=float)
        row={'case_id':case.case_id,'source':case.source,'input_summary':case.input_summary(),'variants':{}}
        row['variants']['initial']={**evaluate_case(case,initial,0),'solve_seconds':0}
        for strategy in strategies[1:]:
            positions,stats=repair(case.solver_inputs,initial,strategy,args.seconds,args.seed)
            record={**evaluate_case(case,positions,stats['solve_seconds']),'solve_seconds':stats['solve_seconds'],'diagnostics':stats}
            initial_metrics=row['variants']['initial']['metrics']; metrics=record['metrics']
            record['Q_improvement_per_second']=(initial_metrics['cost']-metrics['cost'])/max(stats['solve_seconds'],1e-9)
            record['soft_damage_global']={k:metrics[k]-initial_metrics[k] for k in ('boundary_violations','grouping_violations','mib_violations')}
            row['variants'][strategy]=record
            print(case.case_id,strategy,metrics['is_feasible'],round(metrics['cost'],6),flush=True)
        records.append(row); write(args.output/'results.json',results)
    results['summary']=summarize_variants(records,strategies)
    results['wins_vs_random']=sum(r['variants']['conflict']['metrics']['cost']<r['variants']['random']['metrics']['cost'] for r in records)
    results['wins_vs_global']=sum(r['variants']['conflict']['metrics']['cost']<r['variants']['global']['metrics']['cost'] for r in records)
    results['wall_seconds']=time.monotonic()-started; results['status']='completed'
    write(args.output/'results.json',results)
    print(json.dumps(results['summary'],indent=2),flush=True)


if __name__=='__main__': main()
