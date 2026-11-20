"""Frozen E153 backend: exact registered dev500, full MIB scoring, no training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import json,time
import numpy as np
import torch
from ..io import HERE,ROOT,save,sha,sources,exclusive
from ..data import load_rows
from ..public import inputs_only
from ..model import GeometryModel
from ..batches import decode
from .collect import public_batch
from .solver import optimize
from scoring.evaluation import evaluate_case
from scoring.official import official_evaluator
from datasets.cases import ResearchCase
from datasets.reference_layouts import public_hard_reference
from research.quality_log import summary as quality_line
def main():
    auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
    if not auth['training_enabled']:raise PermissionError('Formal pipeline is not activated')
    gate=json.loads((HERE/'preflight.json').read_text(encoding='utf-8'));assert gate['status']=='passed' and gate['sources']==sources()
    frozen=json.loads((HERE/'frozen_backend.json').read_text(encoding='utf-8'));path=ROOT/frozen['checkpoint'];assert sha(path)==frozen['checkpoint_sha256']
    out=HERE/'runs/e153_backend_seed0/evaluation_mib_dev500'
    if out.exists():raise FileExistsError('Preserve existing evaluation; no implicit rerun')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(path,map_location='cpu',weights_only=False);model=GeometryModel(**state['model_config']).cuda().eval();model.load_state_dict(state['model'])
    rows=load_rows('dev_monitor500');results=[]
    save(out/'frozen.json',dict(**frozen,sources=sources(),members=[list(r['member']) for r in rows],scoring='Full grouping/boundary/MIB; neutral runtime research Q on mean-area task'))
    with torch.inference_mode():
        from .repair.canonical_native import warmup
        warmup();model(public_batch(rows[0]['inputs']))
        for i,row in enumerate(rows):
            if any((HERE/n).exists() for n in ('STOP','QUEUE_STOP','PAUSE_REQUEST')):
                save(out/'state.json',dict(status='user_stop',completed=i));return
            inputs=row['inputs'];n=int(inputs[0]);hard_reference=public_hard_reference(inputs[1],inputs[5],inputs[6])
            case=ResearchCase(f'E153.dev{i}',dict(member=list(row['member'])),inputs,hard_reference,row['baseline_metrics'])
            torch.cuda.synchronize();start=time.perf_counter();g=public_batch(inputs);states=model(g)
            draft=(decode(states[-1],g)[0,:n]*g['scale'][0]).cpu().numpy().astype(float);draft_seconds=time.perf_counter()-start
            final,info=optimize(draft,inputs);seconds=time.perf_counter()-start
            final_score=evaluate_case(case,final,seconds);raw=evaluate_case(case,draft,draft_seconds)
            assert final_score['metrics']['is_feasible'] and final_score['metrics']['mib_violations']==0
            record=dict(index=i,member=list(row['member']),n=n,checkpoint_sha256=frozen['checkpoint_sha256'],metrics=final_score['metrics'],raw_metrics=raw['metrics'],
                positions=final.tolist(),draft_positions=draft.tolist(),complete_seconds=seconds,trace=info['trace'],work=info['work'],test_read=False)
            save(out/f'cases/{i:04d}.json',record);results.append(record);save(out/'state.json',dict(status='running',completed=i+1))
            if (i+1)%10==0:print(f'E153 full-MIB dev500 {i+1}/500',flush=True)
    costs=[r['metrics']['cost'] for r in results];times=[r['complete_seconds'] for r in results];weighted=official_evaluator().compute_total_score(costs,[r['n'] for r in results]);grouping=sum(r['metrics']['grouping_violations']==0 for r in results)
    result=dict(status='complete',arm='e153_backend',count=500,best_epoch=frozen['best_epoch'],checkpoint_sha256=frozen['checkpoint_sha256'],
        hard_legal=500,grouping_satisfied=grouping,grouping_target_met=grouping>=475,mib_satisfied=500,
        weighted_Q=weighted,mean_Q=float(np.mean(costs)),mixed_Q=.5*(weighted+float(np.mean(costs))),
        mean_complete_seconds=float(np.mean(times)),p95_complete_seconds=float(np.quantile(times,.95)),max_complete_seconds=max(times),over250ms=sum(t>.25 for t in times),
        scoring_policy='MIB-compatible mean-area derived task, full soft constraints, neutral runtime research Q; not original-area official contest score',test_read=False)
    save(out/'results.json',result);save(out/'state.json',dict(status='complete',completed=500));print(quality_line(result,out/'results.json'),flush=True)
if __name__=='__main__':main()
