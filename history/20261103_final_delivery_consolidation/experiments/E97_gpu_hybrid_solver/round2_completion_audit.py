"""Read saved results/checkpoints only after active pipeline completion."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import json,math,datetime
from pathlib import Path
import numpy as np
import torch
from .train_io.io import save,sha,exclusive
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    torch.set_num_threads(1);study=read(HERE/'round2_study.json');ledger=[r for r in study['candidate_ledger'] if r.get('formal_training_started')];assert ledger and all(r['status']=='complete' for r in ledger),'Wait for registered pipelines to complete'
    locks=[HERE/'runs/round2_execution.lock']
    for row in ledger:
        for phase in ('pipeline','training','evaluation'):locks.append(HERE/'runs'/f'round2_candidate{row["candidate"]}_{phase}.lock')
    for path in locks:
        if path.exists():
            with exclusive(path):pass
    records=[]
    for row in ledger:
        run=HERE/row['output'];evaluation_folder=run/row.get('evaluation_dir','evaluation');config=read(run/'config.json');train=read(run/'results.json');evaluation=read(evaluation_folder/'results.json');assert train['finished'] and evaluation['finished']
        drift=[name for name,digest in config['identity']['source_sha256'].items() if sha(ROOT/name)!=digest];assert not drift,drift
        best=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=False,mmap=True);latest=torch.load(run/'checkpoints/latest.pt',map_location='cpu',weights_only=False,mmap=True)
        required={'model','optimizer','scheduler','rng','progress','identity','elapsed_seconds'};assert required<=best.keys() and required<=latest.keys();assert best['identity']==latest['identity']==config['identity']
        for saved in (best,latest):assert {'python','numpy','torch','cuda'}<=saved['rng'].keys() and {'order','cursor','seen','history'}<=saved['progress'].keys()
        history=latest['progress']['history'];chosen=min(history,key=lambda r:r['dev_monitor']['total'])['epoch'];assert best['progress']['best_epoch']==latest['progress']['best_epoch']==chosen
        reused=evaluation.get('reused_initial',False);folder=HERE/evaluation['reused_from'] if reused else evaluation_folder
        terminal=[read(p) for p in sorted((folder/'terminal').glob('*.json'))];assert len(terminal)==500
        assert all(r['metrics']['is_feasible'] and r['independent_hard_checks']['is_feasible'] for r in terminal)
        weights=np.array([math.exp((r['n']-120)/12) for r in terminal]);weights/=weights.sum();Q=float(sum(w*r['metrics']['cost'] for w,r in zip(weights,terminal)));summary=evaluation['terminal'];assert abs(Q-summary['weighted_Q'])<1e-10
        manifest=read(folder/'gallery/manifest.json');assert manifest['count']==100 and all(sha(folder/'gallery'/r['image'])==r['sha256'] for r in manifest['cases'])
        raw=[r for path in sorted((folder/'raw').glob('*.json')) for r in read(path)];assert len(raw)==2000
        for stage in ('placer','corrected'):
            assert all(stage in r for r in raw)
        raw_metrics=dict(module_coverage_le_10_percent=1-sum(r['corrected']['diagnostics']['N10'] for r in raw)/sum(r['n'] for r in raw),global_worst_module=max(r['corrected']['diagnostics']['max_covered_fraction'] for r in raw),mean_case_worst_module=float(np.mean([r['corrected']['diagnostics']['max_covered_fraction'] for r in raw])))
        extra=dict(draft_mean_seconds=float(np.mean([r['draft_seconds'] for r in terminal])),repair_mean_seconds=float(np.mean([r['repair_seconds'] for r in terminal])),shape_calls_mean=float(np.mean([sum(t.get('phase')=='shape' for t in r['trace']) for r in terminal])))
        records.append(dict(candidate=row['candidate'],run=row['output'],training=train,terminal=summary,extra_timing=extra,raw=raw_metrics,reused_initial=reused,evidence_source=folder.relative_to(HERE).as_posix(),checkpoint_SHA256=sha(run/'checkpoints/best.pt'),own_Loss_unique_best_verified=True,complete_resume_state_verified=True,bound_source_drift=[],gallery_SHA_verified=100))
        del best,latest
    result=dict(status='complete',timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_start_utc=study['start_utc'],deadline_utc=study['deadline_utc'],formal_candidates=len(records),all_required_locks_released=True,records=records,no_new_inference_optimizer_solver_or_scorer_calls=True,test_read=False)
    save(HERE/'runs/round2_completion_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
