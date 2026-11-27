"""Audit already saved evidence; no inference, optimizer, solver or scorer calls."""
import datetime,hashlib,json,math,msvcrt
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    study=read(HERE/'study.json');summary=[];bindings=[]
    for item in study['candidate_ledger']:
        run=HERE/item['output'];training=read(run/'results.json');evaluation=read(run/'evaluation/results.json')
        assert training['finished'] and evaluation['finished']
        assert evaluation['terminal']['count']==500 and evaluation['terminal']['legal']==500
        summary.append(dict(candidate=item['candidate'],training=training,terminal=evaluation['terminal'],raw_count=evaluation['raw_count'],gallery_count=evaluation['gallery_count']))
        good=[p for p in sorted((HERE/'runs').glob(f"20261102_candidate{item['candidate']}_preflight*/results.json")) if read(p).get('status')=='passed']
        assert good
        check=read(good[-1]);bad=[k for k,v in check['source_sha256'].items() if not (ROOT/k).is_file() or sha(ROOT/k)!=v]
        assert not bad,bad
        bindings.append(dict(candidate=item['candidate'],preflight=good[-1].relative_to(HERE).as_posix(),files=len(check['source_sha256']),mismatches=bad))
    folder=HERE/'runs/20261102_candidate8_public_task_seed0';evaluation=folder/'evaluation'
    previous=HERE/'runs/20261102_candidate7_normalized_geometry_seed0/evaluation'
    raw=[r for f in sorted((evaluation/'raw').glob('*.json')) for r in read(f)]
    terminal=[read(f) for f in sorted((evaluation/'terminal').glob('*.json'))]
    prior=[read(f) for f in sorted((previous/'terminal').glob('*.json'))]
    assert len(raw)==2000 and len(terminal)==len(prior)==500
    assert [r['index'] for r in terminal]==list(range(500)) and [r['index'] for r in raw]==list(range(2000))
    assert [r['member'] for r in terminal]==[r['member'] for r in prior]
    assert all(r['metrics']['is_feasible'] and r['independent_hard_checks']['is_feasible'] for r in terminal)
    manifest=read(evaluation/'gallery/manifest.json');assert len(manifest['cases'])==100
    best_sha=sha(folder/'checkpoints/best.pt')
    for r in manifest['cases']:assert sha(evaluation/'gallery'/r['image'])==r['sha256'] and r['checkpoint_sha256']==best_sha
    frozen=read(evaluation/'frozen.json');assert frozen['checkpoint_sha256']==best_sha
    for k,v in frozen['evaluation_sources'].items():assert sha(ROOT/k)==v,k
    lock_results=[]
    for phase in ['training','pipeline','evaluation']:
        path=HERE/f'runs/candidate8_{phase}.lock'
        with path.open('r+b') as h:
            h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1)
            h.seek(0);msvcrt.locking(h.fileno(),msvcrt.LK_UNLCK,1)
        lock_results.append(phase)
    maximum=max(r['n'] for r in terminal);weights=[math.exp((r['n']-maximum)/12) for r in terminal];mass=sum(weights)
    score=sum(w*r['metrics']['cost'] for w,r in zip(weights,terminal))/mass
    assert abs(score-read(evaluation/'results.json')['terminal']['weighted_Q'])<1e-12
    excess=score-1;fallback_excess=sum(w*(r['metrics']['cost']-1) for w,r in zip(weights,terminal) if r['fallback'])/mass
    nonfallback=[(w,r) for w,r in zip(weights,terminal) if not r['fallback']]
    progress=read(folder/'progress.json');best=next(r for r in progress['history'] if r['epoch']==progress['best_epoch'])
    baseline=progress['history'][0]
    delta=[r['metrics']['cost']-p['metrics']['cost'] for r,p in zip(terminal,prior)]
    draft_times=sorted(r['draft_seconds'] for r in terminal)
    # Percentile interpolation exactly as NumPy's default linear quantile.
    def p95(v):
        a=sorted(v);x=(len(a)-1)*.95;lo=math.floor(x);return a[lo]+(x-lo)*(a[math.ceil(x)]-a[lo])
    audit=dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Saved metrics/coordinates/hashes only; no inference/update/solver/scorer',candidates=summary,bindings=bindings,
        candidate8=dict(best_epoch=progress['best_epoch'],best_checkpoint_sha256=best_sha,loss_epoch0=baseline['dev_monitor'],loss_best=best['dev_monitor'],loss_last=progress['history'][-1]['dev_monitor'],
            raw_count=2000,module_covered_le10_fraction=1-sum(r['corrected']['diagnostics']['N10'] for r in raw)/sum(r['n'] for r in raw),worst_module_covered_fraction=max(r['corrected']['diagnostics']['max_covered_fraction'] for r in raw),
            raw_summary=read(evaluation/'summary.json')['raw'],fallback_indices=[r['index'] for r in terminal if r['fallback']],fallback_weight_mass=sum(w for w,r in zip(weights,terminal) if r['fallback'])/mass,fallback_excess_Q=fallback_excess,fallback_share_of_excess_Q=fallback_excess/excess,
            nonfallback_conditional_Q_diagnostic_only=sum(w*r['metrics']['cost'] for w,r in nonfallback)/sum(w for w,r in nonfallback),
            paired_C7=dict(improved=sum(d< -1e-6 for d in delta),worse=sum(d>1e-6 for d in delta),near_equal=sum(abs(d)<=1e-6 for d in delta),weighted_Q_change=score-read(previous/'results.json')['terminal']['weighted_Q']),
            cold_initialization_seconds=frozen['cold_initialization_seconds'],draft_seconds_mean=sum(draft_times)/500,draft_seconds_p95=p95(draft_times),repair_seconds_mean=sum(r['repair_seconds'] for r in terminal)/500,repair_seconds_p95=p95([r['repair_seconds'] for r in terminal]),gallery_hashes_verified=100,all_three_locks_released=lock_results),
        test_read=False,no_new_training_or_evaluation=True)
    (HERE/'runs/final_saved_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(audit['candidate8'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
