"""User-launched single candidate; no scheduler/automation or extra training."""
import argparse,json,subprocess,sys
from pathlib import Path
from .runtime import OUTPUT
from .io import HERE,ROOT,save,exclusive
from .train import verify_preflight,stopped

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--migrate-from',type=Path);p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();module='experiments.E95_single_pass_corrector'
    command=[sys.executable,'-u','-B','-m',module+'.train']
    if a.resume:command.append('--resume')
    elif a.migrate_from:command.extend(['--migrate-from',str(a.migrate_from.resolve())])
    evaluation=[sys.executable,'-u','-B','-m',module+'.evaluate']
    if a.dry_run:print(json.dumps(dict(train=command,evaluate=evaluation),indent=2));return
    verify_preflight()
    with exclusive(HERE/'runs/pipeline.lock'):
        if stopped(OUTPUT):print('STOP present; not starting');return
        result=OUTPUT/'results.json'
        if not result.exists() or not json.loads(result.read_text(encoding='utf-8')).get('finished'):
            save(HERE/'state.json',dict(status='training',test_read=False))
            subprocess.run(command,cwd=ROOT,check=True)
        if stopped(OUTPUT) or not result.exists():return
        if not json.loads(result.read_text(encoding='utf-8')).get('finished'):return
        if not (OUTPUT/'checkpoints/best.pt').exists():raise RuntimeError('No monitored best available')
        save(HERE/'state.json',dict(status='evaluation',test_read=False))
        subprocess.run(evaluation,cwd=ROOT,check=True)
        complete=(OUTPUT/'evaluation/gallery/index.html').exists()
        save(HERE/'state.json',dict(status='complete' if complete else 'evaluation_incomplete',
            primary_calls=1,no_automatic_next_candidate=True,test_read=False))

if __name__=='__main__':main()
