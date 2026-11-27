"""User-run training only. No automatic Q, gallery or later candidate."""
import argparse,json,subprocess,sys,os
from datetime import datetime,timezone
from .io import HERE,ROOT,save,exclusive

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();output=HERE/'runs/20261031_scratch32k_seed0'
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    command=[sys.executable,'-u','-B','-m','experiments.E91_e74_32k_scratch.train',
        '--arm','simple','--output',str(output),'--max-hours',str(plan['maximum_hours']),'--max-epochs','200','--scheduler-start-epoch','15']
    if args.resume:command.append('--resume')
    if args.dry_run:print(json.dumps(command,indent=2));return
    from .preflight import verify
    from .loss_config import load_coefficients
    verify();load_coefficients(HERE/'runs/registered_loss.json')
    if (HERE/'runs/STOP').exists() or (output/'STOP').exists():raise ValueError('STOP present; remove only for authorized resume')
    if args.resume and not (output/'checkpoints/latest.pt').exists():raise ValueError('No checkpoint to resume')
    # All old charges remain; no concurrent E89 GPU work or controller reuse.
    from .budget import global_charged_seconds
    if global_charged_seconds()>=64800:raise ValueError('Existing cumulative18h budget exhausted')
    with exclusive(ROOT/'experiments/E89_spatial_neural_refinement/runs/pipeline.lock'), exclusive(HERE/'runs/pipeline.lock'):
        save(HERE/'state.json',dict(status='running',stage='train_only',pid=os.getpid(),command=command,
            utc=datetime.now(timezone.utc).isoformat(),test_read=False))
        completed=subprocess.run(command,cwd=ROOT)
        if completed.returncode:
            save(HERE/'state.json',dict(status='failed',stage='train_only',exit_code=completed.returncode,test_read=False))
            raise SystemExit(completed.returncode)
        result=json.loads((output/'results.json').read_text(encoding='utf-8'))
        save(HERE/'state.json',dict(status=result['status'],stage='training ended; evaluation requires explicit request',
            best_epoch=result['best_epoch'],test_read=False))

if __name__=='__main__':main()
