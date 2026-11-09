"""User-run training only. No automatic Q, gallery or later candidate."""
import argparse,json,subprocess,sys
from .io import HERE,ROOT

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args();output=HERE/'runs/20261031_aux3_seed0'
    command=[sys.executable,'-u','-B','-m','experiments.E90_auxiliary_weight_placer.train',
        '--arm','simple','--output',str(output),'--max-hours','1','--max-epochs','14','--scheduler-start-epoch','0']
    if args.resume:command.append('--resume')
    if args.dry_run:print(json.dumps(command,indent=2));return
    from .preflight import verify
    from .loss_config import load_coefficients
    verify();load_coefficients(HERE/'runs/registered_loss.json')
    if (HERE/'runs/STOP').exists() or (output/'STOP').exists():raise ValueError('STOP present; remove only for authorized resume')
    if args.resume and not (output/'checkpoints/latest.pt').exists():raise ValueError('No checkpoint to resume')
    subprocess.run(command,cwd=ROOT,check=True)

if __name__=='__main__':main()
