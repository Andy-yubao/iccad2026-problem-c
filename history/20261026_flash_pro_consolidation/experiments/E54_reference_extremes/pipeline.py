"""User-run sequential paired training/evaluation, resumable without old queue."""
import argparse,json,subprocess,sys
from pathlib import Path
from .io import HERE,ROOT,exclusive,sha

ARMS=('no_reference','reference_only')

def run(args):
    chosen=ARMS if args.arm=='all' else (args.arm,)
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    for name,digest in plan['frozen_sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen paired experiment source changed: '+name)
    with exclusive(HERE/'runs/pipeline.lock'):
        if args.resume:
            (HERE/'runs/STOP').unlink(missing_ok=True)
        elif (HERE/'runs/STOP').exists():
            raise RuntimeError('STOP present. Explicit --resume is required.')
        for arm in chosen:
            folder=HERE/'runs'/f'20261026_{arm}_seed0'
            if args.resume:(folder/'STOP').unlink(missing_ok=True)
            if (folder/'STOP').exists():raise RuntimeError('Arm STOP present. Explicit --resume is required.')
            def call(module,extra=()):
                argv=[sys.executable,'-u','-B','-m','experiments.E54_reference_extremes.'+module,'--output',str(folder),*extra]
                print('Running:',subprocess.list2cmdline(argv),flush=True)
                subprocess.run(argv,cwd=ROOT,check=True)
            if args.stage in ('train','all'):
                resume=(folder/'checkpoints/latest.pt').exists()
                if resume and not args.resume:raise RuntimeError('Existing run requires explicit --resume')
                call('train',['--arm',arm,'--seed','0','--device','cuda','--batch-size','32','--engine','cuda_graph',
                              '--max-hours','2','--max-epochs','200','--scheduler-start-epoch','15',*(['--resume'] if resume else [])])
                result=json.loads((folder/'results.json').read_text(encoding='utf-8'))
                if result['status'] not in ('early_stop','max_epochs','budget'):
                    print('Stopped safely; paired queue will not continue:',result['status'],flush=True);return
            if args.stage in ('final','all'):
                if (HERE/'runs/STOP').exists() or (folder/'STOP').exists():return
                call('final')
                if not (folder/'terminal/results.json').exists():return
                call('render_drafts')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--arm',choices=('all',*ARMS),default='all')
    ap.add_argument('--stage',choices=('train','final','all'),default='train')
    ap.add_argument('--resume',action='store_true')
    run(ap.parse_args())

if __name__=='__main__':main()
