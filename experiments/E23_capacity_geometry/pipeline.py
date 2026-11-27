"""Prepare, train one geometry model, then assess its frozen best once on dev500."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from .io import ROOT, HERE, save, exclusive

RUN = HERE/'runs/20261020_capacity160_v5_seed0'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--resume',action='store_true')
    args=ap.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='4'
    if os.environ.get('E16_REGISTRY'): raise ValueError('Remove pilot registry override')
    logs=HERE/'runs/20261020_background'; logs.mkdir(parents=True,exist_ok=True)
    def stage(name,command):
        save(logs/'pipeline.json',{'stage':name,'status':'running','command':command,'test_read':False})
        print(f'===== {name} =====',flush=True)
        try:
            subprocess.run(command,cwd=ROOT,check=True)
        except subprocess.CalledProcessError as error:
            save(logs/'pipeline.json',{'stage':name,'status':'failed','returncode':error.returncode,
                'command':command,'test_read':False})
            raise
    with exclusive(logs/'pipeline.lock'):
        if (RUN/'STOP').exists(): print('STOP exists; no stage started',flush=True); return
        stage('prepare',[sys.executable,'-u','-B','-m','experiments.E23_capacity_geometry.cache'])
        from .cache import CACHE
        if (RUN/'STOP').exists() or (CACHE/'STOP').exists() or not all((CACHE/r/'manifest.json').exists() for r in ('train','dev')):
            save(logs/'pipeline.json',{'stage':'prepare','status':'paused','test_read':False}); return
        command=[sys.executable,'-u','-B','-m','experiments.E23_capacity_geometry.train',
            '--output',str(RUN),'--device','cuda','--max-hours','4','--max-epochs','200','--batch-size','32']
        if args.resume and (RUN/'checkpoints/latest.pt').exists(): command+=['--resume']
        stage('train',command)
        progress=json.loads((RUN/'progress.json').read_text(encoding='utf-8'))
        if not progress['finished'] or (RUN/'STOP').exists():
            save(logs/'pipeline.json',{'stage':'train','status':'paused','test_read':False}); return
        command=[sys.executable,'-u','-B','-m','experiments.E23_capacity_geometry.final_evaluate',
            '--run',str(RUN),'--device','cpu','--workers','2','--resume']
        stage('final_dev500',command)
        complete=(RUN/'final_evaluation/results.json').exists()
        save(logs/'pipeline.json',{'stage':'final_dev500','status':'complete' if complete else 'paused','test_read':False})


if __name__=='__main__': main()
