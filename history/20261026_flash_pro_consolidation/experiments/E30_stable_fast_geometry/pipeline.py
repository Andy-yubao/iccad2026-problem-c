"""Sequential two-arm training and one frozen-best evaluation per arm."""
import argparse
import datetime as dt
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temp.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--policy',required=True)
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--arms',nargs='+',default=['geometry_control','relation_risk'])
    ap.add_argument('--max-hours',type=float,default=2.);ap.add_argument('--jit',action='store_true')
    ap.add_argument('--deadline',default='2026-10-22T08:01:00+08:00');args=ap.parse_args()
    folder=HERE/'runs'/f'20261022_training_seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    deadline=dt.datetime.fromisoformat(args.deadline).timestamp()
    stages=json.loads((folder/'progress.json').read_text())['stages'] if (folder/'progress.json').exists() else []
    def stage(command,output,kind):
        if time.time()>=deadline or (folder/'STOP').exists():return False
        started=time.time();process=subprocess.Popen(command,cwd=ROOT)
        write(folder/'active.json',{'pid':process.pid,'kind':kind,'output':str(output),'command':command,'started':started})
        while process.poll() is None:
            if time.time()>=deadline or (folder/'STOP').exists():
                output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        stages.append({'kind':kind,'output':str(output.relative_to(ROOT)),'command':command,
                       'returncode':process.returncode,'wall_seconds':time.time()-started})
        write(folder/'progress.json',{'stages':stages,'test_read':False})
        if process.returncode:raise RuntimeError('Stage failed: '+kind)
        return not (folder/'STOP').exists() and time.time()<deadline
    for arm in args.arms:
        output=folder/arm
        command=[sys.executable,'-u','-B','-m','experiments.E30_stable_fast_geometry.train','--arm',arm,
            '--output',str(output),'--seed',str(args.seed),'--max-hours',str(args.max_hours),'--max-epochs','200','--batch-size','32']
        if (output/'checkpoints/latest.pt').exists():command.append('--resume')
        if not stage(command,output,'train_'+arm):return
        result=json.loads((output/'results.json').read_text())
        if result['status'] not in ('early_stop','budget','max_epochs'):return
        frozen={'checkpoint':str((output/'checkpoints/best.pt').relative_to(ROOT)),
                'checkpoint_sha256':hashlib.sha256((output/'checkpoints/best.pt').read_bytes()).hexdigest(),
                'best_epoch':result['best_epoch'],'selection':'fixed module-weighted raw repair proxy; no solver',
                'postprocessor':args.policy,'count':500,'panel':'monitor','jit':args.jit,'test_read':False}
        write(output/'frozen.json',frozen)
        tag=f'20261022_{arm}_seed{args.seed}_terminal'
        evaluation=HERE/'runs'/f'{tag}_{args.policy}_monitor500'
        command=[sys.executable,'-u','-B','-m','experiments.E30_stable_fast_geometry.run','--tag',tag,
            '--configs',args.policy,'--count','500','--checkpoint',str(output/'checkpoints/best.pt')]
        if args.jit:command.append('--jit')
        if not stage(command,evaluation,'terminal_'+arm):return
    write(folder/'results.json',{'status':'complete','stages':stages,'test_read':False,'seed':args.seed,
        'policy':args.policy,'jit':args.jit,'single_frozen_best_per_arm':True})

if __name__=='__main__':main()
