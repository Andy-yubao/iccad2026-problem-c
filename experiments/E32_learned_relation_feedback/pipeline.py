"""Sequential registered train and joint terminal comparisons; no adaptive reruns."""
import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def save(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--arms',nargs='+',choices=('replay','draft_only'),default=['replay','draft_only']);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--method',choices=('contact_static','contact_feedback','full_feedback'))
    args=ap.parse_args();folder=HERE/'runs'/f'20261022_pipeline_seed{args.seed}';folder.mkdir(parents=True,exist_ok=True)
    deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();stages=json.loads((folder/'progress.json').read_text())['stages'] if (folder/'progress.json').exists() else []
    for arm in args.arms:
        train=HERE/'runs'/f'20261022_{arm}_seed{args.seed}';terminal=HERE/'runs'/f'20261022_{arm}_seed{args.seed}_terminal500'
        commands=[('train',train,[sys.executable,'-u','-B','-m','experiments.E32_learned_relation_feedback.train','--arm',arm,'--seed',str(args.seed),'--output',str(train),'--max-hours','2','--max-epochs','200']),
            ('terminal',terminal,[sys.executable,'-u','-B','-m','experiments.E32_learned_relation_feedback.evaluate','--checkpoint',str(train/'checkpoints/best.pt'),'--tag',terminal.name])]
        if args.method:commands[-1][2].extend(['--method',args.method])
        for kind,output,command in commands:
            if time.time()>=deadline or (folder/'STOP').exists():return
            if (output/'results.json').exists():
                result=json.loads((output/'results.json').read_text())
                if kind=='train' and result['status'] not in ('early_stop','max_epochs','budget'):raise ValueError('Unfinished training requires explicit resume')
                if kind=='terminal' and result['status']!='complete':raise ValueError('Incomplete terminal result')
                continue
            if kind=='train' and (output/'checkpoints/latest.pt').exists():command.append('--resume')
            started=time.time();process=subprocess.Popen(command,cwd=ROOT)
            save(folder/'active.json',{'kind':kind,'arm':arm,'pid':process.pid,'command':command,'started':started})
            while process.poll() is None:
                if time.time()>=deadline or (folder/'STOP').exists():output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            stages.append({'arm':arm,'kind':kind,'returncode':process.returncode,'seconds':time.time()-started,'command':command})
            save(folder/'progress.json',{'stages':stages,'test_read':False})
            if process.returncode:raise RuntimeError('Stage failed: '+kind+' '+arm)
            if (folder/'STOP').exists() or (output/'STOP').exists():return
    save(folder/'results.json',{'status':'complete','stages':stages,'test_read':False})

if __name__=='__main__':main()
