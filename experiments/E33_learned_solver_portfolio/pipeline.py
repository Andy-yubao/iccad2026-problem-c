"""Sequential checks, train-label headroom, gate fit, calibration and one endpoint."""
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
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--defer-terminal',action='store_true');args=ap.parse_args()
    folder=HERE/'runs'/f'20261022_pipeline_seed{args.seed}';folder.mkdir(parents=True,exist_ok=True);stages=json.loads((folder/'progress.json').read_text())['stages'] if (folder/'progress.json').exists() else []
    deadline=dt.datetime.fromisoformat('2026-10-22T19:39:00+08:00').timestamp();labels=HERE/'runs/prepared_expert_outcomes_v1';train=HERE/'runs'/f'20261022_gate_seed{args.seed}'
    def stage(name,module,arguments,output,complete):
        if complete():return True
        if (folder/'STOP').exists() or time.time()>=deadline:return False
        command=[sys.executable,'-u','-B','-m','experiments.E33_learned_solver_portfolio.'+module,*arguments];started=time.time();process=subprocess.Popen(command,cwd=ROOT)
        save(folder/'active.json',{'kind':name,'pid':process.pid,'command':command,'started':started})
        while process.poll() is None:
            if (folder/'STOP').exists() or time.time()>=deadline:output.mkdir(parents=True,exist_ok=True);(output/'STOP').touch()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:pass
        stages.append({'kind':name,'seconds':time.time()-started,'returncode':process.returncode,'command':command});save(folder/'progress.json',{'stages':stages,'test_read':False})
        if process.returncode:raise RuntimeError('Stage failed: '+name)
        return not (folder/'STOP').exists() and not (output/'STOP').exists() and time.time()<deadline
    read=lambda p:json.loads(p.read_text())
    if not stage('check','check',[],HERE/'runs/20261022_interface_check5',lambda:(HERE/'runs/20261022_interface_check5/results.json').exists()):return
    if not stage('pilot_labels','prepare',['--limit','1000','--resume'],labels,lambda:(labels/'results_1000.json').exists() and read(labels/'results_1000.json')['status'] in ('pilot_complete','complete')):return
    if read(labels/'results_1000.json')['status'] not in ('pilot_complete','complete'):return
    if not stage('headroom','inspect_labels',[],labels,lambda:(labels/'headroom_1000.json').exists()):return
    if not read(labels/'headroom_1000.json')['continue_to_full_labels']:
        save(folder/'results.json',{'status':'stopped_insufficient_train_oracle_headroom','stages':stages,'test_read':False});return
    if not stage('full_labels','prepare',['--limit','16000','--resume'],labels,lambda:(labels/'results_16000.json').exists() and read(labels/'results_16000.json')['status']=='complete'):return
    if read(labels/'results_16000.json')['status']!='complete':return
    command=['--seed',str(args.seed),'--output',str(train),'--max-hours','2','--max-epochs','200']
    if (train/'checkpoints/latest.pt').exists():command.append('--resume')
    if not stage('train','train',command,train,lambda:(train/'results.json').exists() and read(train/'results.json')['status'] in ('early_stop','max_epochs','budget')):return
    if not stage('calibrate','calibrate',['--checkpoint',str(train/'checkpoints/best.pt')],train,lambda:(train/'calibration.json').exists()):return
    if read(train/'calibration.json')['status']!='frozen_candidate':
        save(folder/'results.json',{'status':'stopped_no_safe_gate','stages':stages,'test_read':False});return
    if args.defer_terminal:
        save(folder/'results.json',{'status':'calibrated_terminal_deferred','stages':stages,'test_read':False});return
    endpoint=HERE/'runs'/f'20261022_gate_seed{args.seed}_terminal500'
    if not stage('terminal','evaluate',['--checkpoint',str(train/'checkpoints/best.pt'),'--tag',endpoint.name],endpoint,lambda:(endpoint/'results.json').exists()):return
    save(folder/'results.json',{'status':'complete','stages':stages,'test_read':False})

if __name__=='__main__':main()
