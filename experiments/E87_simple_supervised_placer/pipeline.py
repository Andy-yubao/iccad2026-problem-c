"""One explicitly selected arm; no auto-start of later candidates or stages."""
import argparse,json,subprocess,sys
from pathlib import Path
from .io import HERE,ROOT,save,exclusive

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=('simple','legacy'),default='simple')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    folder=HERE/f'runs/20261030_{a.arm}_seed0';module='experiments.E87_simple_supervised_placer'
    commands=[('calibration',[sys.executable,'-u','-B','-m',module+'.calibrate']),
        ('train',[sys.executable,'-u','-B','-m',module+'.train','--arm',a.arm,'--output',str(folder),'--max-hours',str(2 if a.arm=='simple' else 1),'--scheduler-start-epoch','0']),
        ('final',[sys.executable,'-u','-B','-m',module+'.final','--output',str(folder),'--stage','all']),
        ('draft100',[sys.executable,'-u','-B','-m',module+'.render_drafts','--output',str(folder)])]
    if a.resume and (folder/'checkpoints/latest.pt').exists():commands[1][1].append('--resume')
    if a.dry_run:print(json.dumps(commands,indent=2));return
    from .preflight import verify
    verify()
    with exclusive(HERE/'runs/pipeline.lock'):
        for stage,cmd in commands:
            if (HERE/'runs/STOP').exists() or (folder/'STOP').exists():return
            result_path=folder/'results.json'
            if stage=='train' and result_path.exists() and json.loads(result_path.read_text(encoding='utf-8'))['status'] in ('early_stop','budget','max_epochs'):continue
            with (HERE/f'runs/{a.arm}_{stage}.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(HERE/'state.json',dict(status='running',arm=a.arm,stage=stage,pid=child.pid,test_read=False))
                print(stage,'started, log:',log.name,flush=True);code=child.wait()
            if code:raise RuntimeError(f'{stage} failed ({code}); see {log.name}')
            if stage=='train' and json.loads(result_path.read_text(encoding='utf-8'))['status']=='user_stop':return
            if stage=='final' and not (folder/'terminal/results.json').exists():return
        save(HERE/'state.json',dict(status='complete',arm=a.arm,next_stage='await review; no auto-start',test_read=False))

if __name__=='__main__':main()
