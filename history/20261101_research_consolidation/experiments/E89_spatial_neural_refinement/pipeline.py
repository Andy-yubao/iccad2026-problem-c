"""Registered spatial/pooled pair; each own-best evaluated exactly once."""
import argparse,json,subprocess,sys,os
from datetime import datetime,timezone
from .io import HERE,ROOT,save,exclusive


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('spatial','pooled','both'),default='both')
    p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();module='experiments.E89_spatial_neural_refinement'
    arms=('spatial','pooled') if a.arm=='both' else (a.arm,)
    commands=[]
    for arm in arms:
        output=HERE/f'runs/20261031_{arm}_seed0'
        command=[sys.executable,'-u','-B','-m',module+'.train','--arm',arm,'--output',str(output),
                 '--max-hours','1','--max-epochs','200','--scheduler-start-epoch','0']
        if a.resume and (output/'checkpoints/latest.pt').exists():command.append('--resume')
        commands.extend([(arm,'train',output,command),
            (arm,'validation',output,[sys.executable,'-u','-B','-m',module+'.final','--output',str(output),'--stage','all']),
            (arm,'report',output,[sys.executable,'-u','-B','-m',module+'.report','--arm',arm]),
            (arm,'gallery',output,[sys.executable,'-u','-B','-m',module+'.render_drafts','--output',str(output)])])
    if a.dry_run:print(json.dumps([(arm,stage,cmd) for arm,stage,_,cmd in commands],indent=2));return
    from .preflight import verify
    from .loss_config import load_coefficients
    verify();load_coefficients(HERE/'runs/registered_training_loss.json')
    with exclusive(HERE/'runs/pipeline.lock'):
        save(HERE/'runs/launch.json',dict(command=[sys.executable,*sys.argv],pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),
            authorization='用户要求现在开始E89；已登记空间/汇总广播两分支，各1h，沿用新Loss及自动开发评价',test_read=False))
        for arm,stage,output,command in commands:
            if (HERE/'runs/STOP').exists() or (output/'STOP').exists():return
            result_path=output/'results.json'
            if stage=='train' and result_path.exists() and json.loads(result_path.read_text(encoding='utf-8'))['status'] in ('budget','early_stop','max_epochs'):continue
            with (HERE/f'runs/{arm}_{stage}.log').open('a',encoding='utf-8') as log:
                child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(HERE/'state.json',dict(status='running',arm=arm,stage=stage,pid=child.pid,test_read=False))
                print(arm,stage,'started; log:',log.name,flush=True);code=child.wait()
            if code:
                save(HERE/'state.json',dict(status='failed',arm=arm,stage=stage,exit_code=code,test_read=False))
                raise RuntimeError(f'{arm}/{stage} failed; see {log.name}')
            if stage=='train' and json.loads(result_path.read_text(encoding='utf-8'))['status']=='user_stop':return
            if stage=='validation' and (not (output/'full_dev.json').exists() or not (output/'terminal/results.json').exists()):return
        save(HERE/'state.json',dict(status='complete',stage='await review; no further candidate',test_read=False))
        print('Registered E89 arms completed; tests sealed',flush=True)


if __name__=='__main__':main()
