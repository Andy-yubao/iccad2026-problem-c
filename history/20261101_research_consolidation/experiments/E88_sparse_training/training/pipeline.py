"""One authorized sparse continuation, then dev diagnostics and unique-best Q."""
import argparse,json,subprocess,sys,os
from datetime import datetime,timezone
from .io import HERE,ROOT,save,exclusive

OUTPUT=HERE/'runs/20261031_e90_sparse_new_seed0'

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    module='experiments.E88_sparse_training.training'
    train=[sys.executable,'-u','-B','-m',module+'.train','--arm','simple','--output',str(OUTPUT),'--max-hours','1','--max-epochs','200','--scheduler-start-epoch','0']
    if a.resume:train.append('--resume')
    commands=[('train',train),('validation',[sys.executable,'-u','-B','-m',module+'.final','--output',str(OUTPUT),'--stage','all']),
        ('report',[sys.executable,'-u','-B','-m',module+'.report'])]
    if a.dry_run:print(json.dumps(commands,indent=2));return
    from .preflight import verify
    from .loss_config import load_coefficients
    verify();load_coefficients(HERE/'runs/registered_training_loss.json')
    with exclusive(HERE/'runs/training_pipeline.lock'):
        save(HERE/'runs/training_launch.json',dict(command=[sys.executable,*sys.argv],pid=os.getpid(),utc=datetime.now(timezone.utc).isoformat(),
            authorization='用户明确要求后台续训，训练结束自动测试；测试解释为原开发验证，封存测试不读取',test_read=False))
        for stage,command in commands:
            if (HERE/'runs/STOP').exists() or (OUTPUT/'STOP').exists():return
            result_path=OUTPUT/'results.json'
            if stage=='train' and result_path.exists() and json.loads(result_path.read_text(encoding='utf-8'))['status'] in ('budget','early_stop','max_epochs'):continue
            with (HERE/f'runs/sparse_{stage}.log').open('a',encoding='utf-8') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                save(HERE/'training_state.json',dict(status='running',stage=stage,pid=process.pid,test_read=False))
                print(stage,'started; log:',log.name,flush=True);code=process.wait()
            if code:
                save(HERE/'training_state.json',dict(status='failed',stage=stage,exit_code=code,test_read=False))
                raise RuntimeError(f'{stage} failed; see {log.name}')
            if stage=='train' and json.loads(result_path.read_text(encoding='utf-8'))['status']=='user_stop':return
            if stage=='validation' and (not (OUTPUT/'full_dev.json').exists() or not (OUTPUT/'terminal/results.json').exists()):return
        save(HERE/'training_state.json',dict(status='complete',stage='await review; no automatic later training',test_read=False))
        print('Sparse training and authorized development evaluation complete',flush=True)

if __name__=='__main__':main()
