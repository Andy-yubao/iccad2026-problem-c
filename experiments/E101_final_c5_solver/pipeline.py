"""One authorized background run; bounded stages, no candidates or wakeups."""
import argparse,os,subprocess,sys,time
import torch
from .common import HERE,RUN,PLAN,read,save,checkpoint,sha,check_stop,snapshot,exclusive,utc

def child(module,*args):
    check_stop();name=module.split('.')[-1]+('_'+args[0] if args and not args[0].startswith('--') else '')
    command=[sys.executable,'-u','-B','-m','experiments.E101_final_c5_solver.'+module,*args]
    print('Stage',name,'started',utc(),flush=True);save(RUN/'state.json',dict(status='running',stage=name,pid=os.getpid(),updated_utc=utc(),test_read=module=='evaluate' and args[0]=='local_test'))
    with (RUN/(name+'.log')).open('a',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=HERE.parents[1],stdout=log,stderr=subprocess.STDOUT)
    if result.returncode==2:
        from .common import stopped
        if stopped():raise InterruptedError(f'{name} safely saved at STOP boundary')
    if result.returncode:raise RuntimeError(f'{name} failed with exit {result.returncode}; see {name}.log; no automatic retry')
    check_stop()

def freeze():
    from .train import baseline
    model,cfg=baseline();l=torch.load(RUN/'L/checkpoints/best.pt',map_location='cpu',weights_only=False);f=torch.load(RUN/'F/checkpoints/best.pt',map_location='cpu',weights_only=False)
    model.placer.load_state_dict(l['model']);model.refiner.load_state_dict(f['model']);path=HERE/'delivery/model.pt'
    checkpoint(path,dict(model=model.state_dict(),model_config=cfg,identity=dict(plan_sha256=sha(HERE/'plan.json'),L_best_sha256=sha(RUN/'L/checkpoints/best.pt'),F_best_sha256=sha(RUN/'F/checkpoints/best.pt')),selection='unique stage own-Loss best; no Q or test reselection'))
    save(RUN/'frozen_selection.json',dict(checkpoint_sha256=sha(path),L_best_epoch=l['progress']['best_epoch'],F_best_epoch=f['progress']['best_epoch'],frozen_utc=utc(),test_payloads_read=False,test_authorization=PLAN['authorization']))

def main(resume=False):
    RUN.mkdir(parents=True,exist_ok=True)
    with exclusive(RUN/'pipeline.lock'):
        check_stop()
        if not (RUN/'sources.json').exists():snapshot()
        else:
            from .common import sources
            expected=read(RUN/'sources.json');current=sources()
            assert all(current.get(name)==digest for name,digest in expected.items()),'Frozen dependency changed: explicit new identity required'
        assert read(RUN/'cache/train/complete.json')['count']==64000 and read(RUN/'cache/dev/complete.json')['count']==2000
        assert read(RUN/'preflight/results.json')['status']=='passed'
        for module,args,result in [('train',('L',),RUN/'L/results.json'),('teacher',(),RUN/'teacher/results.json'),('train',('F',),RUN/'F/results.json')]:
            if result.exists() and read(result)['status'] in ('complete','early_stop','budget'):continue
            extra=('--resume',) if resume and module=='train' and (RUN/args[0]/'checkpoints/latest.pt').exists() else ()
            child(module,*args,*extra)
        check_stop()
        if not (RUN/'frozen_selection.json').exists():freeze()
        for role in ('dev_monitor500','local_test'):
            if not (RUN/'evaluation'/role/'results.json').exists():child('evaluate',role)
        save(RUN/'state.json',dict(status='complete',stage='final_test2000_complete',updated_utc=utc(),test_read=True,no_further_training=True))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    try:main(a.resume)
    except BaseException as e:
        save(RUN/'state.json',dict(status='user_stop' if isinstance(e,InterruptedError) else 'error',error=repr(e),updated_utc=utc(),no_automatic_retry=True));raise
