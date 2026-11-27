"""Single authorized high-LR run, then its unique own-best automatic500 evaluation."""
import argparse,json,os,subprocess,sys
from .io import ROOT,HERE,save,exclusive
from .train import run_directory
from .preflight import verify

def invoke(module,arguments,log):
    command=[sys.executable,'-u','-B','-m',module,*arguments]
    with (HERE/f'runs/{log}.stdout.log').open('a',encoding='utf-8') as out,(HERE/f'runs/{log}.stderr.log').open('a',encoding='utf-8') as err:
        child=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        state=json.loads((HERE/'state.json').read_text(encoding='utf-8'));state.update(controller_pid=os.getpid(),child_pid=child.pid,command=command);save(HERE/'state.json',state)
        print('Started',log,'pid',child.pid,flush=True);code=child.wait()
    if code:raise RuntimeError(f'{log} failed code{code}; no automatic retry')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--resume',action='store_true');args=parser.parse_args();verify()
    if (HERE/'runs/STOP').exists():raise ValueError('STOP present')
    with exclusive(HERE/'runs/pipeline.lock'):
        try:
            output=run_directory('F');resume=False;normal=('early_stop','max_epochs','budget');value={}
            if output.exists():
                if not args.resume:raise ValueError('Run exists; explicit --resume required')
                path=output/'results.json';value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
                if value.get('status')=='error':raise ValueError('Diagnose/register error before recovery')
                resume=value.get('status') not in normal
            if not value or resume:
                save(HERE/'state.json',{'status':'training','test_read':False})
                invoke('experiments.E94_posttrain_lr_reset.train',['--arm','F']+(['--resume'] if resume else []),'train')
                value=json.loads((output/'results.json').read_text(encoding='utf-8'))
            if value.get('status') not in normal or (HERE/'runs/STOP').exists():
                save(HERE/'state.json',{'status':'stopped','training':value,'test_read':False});return
            save(HERE/'state.json',{'status':'training_completed','training':value,'test_read':False})
            # Model/source/own-best registration and export happen only after normal end.
            from .qualification.common import freeze
            freeze()
            save(HERE/'state.json',{'status':'validating','training':value,'test_read':False})
            invoke('experiments.E94_posttrain_lr_reset.qualification.run',[],'validation')
            result=json.loads((output/'qualification/status.json').read_text(encoding='utf-8'))
            if result['status']!='completed':
                save(HERE/'state.json',{'status':'evaluation_stopped','training':value,'evaluation':result,'test_read':False});return
            save(HERE/'state.json',{'status':'completed','training':value,'evaluation':result,'next_action':'report only; no new experiment','test_read':False})
        except BaseException as error:
            save(HERE/'state.json',{'status':'failed','error':repr(error),'test_read':False});raise
if __name__=='__main__':main()
