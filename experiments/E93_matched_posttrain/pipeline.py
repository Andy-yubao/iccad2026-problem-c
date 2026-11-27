"""Authorized sequential C then G, separate child processes, train only."""
import argparse,json,os,subprocess,sys
from .io import ROOT,HERE,save,exclusive
from .train import run_directory
from .preflight import verify

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    verify()
    if (HERE/'runs/STOP').exists():raise ValueError('STOP present')
    done={}
    with exclusive(ROOT/'experiments/E89_spatial_neural_refinement/runs/pipeline.lock'),exclusive(HERE/'runs/pipeline.lock'):
        try:
            for arm in ('C','G'):
                output=run_directory(arm);resume=False
                if output.exists():
                    if not args.resume:raise ValueError('Run exists; explicit --resume required')
                    results=output/'results.json'
                    old=json.loads(results.read_text(encoding='utf-8')) if results.exists() else {}
                    if old.get('status') in ('early_stop','max_epochs','budget'):
                        done[arm]=old;continue
                    if old.get('status')=='error':raise ValueError('Diagnose error before registered recovery')
                    resume=True
                if (HERE/'runs/STOP').exists() or (output/'STOP').exists():
                    save(HERE/'state.json',{'status':'user_stop','completed_arms':list(done),'test_read':False});return
                command=[sys.executable,'-u','-B','-m','experiments.E93_matched_posttrain.train','--arm',arm]
                if resume:command.append('--resume')
                with (HERE/f'runs/{arm.lower()}.stdout.log').open('a',encoding='utf-8') as stdout,(HERE/f'runs/{arm.lower()}.stderr.log').open('a',encoding='utf-8') as stderr:
                    child=subprocess.Popen(command,cwd=ROOT,stdout=stdout,stderr=stderr,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    save(HERE/'state.json',{'status':'running','controller_pid':os.getpid(),'child_pid':child.pid,
                        'active_arm':arm,'queued_arms':['G'] if arm=='C' else [],'completed_arms':list(done),
                        'command':command,'stage':'matched post-training only','test_read':False})
                    print('Started',arm,'pid',child.pid,flush=True);exit_code=child.wait()
                if exit_code:raise RuntimeError(f'{arm} failed with code {exit_code}; queue stopped')
                value=json.loads((output/'results.json').read_text(encoding='utf-8'));done[arm]=value
                print('Ended',arm,value['status'],'new updates',value['new_updates'],flush=True)
                if value['status']=='user_stop' or (HERE/'runs/STOP').exists():
                    save(HERE/'state.json',{'status':'user_stop','completed_arms':list(done),'test_read':False});return
            save(HERE/'state.json',{'status':'completed','completed_arms':list(done),
                 'results':{k:{name:v[name] for name in ('status','phase_epoch','new_updates','best','best_epoch','elapsed_seconds')} for k,v in done.items()},
                 'stage':'training ended; no automatic Q or further experiment','test_read':False})
        except BaseException as error:
            save(HERE/'state.json',{'status':'failed','error':repr(error),'completed_arms':list(done),'test_read':False});raise

if __name__=='__main__':main()
