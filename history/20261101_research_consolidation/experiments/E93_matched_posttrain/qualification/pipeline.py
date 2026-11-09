"""Sequential C/G evaluation only. No training and no old controller activation."""
import os,subprocess,sys,time
from ..io import HERE,save,exclusive

def main():
    folder=HERE/'runs/evaluation';folder.mkdir(parents=True,exist_ok=True)
    with exclusive(HERE/'runs/evaluation_pipeline.lock'):
        save(folder/'state.json',{'status':'running','pid':os.getpid(),'order':['C','G'],'test_read':False})
        for arm in ('C','G'):
            env=dict(os.environ,E93_EVAL_ARM=arm,E93_EVAL_THREADS='4')
            command=[sys.executable,'-u','-B','-m','experiments.E93_matched_posttrain.qualification.run']
            with (folder/f'{arm.lower()}.stdout.log').open('a',encoding='utf-8') as out,(folder/f'{arm.lower()}.stderr.log').open('a',encoding='utf-8') as err:
                child=subprocess.Popen(command,env=env,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                save(folder/'state.json',{'status':'running','pid':os.getpid(),'arm':arm,'child_pid':child.pid,'command':command,'test_read':False})
                code=child.wait()
            status_path=HERE/f'runs/20261101_{arm.lower()}_seed0/qualification/status.json'
            import json
            status=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
            if code or status.get('status')!='completed':
                save(folder/'state.json',{'status':'error' if code else 'stopped','arm':arm,'returncode':code,'child_status':status,'test_read':False})
                raise SystemExit(code or 1)
        from .compare import combined
        combined()
        save(folder/'state.json',{'status':'completed','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'test_read':False})
if __name__=='__main__':main()
