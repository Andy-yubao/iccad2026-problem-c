"""One authorized serial evaluation; idle wait uses no torch/GPU imports."""
from pathlib import Path
import json,os,sys,time,subprocess,datetime,hashlib,msvcrt
BUNDLE=Path(__file__).resolve().parent;ROOT=BUNDLE.parents[3]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(value):
    temp=BUNDLE/'state.tmp';temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(temp,BUNDLE/'state.json')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def main():
    with (BUNDLE/'queue.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        expected=read(BUNDLE/'controller_sources.json')
        assert all(hashlib.sha256((BUNDLE/n).read_bytes()).hexdigest()==h for n,h in expected.items())
        plan=read(BUNDLE/'plan.json');prior=Path(plan['serial_after'])
        save(dict(status='queued',stage='waiting_for_chain64k_to_finish',pid=os.getpid(),wait_for=str(prior),checkpoint_sha256=plan['checkpoint_sha256'],updated_utc=utc(),test_read=False))
        while read(prior)['status']!='complete':
            if (BUNDLE/'STOP').exists():raise InterruptedError('STOP while queued')
            if read(prior)['status'] in ('error','user_stop'):raise RuntimeError('Prior chain stopped; baseline not started and no retry scheduled')
            time.sleep(10)
        if (BUNDLE/'STOP').exists():raise InterruptedError('STOP before evaluation')
        if not (BUNDLE/'results.json').exists():
            env=dict(os.environ);env.pop('E16_REGISTRY',None)
            with (BUNDLE/'evaluation.log').open('a',encoding='utf-8') as log:
                p=subprocess.Popen([str(ROOT/'.conda/envs/iccad2026/python.exe'),'-u','-B',str(BUNDLE/'evaluate.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                save(dict(status='running',stage='original_C5_test2000',pid=os.getpid(),child_pid=p.pid,updated_utc=utc(),test_read=True))
                code=p.wait()
            if code and (BUNDLE/'STOP').exists():raise InterruptedError('Evaluation safely stopped between cases')
            if code:raise RuntimeError(f'Baseline evaluation exited{code}; see evaluation.log')
        assert read(BUNDLE/'results.json')['status']=='complete'
        save(dict(status='complete',stage='original_C5_test2000_complete',updated_utc=utc(),test_read=True,no_retraining=True,no_test_feedback=True))
if __name__=='__main__':
    try:main()
    except BaseException as e:save(dict(status='user_stop' if isinstance(e,InterruptedError) else 'error',error=repr(e),updated_utc=utc(),no_automatic_retry=True));raise
