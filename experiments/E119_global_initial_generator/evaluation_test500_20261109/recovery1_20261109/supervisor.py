"""Console-free supervisor; one worker, explicit status, no retry."""
import hashlib,json,os,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def now():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.'+str(os.getpid())+'.tmp')
    temp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for attempt in range(30):
        try:os.replace(temp,p);return
        except PermissionError:
            if attempt==29:raise
            time.sleep(.05)
state=dict(status='starting',supervisor_pid=os.getpid(),started_at=now(),automatic_retry=False,training=False,test_read=True)
def beat(**kw):state.update(kw);save(HERE/'supervisor_state.json',dict(**state,updated_at=now()))
try:
    if (HERE/'supervisor_state.json').exists():raise FileExistsError('Recovery already dispatched; no retry')
    beat()
    plan=read(HERE/'plan.json')
    for name,digest in plan['sources'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,('Source changed',name)
    assert not (HERE/'STOP').exists() and not (HERE.parent/'STOP').exists()
    with (HERE/'stdout.log').open('x',encoding='utf-8') as out,(HERE/'stderr.log').open('x',encoding='utf-8') as err:
        child=subprocess.Popen([str(ROOT/'.conda/envs/iccad2026/python.exe'),'-B','-u',str(HERE/'run.py')],
            cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,close_fds=True,
            creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
        beat(status='running',worker_pid=child.pid)
        while child.poll() is None:
            progress=HERE/'output/state.json'
            if progress.exists():
                try:
                    p=read(progress);beat(completed=p['completed'],stage=p['stage'],worker_updated_at=p['updated_at'])
                except (OSError,json.JSONDecodeError):pass
            else:beat(stage='worker_imports_and_identity_verification')
            time.sleep(3)
    result=HERE/'output/results.json'
    if child.returncode==0 and result.exists() and read(result)['status']=='complete':
        beat(status='complete',completed=500,returncode=0,completed_at=now())
    else:raise RuntimeError('Worker terminated: returncode='+str(child.returncode)+'; no retry')
except BaseException as error:
    beat(status='needs_attention',error=repr(error),traceback=traceback.format_exc())
    sys.exit(1)
