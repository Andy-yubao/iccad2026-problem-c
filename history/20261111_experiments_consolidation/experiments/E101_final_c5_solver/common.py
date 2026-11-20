"""Final-run identities, bounded atomic writes, OS locks and stop boundaries."""
import os
for _k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import contextlib, datetime, hashlib, json, random, time, uuid, zipfile
import numpy as np
import torch
from datasets.paths import ROOT
HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/20261103_final64k_seed0'
ARCHIVE=ROOT/'history/20261103_final_delivery_consolidation'
PLAN=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def replace(a,b):
    for i in range(8):
        try: os.replace(a,b);return
        except PermissionError:
            if i==7:raise
            time.sleep(.05*(i+1))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False,default=lambda x:x.item() if isinstance(x,np.generic) else str(x))+'\n',encoding='utf-8');replace(t,p)
def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v
def checkpoint(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    torch.save(cpu(v),t);replace(t,p)
def stopped():return (RUN/'STOP').exists() or (HERE/'runs/STOP').exists()
def check_stop():
    if stopped():raise InterruptedError('User STOP: no subsequent stage')
def seed():
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def sources():
    files=[p for p in HERE.rglob('*.py') if 'runs' not in p.parts and '__pycache__' not in p.parts]
    files += [HERE/'plan.json',HERE/'provenance.json']
    for folder in ('datasets','scoring','visualization'):
        files += [p for p in (ROOT/folder).rglob('*.py') if '__pycache__' not in p.parts]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}
def snapshot():
    identity=sources();save(RUN/'sources.json',identity)
    with zipfile.ZipFile(RUN/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for n in identity:z.write(ROOT/n,n)
    return identity
@contextlib.contextmanager
def exclusive(p):
    import msvcrt
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:yield
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
