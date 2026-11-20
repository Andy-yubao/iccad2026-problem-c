"""Local paths, atomic evidence and fixed numerical contracts."""
import hashlib
import json
import os
import time
from pathlib import Path
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OWNER=Path(__file__).resolve().parent
RUN=OWNER/'runs/20261104_bank1000x5_seed0'
KEYS=('position','shape','penetration','module_load_mean','module_max_excess10',
      'wirelength','compactness','boundary_contribution','grouping_contribution','mib_contribution')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
SAVE_RETRY_STATS=dict(events=0,retries=0,max_retries=0)
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    # Windows readers may briefly open a status file without FILE_SHARE_DELETE.
    # Keep the old complete JSON visible until the same atomic replace succeeds.
    # Persistent access failures still raise; never truncate/delete the target.
    deadline=time.monotonic()+5.
    retries=0
    while True:
        try:
            os.replace(tmp,p)
            if retries:
                SAVE_RETRY_STATS['events']+=1
                SAVE_RETRY_STATS['retries']+=retries
                SAVE_RETRY_STATS['max_retries']=max(SAVE_RETRY_STATS['max_retries'],retries)
            return
        except PermissionError as exc:
            if os.name!='nt' or exc.winerror not in (5,32,33) or time.monotonic()>=deadline:
                raise
            retries+=1
            time.sleep(min(.01*retries,.2,max(0.,deadline-time.monotonic())))

def sources():
    files=[p for p in OWNER.rglob('*.py') if not {'runs','assets','evidence','__pycache__'}.intersection(p.relative_to(OWNER).parts)]
    files += [OWNER/n for n in ('plan.json','model_manifest.json','panel.json','implementation_provenance.json','postprocessor/budget.json','README.md','launch.ps1')]
    files += list((ROOT/'datasets').glob('*.py'))+list((ROOT/'scoring').glob('*.py'))
    files += [ROOT/n for n in ('datasets/expanded_reference/__init__.py','protocol/paths.json','protocol/reference_workset_v5.json',
        'protocol/reference_workset_v6.json','protocol/split_v1.json','third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
        'third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}

def setup():
    import torch,numba
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.manual_seed(0);np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    numba.set_num_threads(1)

def check():
    if (RUN/'STOP').exists() or (OWNER/'STOP').exists():raise InterruptedError('E103 user stop')

def case_weights(ns):
    ns=np.asarray(ns,dtype=float)
    w=np.exp((ns-ns.max())/12)
    return .5*(w/w.sum()+np.ones(len(ns))/len(ns))

def q_summary(qs,ns):
    qs=np.asarray(qs,dtype=float);ns=np.asarray(ns,dtype=float)
    w=np.exp((ns-ns.max())/12)
    return dict(weighted_Q=float(qs@w/w.sum()),mean_Q=float(qs.mean()),mixed_Q=float(qs@case_weights(ns)))
