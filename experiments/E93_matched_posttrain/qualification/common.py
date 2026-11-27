"""E93 branch-specific frozen evaluation, without E92 research lifecycle actions."""
import json,os,random,time
from pathlib import Path
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha,checkpoint,sources as training_sources
from ..dataset_view import REGISTRY

HOME=Path(__file__).resolve().parent
ARM=os.environ.get('E93_EVAL_ARM','C')
if ARM not in ('C','G'):raise ValueError('Invalid evaluation arm')
RUN=HERE/f'runs/20261101_{ARM.lower()}_seed0'
OUTPUT=RUN/'qualification'
REGISTRATION=HERE/'runs/evaluation_registration.json'

def seed_all(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def registration():
    value=json.loads(REGISTRATION.read_text(encoding='utf-8'))
    for name,digest in value['evidence_files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen evidence changed: '+name)
    provenance=json.loads((HOME/'provenance.json').read_text(encoding='utf-8'))
    for name,item in provenance.items():
        if sha(HERE/name)!=item['sha256']:raise ValueError('Copied runtime changed: '+name)
    if sha(HERE/'model.py')!=sha(HOME/'runtime/model.py'):raise ValueError('Architecture drift')
    return value

def sources():
    result=training_sources()
    files=[p for p in HOME.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json') and '__pycache__' not in p.parts]
    files+=list((ROOT/'visualization').glob('*.py'))+[REGISTRATION]
    result.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
    return result

def ready():
    value=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    if value['status']!='completed' or value['completed_arms']!=['C','G']:
        return False,'both branches must finish before evaluation'
    return True,'both branches completed normally'

def freeze():
    ok,reason=ready()
    if not ok:raise RuntimeError(reason)
    reg=registration();path=RUN/f'checkpoints/{ARM}_best.pt'
    latest=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if latest['status'] not in ('budget','max_epochs','early_stop'):raise ValueError('Abnormal training end')
    if latest['progress']['phase']!=ARM or state['progress']['phase']!=ARM or state['identity']!=latest['identity']:
        raise ValueError('Branch/identity mismatch')
    if state['identity']['sources']!=training_sources() or state['identity']['registry_sha256']!=sha(REGISTRY):
        raise ValueError('Training source or dataset drift')
    chosen=latest['progress']['best'][ARM];epoch=latest['progress']['best_epoch'][ARM]
    history=[r for r in latest['progress']['history'] if r['phase']==ARM]
    if not history or chosen!=min(r['dev_monitor']['total'] for r in history):raise ValueError('Wrong own-Loss best')
    if state['progress']['best'][ARM]!=chosen or state['progress']['phase_epoch']!=epoch:raise ValueError('Wrong checkpoint epoch')
    expected=reg['candidates'][ARM]
    if sha(path)!=expected['sha256'] or epoch!=expected['best_epoch'] or chosen!=expected['selected_loss']:
        raise ValueError('Registered candidate drift')
    identity={'arm':ARM,'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),
              'best_epoch':epoch,'selected_dev500_loss':chosen,'model_config':state['model_config'],
              'sources':sources(),'registration_sha256':sha(REGISTRATION),'registry_sha256':sha(REGISTRY),
              'workers':4,'threads_per_worker':2,'count':500,'selection':'unique own-arm fixed Loss best; no Q reselect','test_read':False}
    OUTPUT.mkdir(parents=True,exist_ok=True);frozen=OUTPUT/'frozen.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=identity:raise ValueError('Frozen identity changed')
    save(frozen,identity);export=OUTPUT/'candidate_model.pt'
    if not export.exists():checkpoint(export,{'model':state['model'],'model_config':state['model_config'],'source_checkpoint_sha256':identity['checkpoint_sha256']})
    else:
        item=torch.load(export,map_location='cpu',weights_only=False)
        if item['source_checkpoint_sha256']!=identity['checkpoint_sha256'] or any(not torch.equal(item['model'][k],v) for k,v in state['model'].items()):
            raise ValueError('Export drift')
    return state,identity

def auxiliary_seconds():
    path=HERE/'runs/evaluation_budget.json'
    return json.loads(path.read_text(encoding='utf-8'))['charged_seconds'] if path.exists() else 0.

def charge_auxiliary(seconds,label):
    path=HERE/'runs/evaluation_budget.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'charged_seconds':0.,'entries':[]}
    value['charged_seconds']+=seconds
    value['entries'].append({'label':label,'seconds':seconds,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    save(path,value)
