"""Frozen identity, lifecycle guard, task auxiliary budget and readiness checks."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha,checkpoint,sources as training_sources
from ..train import RUN
from ..dataset_view import REGISTRY

HOME=Path(__file__).resolve().parent
OUTPUT=RUN/'qualification'
REGISTRATION=HERE/'runs/qualification_registration.json'

def seed_all(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def registration():
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
    if sha(REGISTRATION)!=study['qualification_registration_sha256']:
        raise ValueError('Pre-registered qualification changed')
    value=json.loads(REGISTRATION.read_text(encoding='utf-8'))
    for name,digest in value['baseline']['source_files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen E90 evidence changed: '+name)
    provenance=json.loads((HOME/'provenance.json').read_text(encoding='utf-8'))
    for name,item in provenance.items():
        if sha(HERE/name)!=item['sha256']:raise ValueError('Frozen copied evaluator changed: '+name)
    if sha(HERE/'model.py')!=sha(HOME/'runtime/model.py'):
        raise ValueError('Terminal and training architectures differ')
    return value

def sources():
    result=training_sources()
    files=[p for p in HOME.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json') and '__pycache__' not in p.parts]
    files += list((ROOT/'visualization').glob('*.py'))+[REGISTRATION]
    result.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
    return result

def ready():
    value=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    if value['status']=='running':return False,'layout training still active'
    if value['status'] not in ('budget','max_epochs','early_stop'):
        return False,'layout did not end normally: '+value['status']
    return True,'normal training end; F own-best must still be checked'

def freeze():
    ok,reason=ready()
    if not ok:raise RuntimeError(reason)
    reg=registration();path=RUN/'checkpoints/F_best.pt'
    if not path.exists():raise RuntimeError('No F own-best; incomplete curriculum is a failed gate, not permission to force a phase switch')
    latest=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if latest['status'] not in ('budget','max_epochs','early_stop') or latest['progress']['phase']!='F':
        raise ValueError('Wrong terminal training state')
    if state['progress']['phase']!='F' or state['identity']!=latest['identity']:
        raise ValueError('Own-best phase/identity mismatch')
    if state['identity']['sources']!=training_sources() or state['identity']['registry_sha256']!=sha(REGISTRY):
        raise ValueError('Training source or dataset drift')
    chosen=latest['progress']['best']['F'];epoch=latest['progress']['best_epoch']['F']
    history=[r for r in latest['progress']['history'] if r['phase']=='F']
    if not history or chosen!=min(r['dev_monitor']['total'] for r in history):
        raise ValueError('Not the registered mean-Loss best')
    if state['progress']['best']['F']!=chosen or state['progress']['phase_epoch']!=epoch:
        raise ValueError('Checkpoint is not F own-Loss best')
    identity={'checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),
              'best_F_epoch':epoch,'selected_dev500_F_loss':chosen,'model_config':state['model_config'],
              'sources':sources(),'registration_sha256':sha(REGISTRATION),
              'registry_sha256':sha(REGISTRY),'workers':4,'threads_per_worker':2,'count':500,
              'selection':'unique F own-Loss best; no Q or iteration-count reselect','test_read':False}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    frozen=OUTPUT/'frozen.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Cannot replace frozen candidate/evaluation')
    save(frozen,identity)
    export=OUTPUT/'candidate_model.pt'
    if not export.exists():
        checkpoint(export,{'model':state['model'],'model_config':state['model_config'],
                           'source_checkpoint_sha256':identity['checkpoint_sha256']})
    else:
        item=torch.load(export,map_location='cpu',weights_only=False)
        if item['source_checkpoint_sha256']!=identity['checkpoint_sha256'] or any(not torch.equal(item['model'][k],v) for k,v in state['model'].items()):
            raise ValueError('Frozen exported model drift')
    deployment=OUTPUT/'deployment.json'
    if not deployment.exists():
        save(deployment,{'candidate_model':export.relative_to(ROOT).as_posix(),
             'candidate_model_sha256':sha(export),'source_checkpoint_sha256':identity['checkpoint_sha256'],
             'approved_for_corrector_training':False,'approval_authority':'assessment.json; must pass all registered gates'})
    elif json.loads(deployment.read_text(encoding='utf-8'))['candidate_model_sha256']!=sha(export):
        raise ValueError('Deployment artifact changed')
    return state,identity

def auxiliary_seconds():
    total=0.
    for p,key in [(HERE/'runs/preparation_budget.json','charged_seconds'),(HERE/'runs/task_auxiliary_budget.json','charged_seconds')]:
        if p.exists():total+=json.loads(p.read_text(encoding='utf-8'))[key]
    study=json.loads((HERE/'study.json').read_text(encoding='utf-8'))
    for name in study.get('corrector_preparation_ledgers',[]):
        total+=json.loads((ROOT/name).read_text(encoding='utf-8'))['charged_seconds']
    return total

def charge_auxiliary(seconds,label):
    path=HERE/'runs/task_auxiliary_budget.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'charged_seconds':0.,'entries':[]}
    value['charged_seconds']+=seconds
    value['entries'].append({'label':label,'seconds':seconds,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    save(path,value)
