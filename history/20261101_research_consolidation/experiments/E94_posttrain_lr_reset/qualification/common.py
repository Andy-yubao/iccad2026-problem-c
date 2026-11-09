"""E94 future unique own-best freeze; independent evaluation source registration."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha,checkpoint,sources as training_sources
from ..dataset_view import REGISTRY
HOME=Path(__file__).resolve().parent
ARM='F';RUN=HERE/'runs/20261101_lr1e3_seed0';OUTPUT=RUN/'qualification'
REGISTRATION=HERE/'runs/evaluation_registration.json'

def seed_all(seed,threads=4):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(threads)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

def registration():
    value=json.loads(REGISTRATION.read_text(encoding='utf-8'))
    for name,digest in value['evidence_files'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen baseline evidence changed: '+name)
    for name,item in json.loads((HOME/'provenance.json').read_text(encoding='utf-8')).items():
        if sha(HERE/name)!=item['sha256']:raise ValueError('Copied runtime drift: '+name)
    if sha(HERE/'model.py')!=sha(HOME/'runtime/model.py'):raise ValueError('Architecture drift')
    return value

def evaluator_sources():
    files=[p for p in HOME.rglob('*') if p.is_file() and p.suffix in ('.py','.md','.json') and '__pycache__' not in p.parts]
    files+=list((ROOT/'visualization').glob('*.py'))+[REGISTRATION]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}

def sources():return {**training_sources(),**evaluator_sources()}

def ready():
    value=json.loads((HERE/'state.json').read_text(encoding='utf-8'))
    return value['status'] in ('training_completed','validating','completed'),'normal training endpoint required'

def freeze():
    if not ready()[0]:raise RuntimeError(ready()[1])
    registration()
    checked=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if checked['evaluation_sources']!=evaluator_sources():raise ValueError('Evaluation source changed after preflight')
    latest=torch.load(RUN/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    path=RUN/'checkpoints/F_best.pt';state=torch.load(path,map_location='cpu',weights_only=False)
    if latest['status'] not in ('budget','max_epochs','early_stop') or latest['progress']['phase']!='F':raise ValueError('Abnormal training end')
    if state['progress']['phase']!='F' or state['identity']!=latest['identity']:raise ValueError('Best/terminal mismatch')
    if state['identity']['sources']!=training_sources() or state['identity']['registry_sha256']!=sha(REGISTRY):raise ValueError('Source/data drift')
    chosen=latest['progress']['best']['F'];epoch=latest['progress']['best_epoch']['F'];history=latest['progress']['history']
    if chosen!=min(r['dev_monitor']['total'] for r in history) or state['progress']['best']['F']!=chosen or state['progress']['phase_epoch']!=epoch:
        raise ValueError('Not unique fixed F Loss own-best')
    identity={'arm':'F','checkpoint':path.relative_to(ROOT).as_posix(),'checkpoint_sha256':sha(path),'best_epoch':epoch,
              'selected_dev500_loss':chosen,'model_config':state['model_config'],'sources':sources(),
              'registration_sha256':sha(REGISTRATION),'registry_sha256':sha(REGISTRY),'workers':4,'threads_per_worker':2,
              'count':500,'selection':'unique own fixed F Loss best; no Q reselect','test_read':False}
    OUTPUT.mkdir(parents=True,exist_ok=True);frozen=OUTPUT/'frozen.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=identity:raise ValueError('Frozen identity changed')
    save(frozen,identity);export=OUTPUT/'candidate_model.pt'
    if not export.exists():checkpoint(export,{'model':state['model'],'model_config':state['model_config'],'source_checkpoint_sha256':identity['checkpoint_sha256']})
    else:
        saved=torch.load(export,map_location='cpu',weights_only=False)
        if saved['source_checkpoint_sha256']!=identity['checkpoint_sha256'] or any(not torch.equal(saved['model'][k],v) for k,v in state['model'].items()):raise ValueError('Export changed')
    return state,identity

def auxiliary_seconds():
    p=HERE/'runs/evaluation_budget.json'
    return json.loads(p.read_text(encoding='utf-8'))['charged_seconds'] if p.exists() else 0.

def charge_auxiliary(seconds,label):
    p=HERE/'runs/evaluation_budget.json';value=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'charged_seconds':0.,'entries':[]}
    value['charged_seconds']+=seconds;value['entries'].append({'label':label,'seconds':seconds,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())});save(p,value)
