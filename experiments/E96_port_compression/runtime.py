"""Separate prepared arms, deterministic state and explicit execution boundary."""
import json,random,time
import numpy as np
import torch
from .io import HERE,save,sources

ARMS=('centroid',)
BASE=dict(width=192,layers=12,heads=4,dropout=.05,normalize_local=True,attention_backend='efficient')
CONFIG=dict(BASE,port_mode='centroid')

def model_config(arm,device='cuda'):
    if arm not in ARMS:raise ValueError('Unregistered arm')
    return dict(BASE,port_mode=arm,attention_backend='manual' if device=='cpu' else 'efficient')

def run_dir(arm):
    if arm not in ARMS:raise ValueError('Unregistered arm')
    return HERE/f'runs/20261101_{arm}_bbox05_seed0'

OUTPUT=run_dir('centroid')

def seed(value=0,threads=2):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False

def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])

def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda']:torch.cuda.set_rng_state_all(value['cuda'])

def charge(seconds,kind='preparation',output=None):
    path=(output if output is not None else HERE/'runs')/f'{kind}_budget.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(charged_seconds=0.,checks=[])
    value['charged_seconds']+=seconds;value['checks'].append(dict(seconds=seconds,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    save(path,value)

def preflight_dir(arm,device):
    return HERE/f'runs/preflight_bbox05_{device}_{arm}'

def verify_preflight(arm,device='cuda'):
    path=preflight_dir(arm,device)/'results.json'
    if not path.exists():raise ValueError(f'{device} preflight required for {arm}')
    value=json.loads(path.read_text(encoding='utf-8'))
    if value['status']!='passed' or value['sources']!=sources():raise ValueError('Current-source preflight required')
    return value

def state(arm,value):
    path=HERE/'state.json';old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(arms={})
    old.setdefault('arms',{})[arm]=value;old.update(status=value['status'],next_arm='centroid',automatic_next_arm=False,test_read=False,
        formal_training_started=old.get('formal_training_started',False) or value['status'] in ('running','evaluation','complete'))
    save(path,old)
