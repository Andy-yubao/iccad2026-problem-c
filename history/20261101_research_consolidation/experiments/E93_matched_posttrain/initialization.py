"""One immutable P best36 for weights/Adam, P terminal39 for RNG/data schedule."""
import json,copy
import torch
from .io import ROOT,HERE,sha

def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))

def verify():
    value=provenance()
    for name,expected in value['protected_artifacts'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Protected parent changed: '+name)
    return value

def load_parent(model,optimizer):
    value=verify()
    parent=torch.load(ROOT/value['parent_model_and_optimizer'],map_location='cpu',weights_only=False)
    terminal=torch.load(ROOT/value['parent_rng_and_data_epoch'],map_location='cpu',weights_only=False)
    if parent['progress']['phase']!='P' or parent['progress']['phase_epoch']!=36 or parent['progress']['step']!=36000:
        raise ValueError('Requested P best36 not found')
    if terminal['progress']['phase']!='P' or terminal['progress']['epoch']!=39 or terminal['progress']['step']!=39000:
        raise ValueError('Wrong P terminal RNG/sampling boundary')
    if parent['identity']['registry_sha256']!=terminal['identity']['registry_sha256']:
        raise ValueError('Parent dataset versions differ')
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('Unregistered freezing')
    model.load_state_dict(parent['model']);optimizer.load_state_dict(parent['optimizer'])
    for group in optimizer.param_groups:
        if group['weight_decay']!=.0003:raise ValueError('Unexpected inherited weight decay')
        group['lr']=1e-4
    return {'model_step':36000,'model_epoch':36,'data_epoch':39,'rng':terminal['rng'],
            'seen':parent['progress']['seen'].copy(),'projection':copy.deepcopy(parent['graph_step_control']),
            'parent_elapsed_seconds':parent['elapsed_seconds'],
            'parent_checkpoint_sha256':sha(ROOT/value['parent_model_and_optimizer']),
            'rng_checkpoint_sha256':sha(ROOT/value['parent_rng_and_data_epoch'])}
