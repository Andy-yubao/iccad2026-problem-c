"""Strict weights-only transition from immutable own-Loss E76 best29."""
import json
import torch
from .io import ROOT,HERE,sha


def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))


def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,expected in values.items():
        if sha(ROOT/name)!=expected:raise ValueError('Protected checkpoint changed: '+name)
    return values


def metadata():
    p=provenance()
    if sha(ROOT/p['initialization_checkpoint'])!=p['initialization_sha256']:
        raise ValueError('E76 source changed')
    return {'mode':'Frozen entire E76 best29; public E74-predicted frame and fresh local corrector',
        'checkpoint':p['initialization_checkpoint'],'sha256':p['initialization_sha256'],
        'source_best_epoch':29,'source_best_step':29000,
        'pretraining_elapsed_seconds':p['pretraining_elapsed_seconds'],
        'optimizer':'Fresh AdamW only new corrector; old moments protected, no claim of exact continuation',
        'scheduler':'Fresh original four-tier rule; new stage max2h/200 epochs',
        'initial_function':'Projected E76, not exact unprojected source'}


def load(model,config):
    meta=metadata();state=torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)
    if state['model_config']!=config or state['identity']['implementation_check']:
        raise ValueError('Source architecture or role mismatch')
    for original,local in (('model.py','source_model.py'),('base_model.py','base_model.py'),
                           ('corrector.py','corrector.py'),('features.py','features.py'),('batches.py','batches.py')):
        expected=state['identity']['sources']['experiments/E76_bounded_refinement/'+original]
        if sha(HERE/local)!=expected:raise ValueError('Changed source semantics: '+original)
    model.source.load_state_dict(state['model'],strict=True)
    return state


def configure_trainability(model):
    model.source.requires_grad_(False);model.source.eval();return model
