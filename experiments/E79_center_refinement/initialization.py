"""Freeze immutable E78 best21, initialize only new zero-output center head."""
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
    if sha(ROOT/p['initialization_checkpoint'])!=p['initialization_sha256']:raise ValueError('E78 source changed')
    return {'mode':'Frozen full E78 best21 and new zero-output center-only head',
        'checkpoint':p['initialization_checkpoint'],'sha256':p['initialization_sha256'],
        'source_best_epoch':21,'source_best_step':21000,'pretraining_elapsed_seconds':p['pretraining_elapsed_seconds'],
        'optimizer':'Fresh AdamW only new center head; source optimizer untouched',
        'scheduler':'New fixed-Loss stage, original four-tier rule, <=2h/200 epochs',
        'initial_function':'E78 physical layout within FP32 tolerance, all shapes exact; output additionally carries detached public source state'}
def load(model,config):
    meta=metadata();state=torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)
    if state['model_config']!=config or state['identity']['implementation_check']:raise ValueError('Source role/config mismatch')
    for name in ('model.py','base_model.py','source_model.py','pretrained_model.py','corrector.py','frame.py','mib_projection.py','features.py','batches.py'):
        local='warm_model.py' if name=='model.py' else name
        if sha(HERE/local)!=state['identity']['sources']['experiments/E78_connectivity_posttrain/'+name]:raise ValueError('Changed source semantics: '+name)
    model.source.load_state_dict(state['model'],strict=True)
    return state
def configure_trainability(model):
    model.source.requires_grad_(False);model.source.eval();return model
