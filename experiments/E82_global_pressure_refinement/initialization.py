"""Preserve every E81 parameter; learn only zero public-pressure embeddings."""
import json
import torch
from .io import HERE,ROOT,sha
def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,digest in values.items():
        if sha(ROOT/name)!=digest:raise ValueError('Protected checkpoint changed: '+name)
    return values
def metadata():
    p=provenance()
    if sha(ROOT/p['initialization_checkpoint'])!=p['initialization_sha256']:raise ValueError('E81 parent changed')
    return {'mode':'All E81 placers and F weights frozen/eval; only three zero4-to192 pressure embeddings trained',
      'checkpoint':p['initialization_checkpoint'],'sha256':p['initialization_sha256'],
      'initial_function':'E81 within FP32 tolerance, no new random correction stage',
      'optimizer':'Fresh AdamW only2304 new weights; E81 optimizer and elapsed retained unchanged',
      'context':'Source-draft public frame/anchor lower and upper16-step path pressures, detached, clamped[-8,8], shared by F stages'}
def load(model,config):
    m=metadata();source=torch.load(ROOT/m['checkpoint'],map_location='cpu',weights_only=False)
    if source['model_config']!=config or source['identity']['implementation_check']:raise ValueError('Parent config/role mismatch')
    result=model.load_state_dict(source['model'],strict=False)
    expected={'refiner.'+name+'.pressure.0.weight' for name in ('first','second','third')}
    if set(result.missing_keys)!=expected or result.unexpected_keys:raise ValueError('Unexpected adapter mapping')
    for name in ('first','second','third'):
        if torch.count_nonzero(getattr(model.refiner,name).pressure[0].weight):raise ValueError('Nonzero adapter initialization')
    return source
def configure_trainability(model):
    model.requires_grad_(False)
    for name in ('first','second','third'):getattr(model.refiner,name).pressure.requires_grad_(True)
    model.placer.eval();model.alternative.eval()
    return model
