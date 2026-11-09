"""Immutable common starting weights; optimizer/scheduler are fresh per branch."""
import torch
from .io import ROOT,HERE,sha

SOURCE=ROOT/'experiments/E53_loss_geometry/runs/20261026_geometry_seed0/checkpoints/best.pt'
DIGEST='1cc0e5ad5dad483e1bc6f35a9d26333a5ee2aa591de6f42f79128e254261d3ef'

def metadata():
    if sha(SOURCE)!=DIGEST:raise ValueError('Common E53 best62 starting checkpoint changed')
    return {'mode':'weights-only branch from E53 geometry best62',
            'checkpoint':SOURCE.relative_to(ROOT).as_posix(),'sha256':DIGEST,
            'optimizer':'fresh AdamW, moments and step reset identically in both arms',
            'scheduler':'fresh fixed four-tier policy; new objective/early-stop history',
            'seed_order':'both seed0, identical new data order and RNG initialization',
            'inherited_reference_training':True}

def load(model, model_config):
    metadata();state=torch.load(SOURCE,map_location='cpu',weights_only=False)
    if state['model_config']!=model_config:raise ValueError('Source architecture differs')
    for name in ('model.py','features.py','batches.py'):
        expected=state['identity']['sources']['experiments/E53_loss_geometry/'+name]
        if sha(HERE/name)!=expected:raise ValueError('Source tensor/model semantics differ: '+name)
    model.load_state_dict(state['model'],strict=True)
    return state
