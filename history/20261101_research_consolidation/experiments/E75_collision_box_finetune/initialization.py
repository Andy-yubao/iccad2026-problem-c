"""Branch from immutable E74 best42; never write the source checkpoints."""
import json
import torch
from .io import ROOT, HERE, sha


def provenance():
    return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))


def verify_protected():
    values = provenance()['protected_checkpoints_sha256']
    for name, expected in values.items():
        if sha(ROOT/name) != expected:
            raise ValueError('Protected checkpoint changed: '+name)
    return values


def metadata():
    p = provenance()
    if sha(ROOT/p['initialization_checkpoint']) != p['initialization_sha256']:
        raise ValueError('E74 best42 initialization changed')
    return {'mode': 'E74 best42 weights; unchanged width192; new fixed collision/box post-training',
            'checkpoint': p['initialization_checkpoint'], 'sha256': p['initialization_sha256'],
            'source_best_epoch': p['source_best_epoch'], 'source_best_step': p['source_best_step'], 'pretraining_elapsed_seconds': p['pretraining_elapsed_seconds'],
            'optimizer': 'fresh AdamW; source optimizer state untouched',
            'scheduler': 'new post-training stage for all arms; fresh fixed four-tier policy and early-stop history',
            'budget': 'new stage at most 2h/200 epochs; inherited pretraining time recorded separately'}


def load(model, config):
    identity = metadata()
    state = torch.load(ROOT/identity['checkpoint'], map_location='cpu', weights_only=False)
    old_config=dict(config)
    if state['model_config'] != old_config or state['identity']['implementation_check']:
        raise ValueError('Source architecture or checkpoint role mismatch')
    for name in ('model.py', 'features.py', 'batches.py'):
        old = state['identity']['sources']['experiments/E74_collision_32k_posttrain/'+name]
        if sha(HERE/name) != old:
            raise ValueError('Source model/input semantics changed: '+name)
    model.load_state_dict(state['model'],strict=True)
    return state


def configure_trainability(model):
    """Retain the learned graph mapping, fixed throughout this post-training."""
    for layer in model.layers:
        for parameter in layer.local.parameters():parameter.requires_grad_(False)
    return model
