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
    return {'mode': 'Frozen E74 best42 encoder; new zero-initialized bounded correction only',
            'checkpoint': p['initialization_checkpoint'], 'sha256': p['initialization_sha256'],
            'source_best_epoch': p['source_best_epoch'], 'source_best_step': p['source_best_step'], 'pretraining_elapsed_seconds': p['pretraining_elapsed_seconds'],
            'optimizer': 'fresh AdamW; source optimizer state untouched',
            'scheduler': 'new post-training stage for all arms; fresh fixed four-tier policy and early-stop history',
            'budget': 'new stage at most 2h/200 epochs; inherited pretraining time recorded separately'}


def load(model, config):
    identity = metadata()
    state = torch.load(ROOT/identity['checkpoint'], map_location='cpu', weights_only=False)
    old_config={key:config[key] for key in ('width','layers','rectangle','ffn_dropout')}
    if state['model_config'] != old_config or state['identity']['implementation_check']:
        raise ValueError('Source architecture or checkpoint role mismatch')
    for name in ('model.py', 'features.py', 'batches.py'):
        old = state['identity']['sources']['experiments/E74_collision_32k_posttrain/'+name]
        if sha(HERE/('base_model.py' if name=='model.py' else name)) != old:
            raise ValueError('Source model/input semantics changed: '+name)
    model.backbone.load_state_dict(state['model'],strict=True)
    return state


def configure_trainability(model):
    model.backbone.requires_grad_(False);model.backbone.eval();return model
