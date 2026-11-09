"""Branch from immutable E42 best51; never write the source checkpoints."""
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
        raise ValueError('E42 best51 initialization changed')
    return {'mode': 'weights-only post-training from E42 best51',
            'checkpoint': p['initialization_checkpoint'], 'sha256': p['initialization_sha256'],
            'source_best_epoch': 51, 'source_best_step': 25500, 'pretraining_elapsed_seconds': p['pretraining_elapsed_seconds'],
            'optimizer': 'fresh AdamW; source optimizer state untouched',
            'scheduler': 'new post-training stage for all arms; fresh fixed four-tier policy and early-stop history',
            'budget': 'new stage at most 2h/200 epochs; inherited pretraining time recorded separately'}


def load(model, config):
    identity = metadata()
    state = torch.load(ROOT/identity['checkpoint'], map_location='cpu', weights_only=False)
    if state['model_config'] != config or state['identity']['implementation_check']:
        raise ValueError('Source architecture or checkpoint role mismatch')
    for name in ('model.py', 'features.py', 'batches.py'):
        old = state['identity']['sources']['experiments/E42_joint_regularization/'+name]
        if sha(HERE/name) != old:
            raise ValueError('Source model/input semantics changed: '+name)
    model.load_state_dict(state['model'], strict=True)
    return state
