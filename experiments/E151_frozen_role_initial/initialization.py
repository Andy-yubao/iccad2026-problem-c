"""Load and verify the immutable initial generator for this experiment."""
import torch

from .io import ROOT, sha


def load_parent_initial(plan):
    parent = plan['parent_initial']
    path = ROOT / parent['checkpoint']
    if sha(path) != parent['checkpoint_sha256']:
        raise ValueError('Frozen E149 checkpoint hash mismatch')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if (checkpoint['identity']['revision'] != 'E149.role_initial.v1.seed0'
            or checkpoint['progress']['best_epoch'] != parent['best_epoch']):
        raise ValueError('Unexpected frozen E149 checkpoint identity')
    return checkpoint['model']


def verify_parent_initial(model, initial):
    current = model.initial_generator.state_dict()
    if (model.initial_generator.training
            or any(p.requires_grad or p.grad is not None for p in model.initial_generator.parameters())
            or current.keys() != initial.keys()
            or not all(torch.equal(value.detach().cpu(), initial[name]) for name, value in current.items())):
        raise RuntimeError('E149 L0 changed or entered train mode')
