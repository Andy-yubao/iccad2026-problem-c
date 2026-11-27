import torch

from .io import ROOT, sha


def load_parent(plan):
    registered = plan['parent']
    path = ROOT / registered['checkpoint']
    if sha(path) != registered['checkpoint_sha256']:
        raise ValueError('E151 checkpoint hash changed')
    state = torch.load(path, map_location='cpu', weights_only=False)
    if (state['identity']['revision'] != 'E151.frozen_role_initial.v1.seed0'
            or state['progress']['epoch'] != registered['best_epoch']
            or state['progress']['best_epoch'] != registered['best_epoch']):
        raise ValueError('Unexpected E151 checkpoint identity')
    return state


def verify_frozen_prefix(model, initial):
    for name, parameter in model.named_parameters():
        trained = name.startswith('units.2.')
        if parameter.requires_grad != trained:
            raise RuntimeError(f'Trainability changed: {name}')
        if not trained and (parameter.grad is not None or not torch.equal(parameter.detach().cpu(), initial[name])):
            raise RuntimeError(f'Frozen parameter changed: {name}')
    if any(module.training for frozen in model.frozen_modules for module in frozen.modules()):
        raise RuntimeError('Frozen prefix entered train mode')
