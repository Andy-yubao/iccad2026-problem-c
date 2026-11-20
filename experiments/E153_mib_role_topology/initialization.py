"""Only this experiment's new L0 may initialize its own backend stage."""
import json,torch
from .io import HERE,ROOT,sha,sources
def load_initial():
    frozen=json.loads((HERE/'frozen_initial.json').read_text(encoding='utf-8'));path=ROOT/frozen['checkpoint']
    if sha(path)!=frozen['checkpoint_sha256']:raise ValueError('New L0 checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']['revision']!='E153.mib_role_topology.v1.seed0' or state['identity']['arm']!='e153_initial' or state['identity']['sources']!=sources():raise ValueError('Only E153 fresh L0 is allowed')
    return {name.removeprefix('initial_generator.'):value for name,value in state['model'].items() if name.startswith('initial_generator.')}
def verify_initial(model,reference):
    if model.initial_generator.training:raise RuntimeError('Frozen L0 entered training mode')
    for name,parameter in model.initial_generator.named_parameters():
        if parameter.requires_grad or parameter.grad is not None or not torch.equal(parameter.detach().cpu(),reference[name]):raise RuntimeError('Frozen L0 changed')
