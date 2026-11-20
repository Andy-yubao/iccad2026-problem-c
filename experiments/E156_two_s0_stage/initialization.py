"""Load the fixed E155 best51 parent, with no optimizer migration."""
import json,torch
from .io import HERE,ROOT,sha,sources
def load_initial():
    frozen=json.loads((HERE/'frozen_initial.json').read_text(encoding='utf-8'));path=ROOT/frozen['checkpoint']
    if frozen!=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))['parent_initial']:raise ValueError('Frozen parent differs from registered best51')
    if sha(path)!=frozen['checkpoint_sha256']:raise ValueError('New L0 checkpoint changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']['revision']!='E155.fresh_mib_l0.v1.seed0' or state['identity']['arm']!='e155_initial':raise ValueError('Only frozen E155 best51 is allowed')
    if frozen['best_epoch']!=51 or state['progress']['best_epoch']!=51:raise ValueError('Parent selection changed')
    for name,digest in state['identity']['sources'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen E155 source changed: '+name)
    return {name.removeprefix('initial_generator.'):value for name,value in state['model'].items() if name.startswith('initial_generator.')}
def verify_initial(model,reference):
    if model.initial_generator.training:raise RuntimeError('Frozen L0 entered training mode')
    for name,parameter in model.initial_generator.named_parameters():
        if parameter.requires_grad or parameter.grad is not None or not torch.equal(parameter.detach().cpu(),reference[name]):raise RuntimeError('Frozen L0 changed')
