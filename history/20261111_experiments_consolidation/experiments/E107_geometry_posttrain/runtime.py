"""Frozen E105 parent, no optimized tensors or reference inputs."""
import json
import torch
from .io import HERE,sha
from .model import GeometryModel

def parent_model():
    plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'));path=HERE/plan['parent_checkpoint']
    if sha(path)!=plan['parent_checkpoint_sha256']:raise ValueError('Frozen parent drift')
    state=torch.load(path,map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=[]):model=GeometryModel(**state['model_config'])
    model=model.cuda().eval();model.load_state_dict(state['model'],strict=True)
    model.requires_grad_(False)
    return model
