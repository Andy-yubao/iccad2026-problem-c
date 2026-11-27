"""Same immutable E74/F81 weights in both arms; fresh continuation AdamW."""
import json
import torch
from .io import HERE,ROOT,sha
def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,digest in values.items():
        if sha(ROOT/name)!=digest:raise ValueError('Protected parent changed: '+name)
    return values
def metadata():
    verify_protected()
    return dict(mode='Frozen E74 best42 + exact F81 best26 weights in BOTH arms',parents=provenance()['parents'],optimizer='Fresh all-F AdamW in both arms',previous_global_charged_seconds=34604.39659300004,scope='New explicitly authorized phase; historical budget retained')
def load(model,config):
    verify_protected();parents=provenance()['parents']
    placer=torch.load(ROOT/parents['placer']['checkpoint'],map_location='cpu',weights_only=False)
    correction=torch.load(ROOT/parents['correction']['checkpoint'],map_location='cpu',weights_only=False)
    model.placer.load_state_dict(placer['model'],strict=True)
    model.refiner.load_state_dict({k[len('refiner.'):]:v for k,v in correction['model'].items() if k.startswith('refiner.')},strict=True)
    return correction
def configure_trainability(model):
    model.placer.requires_grad_(False);model.placer.eval();model.refiner.requires_grad_(True);return model
