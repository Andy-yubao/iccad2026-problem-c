"""Fresh E74-architecture weights; historical checkpoints are comparison only."""
import json
from .io import HERE,ROOT,sha
def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,digest in values.items():
        if sha(ROOT/name)!=digest:raise ValueError('Historical checkpoint changed: '+name)
    return values
def metadata():
    return dict(mode='random E74 architecture; no checkpoint load or Net2Wider',optimizer='fresh AdamW; no inherited moments/steps',rng='new seed0 stream',trainability='entire192x12 placer including graph-local; no frozen parameters',scheduler='fresh training; epochs1-14 protected, scheduling starts epoch15',forward='exact E74 inference architecture; no added corrector or MIB/frame projection')
def configure_trainability(model):
    model.requires_grad_(True);return model
def load(model,config):
    if not all(p.requires_grad for p in model.parameters()):raise ValueError('Unexpected frozen parameter')
    return None
