"""Fork completed E74 best with its Adam moments and RNG; preserve the original."""
import json,torch
from .io import HERE,ROOT,sha
def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
def verify_protected():
    p=provenance();name=p['checkpoint']
    if sha(ROOT/name)!=p['sha256']:raise ValueError('Frozen E74 parent changed')
    return {name:p['sha256']}
def metadata():
    verify_protected();p=provenance()
    return dict(checkpoint=p['checkpoint'],sha256=p['sha256'],source_step=42000,source_epoch=42,
        optimizer='preserve Adam moments/steps in both arms; new-stage LR=1e-3',rng='same parent RNG',
        scheduler='completed-parent branch: epoch0 baseline, no protection',prior_global_training_seconds=38831.07569789997,parent_training_exposure=32000)
def configure_trainability(model):
    for layer in model.layers:
        for parameter in layer.local.parameters():parameter.requires_grad_(False)
    return model
def load(model,config):
    verify_protected();p=provenance();s=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=False)
    if s['model_config']!=config or s['identity']['implementation_check']:raise ValueError('Parent architecture/role changed')
    for name in ('model.py','features.py','batches.py'):
        if sha(HERE/name)!=s['identity']['sources']['experiments/E74_collision_32k_posttrain/'+name]:raise ValueError('Parent semantics changed')
    model.load_state_dict(s['model'],strict=True);return s
