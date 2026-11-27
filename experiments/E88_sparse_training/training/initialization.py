"""Continue frozen E90 best12 with Adam and RNG intact; fresh LR1e-3 and scheduler."""
import json,torch
from .io import HERE,CODE,ROOT,sha

def provenance():return json.loads((CODE/'provenance.json').read_text(encoding='utf-8'))

def verify_protected():
    p=provenance();frozen=json.loads((ROOT/p['checkpoint']).parent.parent.joinpath('frozen.json').read_text(encoding='utf-8'))
    if sha(ROOT/p['checkpoint'])!=p['sha256'] or p['sha256']!=frozen['checkpoint_sha256']:raise ValueError('E90 best changed')
    if sha(ROOT/p['latest'])!=frozen['latest_checkpoint_sha256']:raise ValueError('E90 latest changed')
    return {p['checkpoint']:p['sha256'],p['latest']:frozen['latest_checkpoint_sha256']}

def metadata():
    verify_protected();p=provenance()
    return dict(checkpoint=p['checkpoint'],sha256=p['sha256'],source_step=6000,source_epoch=12,
        optimizer='preserve exact Adam moments and step tensors; explicit new-stage LR1e-3',rng='exact E90 best RNG',
        scheduler='new scheduler: epoch0 baseline, no protection, four exact tiers, patience3; inherited scheduler kept only in source checkpoint',
        progress='new data-stage cursor/epoch; source progress retained by immutable E90 checkpoint and provenance',
        parent_training_exposure=32000)

def configure_trainability(model):
    for layer in model.layers:
        for parameter in layer.local.parameters():parameter.requires_grad_(False)
    return model

def load(model,config):
    verify_protected();p=provenance();s=torch.load(ROOT/p['checkpoint'],map_location='cpu',weights_only=False)
    if s['model_config']!=config or s['identity']['implementation_check']:raise ValueError('Invalid E90 source')
    for name in ('model.py','features.py','batches.py','simple_loss.py','cuda_step.py','step_control.py'):
        if sha(CODE/name)!=s['identity']['sources'][p['model_source_prefix']+name]:raise ValueError('Core semantics changed: '+name)
    model.load_state_dict(s['model'],strict=True);return s
