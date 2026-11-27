"""Exact full E77 weights; new fixed-Loss stage on its trained corrector."""
import json
import torch
from .io import ROOT,HERE,sha

def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))

def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,expected in values.items():
        if sha(ROOT/name)!=expected:raise ValueError('Protected checkpoint changed: '+name)
    return values

def metadata():
    p=provenance()
    if sha(ROOT/p['initialization_checkpoint'])!=p['initialization_sha256']:raise ValueError('E77 source changed')
    return {'mode':'Full E77 best38 weights; public MIB projection and new fixed corridor Loss; same trained corrector',
        'checkpoint':p['initialization_checkpoint'],'sha256':p['initialization_sha256'],
        'source_best_epoch':38,'source_best_step':38000,'pretraining_elapsed_seconds':p['pretraining_elapsed_seconds'],
        'optimizer':'Fresh AdamW only existing last corrector, source state unchanged',
        'initial_function':'MIB-coherent projection of E77; not exact unprojected function',
        'scheduler':'New fixed-Loss stage with fresh original four-tier rule and<=2h/200 epochs',
        'budget':'Phase2 new candidate; all phase1 cost recorded separately, no relabeling as exact continuation'}

def load(model,config):
    meta=metadata();state=torch.load(ROOT/meta['checkpoint'],map_location='cpu',weights_only=False)
    if state['model_config']!=config or state['identity']['implementation_check']:raise ValueError('Source role/config mismatch')
    for name in ('model.py','base_model.py','source_model.py','corrector.py','frame.py','features.py','batches.py'):
        local='pretrained_model.py' if name=='model.py' else name
        if sha(HERE/local)!=state['identity']['sources']['experiments/E77_compact_refinement/'+name]:
            raise ValueError('Changed pretrained model semantics: '+name)
    model.load_state_dict(state['model'],strict=True)
    return state

def configure_trainability(model):
    model.source.requires_grad_(False);model.source.eval();return model
