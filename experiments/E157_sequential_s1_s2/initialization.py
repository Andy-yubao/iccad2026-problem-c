"""Transfer only frozen upstream weights; fresh parameters/Adam at each phase."""
import json,torch
from .io import HERE,ROOT,sha,sources

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def parent_descriptor(arm):
    return read(HERE/('parent_s0.json' if arm=='e157_s1' else 'frozen_s1.json'))

def initialize_model(model,arm,disposable_parent=None):
    if disposable_parent is not None:
        mapped={k:v for k,v in disposable_parent.items()}
    else:
        desc=parent_descriptor(arm);path=ROOT/desc['checkpoint']
        if sha(path)!=desc['checkpoint_sha256']:raise ValueError('Frozen parent checkpoint changed')
        state=torch.load(path,map_location='cpu',weights_only=False)
        if arm=='e157_s1':
            if desc!=read(HERE/'plan.json')['parent_s0'] or desc['best_epoch']!=79:raise ValueError('Expected E156 best79')
            if state['identity']['revision']!='E156.two_s0.v1.seed0' or state['identity']['arm']!='e156_s0_backend':raise ValueError('Wrong S0 identity')
            for name,digest in state['identity']['sources'].items():
                if sha(ROOT/name)!=digest:raise ValueError('Parent S0 source changed: '+name)
            mapped={('s0_encoder.'+k[len('problem_encoder.'):]) if k.startswith('problem_encoder.') else k:v for k,v in state['model'].items()}
        else:
            if state['identity']['sources']!=sources() or state['identity']['arm']!='e157_s1':raise ValueError('Only own frozen S1 best may initialize S2')
            mapped=state['model']
    target=model.state_dict();frozen={k:v for k,v in mapped.items() if k.split('.')[0] in model.frozen_names}
    expected={k for k in target if k.split('.')[0] in model.frozen_names}
    if set(frozen)!=expected:raise ValueError('Frozen prefix coverage changed')
    target.update(frozen);model.load_state_dict(target)
    return {k:v.detach().cpu().clone() for k,v in frozen.items()}

def verify_frozen(model,reference):
    current=model.state_dict()
    for k,v in reference.items():
        if not torch.equal(current[k].detach().cpu(),v):raise RuntimeError('Frozen prefix changed: '+k)
    for name in model.frozen_names:
        module=getattr(model,name)
        if module.training:raise RuntimeError('Frozen prefix entered train mode: '+name)
        if any(p.requires_grad or p.grad is not None for p in module.parameters()):raise RuntimeError('Frozen prefix has gradients')
