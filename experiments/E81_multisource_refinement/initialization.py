"""Uniform Net2Wider of all three F heads; placers stay immutable/eval."""
import json
import torch
from .io import HERE,ROOT,sha

def provenance():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))
def verify_protected():
    values=provenance()['protected_checkpoints_sha256']
    for name,expected in values.items():
        if sha(ROOT/name)!=expected:raise ValueError('Protected checkpoint changed: '+name)
    return values
def metadata():
    p=provenance()
    for row in p['parents'].values():
        if sha(ROOT/row['checkpoint'])!=row['sha256']:raise ValueError('Warm-start parent changed')
    return {'mode':'Frozen P42 deployment and P74 training source; warm uniform widening F96/32 to192/64',
      'parents':p['parents'],'optimizer':'Fresh AdamW for all F, source moments retained separately',
      'initial_function':'E80 on P42 and E79 on P74, within preregistered FP32 tolerance',
      'scope':'No new randomly initialized correction stage, no placer update, no budget reset'}

def widen(old,new_width=192,new_message=64):
    """Duplicate hidden channels and divide each replicated incoming weight."""
    ow=old['node.0.weight'].shape[0];om=old['edge.weight'].shape[0]
    if new_width%ow or new_message%om:raise ValueError('Only uniform integer duplication is supported')
    node=torch.arange(new_width)%ow;message=torch.arange(new_message)%om
    joined=torch.cat((node,message+ow));copies=torch.cat((torch.full((new_width,),new_width/ow),torch.full((new_message,),new_message/om)))
    out={}
    for key,value in old.items():
        if key in ('node.0.weight','node.0.bias','node.2.weight','node.2.bias'):
            out[key]=value.index_select(0,node)
        elif key in ('receiver.weight','sender.weight'):
            out[key]=value.index_select(0,message).index_select(1,node)/(new_width/ow)
        elif key in ('receiver.bias','edge.weight','edge.bias'):
            out[key]=value.index_select(0,message)
        elif key=='score.weight':out[key]=value.index_select(1,message)/(new_message/om)
        elif key=='score.bias':out[key]=value.clone()
        elif key=='output.0.weight':out[key]=value.index_select(0,node).index_select(1,joined)/copies[None]
        elif key=='output.0.bias':out[key]=value.index_select(0,node)
        elif key=='output.3.weight':out[key]=value.index_select(1,node)/(new_width/ow)
        elif key=='output.3.bias':out[key]=value.clone()
        else:raise ValueError('Unexpected F tensor: '+key)
    return out

def load(model,config):
    meta=metadata();p=provenance();parent=p['parents']
    source=torch.load(ROOT/parent['correction']['checkpoint'],map_location='cpu',weights_only=False)
    placer=torch.load(ROOT/parent['placer']['checkpoint'],map_location='cpu',weights_only=False)
    model.placer.load_state_dict(placer['model'],strict=True)
    prefix='source.source.backbone.'
    model.alternative.load_state_dict({k[len(prefix):]:v for k,v in source['model'].items() if k.startswith(prefix)},strict=True)
    for new,prefix in (('first','source.source.corrector.'),('second','source.corrector.'),('third','corrector.')):
        old={k[len(prefix):]:v for k,v in source['model'].items() if k.startswith(prefix)}
        getattr(model.refiner,new).load_state_dict(widen(old,config['corrector_width'],config['message_width']),strict=True)
    return source

def configure_trainability(model):
    model.placer.requires_grad_(False);model.placer.eval()
    model.alternative.requires_grad_(False);model.alternative.eval()
    model.refiner.requires_grad_(True)
    return model
