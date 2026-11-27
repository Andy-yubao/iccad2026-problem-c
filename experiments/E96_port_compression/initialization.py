"""Copy pretrained E74 neural tensors, never share mutable parameters."""
import json,torch
from .io import HERE,ROOT,sha

def parent():return json.loads((HERE/'provenance.json').read_text(encoding='utf-8'))['placer']
def load(model):
    source=parent();path=ROOT/source['checkpoint']
    if sha(path)!=source['sha256']:raise ValueError('Frozen E90 parent changed')
    state=torch.load(path,map_location='cpu',weights_only=False)
    model.placer.load_state_dict(state['model'],strict=True)
    model.corrector.input.load_state_dict(model.placer.input.state_dict())
    model.corrector.state_embed.load_state_dict(model.placer.state_embed.state_dict())
    for new,old in zip(model.corrector.layers,model.placer.layers):new.load_state_dict(old.state_dict(),strict=True)
    model.corrector.head[1].load_state_dict(model.placer.head[0].state_dict())
    for a,b in zip(model.corrector.layers.parameters(),model.placer.layers.parameters()):
        assert a.data_ptr()!=b.data_ptr()
    return source
