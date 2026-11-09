"""Read-only indexed E17 cache reuse; free reference geometry is supervision."""
import torch
from datasets.reference_layouts import public_hard_reference
from datasets.reference_layouts import ReferenceLayouts
from ..ports import attach
from ..batches import sample, collate as batch_collate

_loaded = {}
_resident = {}


def register_resident(pack):
    for i, row in enumerate(pack.rows): _resident[id(row)] = (pack, i)


def prepare(item):
    case = item['case']
    geometry = sample(case.solver_inputs, torch.as_tensor(item['positions']).float())
    case.reference_positions = public_hard_reference(
        case.solver_inputs[1], case.solver_inputs[5], case.solver_inputs[6])
    return {'graph': geometry, 'member': item['member'], 'case': case}


def collate(rows, device):
    if str(device).startswith('cuda') and rows and id(rows[0]) in _resident:
        pack = _resident[id(rows[0])][0]
        entries = [_resident[id(row)] for row in rows]
        if any(p is not pack for p, _ in entries): raise ValueError('Cannot mix resident data roles')
        return pack.batch([i for _, i in entries]), None
    g=batch_collate([row['graph'] for row in rows],list(range(len(rows))),device)
    b,n=g['valid'].shape;p=max(1,max(r['graph']['pin_weights'].shape[1] for r in rows))
    for key,tail,dtype in (('pin_coordinates',(p,2),torch.float32),('pin_weights',(p,),torch.float32),('pin_valid',(p,),torch.bool),('wire_b2b',(n,),torch.float32)):
        value=torch.zeros((b,n,*tail),dtype=dtype)
        for i,r in enumerate(rows):
            x=r['graph'][key];nn=r['graph']['n']
            if key=='wire_b2b':value[i,:nn,:nn]=x
            else:value[i,:nn,:x.shape[1]]=x
        g[key]=value.to(device)
    return g,None


def load_rows(role, indices=None, *, use_cache=True):
    dataset=ReferenceLayouts(role)
    selected=list(range(len(dataset))) if indices is None else list(indices)
    if use_cache:
        from ..cache import load_rows as cached
        return [attach(r) for r in cached(role,selected)]
    return [attach(prepare(dataset[i])) for i in selected]
