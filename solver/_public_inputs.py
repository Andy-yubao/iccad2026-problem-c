"""Public input mapping from the frozen E169 evaluation."""
import numpy as np
import torch

from ._model.features import graph_features
from ._model.ports import public_ports
from ._model.role_features import build as build_roles


def public_graph(inputs,pad=120):
    """Exactly E114 conditioning/decoding, without constructing any target tensor."""
    g=graph_features(inputs)
    n=int(inputs[0]);scale=g['scale'];spec=g['specs']
    constraints=torch.as_tensor(inputs[5]).long().clone();constraints[:,2]=0
    known=torch.cat(((spec[:,:2]+spec[:,2:]/2)/scale,
        (spec[:,2].clamp_min(1e-8)/spec[:,3].clamp_min(1e-8)).log()[:,None]),1)
    known=torch.cat((known[:,:2]*g['preplaced'][:,None],known[:,2:]*g['locked'][:,None]),1)
    data=dict(features=g['features'],adjacency=g['adjacency'],group=g['relations'][0],
        mask=g['mask'],area=g['areas']/scale.square(),known=known,
        specs=torch.where(g['locked'][:,None],spec/scale,torch.zeros_like(spec)),constraints=constraints)
    out={}
    for key,value in data.items():
        shape=(pad,pad) if key in ('adjacency','group') else (pad,*value.shape[1:])
        out[key]=torch.zeros((1,*shape),dtype=value.dtype)
        if key in ('adjacency','group'):out[key][0,:n,:n]=value
        else:out[key][0,:n]=value
    out['valid']=torch.arange(pad)[None,:]<n
    out['scale']=scale[None]
    # Match the resident public-role layout, with module slots preceding ports.
    ports = public_ports(inputs, scale)
    roles = build_roles(dict(ports, n=n, constraints=constraints))
    m = len(roles['port_features'])
    out['role_features'] = torch.zeros(1, pad, 16)
    out['role_features'][0, :n] = torch.from_numpy(roles['role_features'])
    out['port_features'] = torch.from_numpy(roles['port_features'])[None]
    out['port_valid'] = torch.ones(1, m, dtype=torch.bool)
    out['role_edges'] = torch.zeros(1, pad, pad + m)
    out['role_edges'][0, :n, :n] = ports['wire_b2b'] * 2
    out['role_edges'][0, :n, pad:] = torch.from_numpy(roles['port_edges'])
    out['role_group'] = torch.zeros_like(out['role_edges'])
    out['role_group'][0, :n, :n] = data['group']
    out['role_hops'] = torch.full((1, pad, pad + m), -1, dtype=torch.int16)
    hops = torch.from_numpy(roles['hops'])
    out['role_hops'][0, :n, :n] = hops[:, :n]
    out['role_hops'][0, :n, pad:] = hops[:, n:]
    return out

def repair_inputs(inputs):
    copied=tuple(int(v) if i==0 else np.asarray(v).copy() for i,v in enumerate(inputs))
    copied[5][:,2]=0
    return copied
