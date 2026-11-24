"""Independent unlabelled preprocessing for preparation parity checks."""
import torch
from .features import graph_features
from .ports import public_ports
from .role_features import build
from .resident import ResidentRows
from .public import inputs_only

def public_batch(inputs,device='cuda'):
    g=graph_features(inputs);n=int(inputs[0]);scale=g['scale']
    constraints=torch.as_tensor(inputs[5]).long().clone()
    known=torch.cat(((g['specs'][:,:2]+g['specs'][:,2:]/2)/scale,
        (g['specs'][:,2].clamp_min(1e-8)/g['specs'][:,3].clamp_min(1e-8)).log()[:,None]),1)
    known=torch.cat((known[:,:2]*g['preplaced'][:,None],known[:,2:]*g['locked'][:,None]),1)
    graph=dict(features=g['features'],adjacency=g['adjacency'],mib=g['relations'][0],group=g['relations'][1],
        mask=g['mask'],area=g['areas']/scale.square(),known=known,
        specs=torch.where(g['locked'][:,None],g['specs']/scale,torch.zeros_like(g['specs'])),
        constraints=constraints,scale=scale,n=n,target=torch.zeros(n,3),reference=torch.ones(n,4))
    graph.update(public_ports(inputs,scale))
    row=dict(graph=graph,role=build(graph),member=('unlabelled',0))
    return inputs_only(ResidentRows([row],device,max_nodes=120).batch([0]))
