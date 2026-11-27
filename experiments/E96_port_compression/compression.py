"""Public-only, fixed-size compression. Full original geometry remains in Loss."""
import torch

MODES=('centroid','regional32')

def network_view(g,mode):
    if mode not in MODES:raise ValueError('Unknown registered port representation')
    if any(k in g for k in ('reference','target','labels')):raise ValueError('Labels forbidden')
    result=dict(g)
    if mode=='centroid':
        # The existing17 features contain per-module total port weight and mean.
        result.update(port_xy=g['port_xy'][:,:0],port_valid=g['port_valid'][:,:0],p2b_weights=g['p2b_weights'][:,:,:0])
        return result
    xy=g['port_xy'];pw=g['p2b_weights'];mass=pw.sum(1)
    if xy.shape[1]==0:
        result.update(port_xy=xy.new_zeros(xy.shape[0],32,2),
            port_valid=torch.zeros(xy.shape[0],32,dtype=torch.bool,device=xy.device),
            p2b_weights=pw.new_zeros(pw.shape[0],pw.shape[1],32))
        return result
    selected=g['port_valid']&(mass>0)
    has=selected.any(1)
    lo=xy.masked_fill(~selected[...,None],float('inf')).amin(1)
    hi=xy.masked_fill(~selected[...,None],-float('inf')).amax(1)
    lo=torch.where(has[:,None],lo,torch.zeros_like(lo));hi=torch.where(has[:,None],hi,lo)
    relative=(xy-lo[:,None])/(hi-lo).clamp_min(1e-12)[:,None]
    ix=(relative[...,0]*8).floor().long().clamp(0,7)
    iy=(relative[...,1]*4).floor().long().clamp(0,3)
    index=iy*8+ix
    weights=mass*selected
    # Deterministic scatter_add performs a host-dependent path in this CUDA
    # runtime. A fixed membership matrix gives the same sums and is capturable.
    membership=(index[...,None]==torch.arange(32,device=xy.device)[None,None]).to(xy.dtype)
    total=torch.bmm(weights[:,None],membership).squeeze(1)
    first=torch.bmm((xy*weights[...,None]).transpose(1,2),membership).transpose(1,2)
    centers=first/total.clamp_min(1e-12)[...,None]
    edges=torch.bmm(pw*selected[:,None],membership)
    result.update(port_xy=centers,port_valid=total>0,p2b_weights=edges)
    return result
