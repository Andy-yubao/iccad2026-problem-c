"""Detached hypothetical equal-area shape responses; never execute an action."""
import torch

DELTAS=(.2,-.2,.8,-.8)
METRICS=('overlap','group_contact','boundary','bbox')
FEATURE_NAMES=tuple(f'shape_{d:+.1f}_{metric}' for d in DELTAS for metric in METRICS)
_BITS={}

@torch.no_grad()
def shape_features(p,g):
    p=p.detach();lo=p[...,:2];wh=p[...,2:];hi=lo+wh;center=lo+wh/2
    valid=g['valid'];area=g['area'].clamp_min(1e-12);local=area.sqrt()
    n=p.shape[1];ids=torch.arange(n,device=p.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
    same=(g['group']>0)&pairs;has=same.any(-1)
    scale=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
    # Other-rectangle extrema, excluding the moving module. A lone module uses itself.
    other_lo=lo[:,None].expand(-1,n,-1,-1).masked_fill(~pairs[...,None],float('inf')).amin(2)
    other_hi=hi[:,None].expand(-1,n,-1,-1).masked_fill(~pairs[...,None],-float('inf')).amax(2)
    key=(str(p.device),g['constraints'].dtype)
    if key not in _BITS:_BITS[key]=torch.tensor((1,2,4,8),device=p.device,dtype=g['constraints'].dtype)
    bits=_BITS[key]
    required=((g['constraints'][...,4,None]&bits)!=0)&valid[...,None]
    total=(area*valid).sum(-1).clamp_min(1e-12)

    mib_peers=(g['mib']>0)
    def measurements(qlo,qwh,linked=False):
        qhi=qlo+qwh
        # Query i changes its entire MIB class together, other classes stay fixed.
        olo=torch.where(mib_peers[...,None],qlo[:,None],lo[:,None]) if linked else lo[:,None].expand(-1,n,-1,-1)
        ohi=torch.where(mib_peers[...,None],qhi[:,None],hi[:,None]) if linked else hi[:,None].expand(-1,n,-1,-1)
        owh=ohi-olo
        signed=torch.minimum(qhi[:,:,None],ohi)-torch.maximum(qlo[:,:,None],olo)
        load=(signed.clamp_min(0).prod(-1)*pairs).sum(-1)/area
        fx=torch.minimum((qhi[:,:,None,0]-olo[...,0]).square(),(ohi[...,0]-qlo[:,:,None,0]).square())
        fy=torch.minimum((qhi[:,:,None,1]-olo[...,1]).square(),(ohi[...,1]-qlo[:,:,None,1]).square())
        ty=(.01*torch.minimum(qwh[:,:,None,1],owh[...,1])-signed[...,1]).clamp_min(0).square()
        tx=(.01*torch.minimum(qwh[:,:,None,0],owh[...,0])-signed[...,0]).clamp_min(0).square()
        contact=torch.minimum((fx+ty)/scale.square(),(fy+tx)/scale.square())
        contact=contact.masked_fill(~same,float('inf')).amin(-1)
        contact=torch.where(has,contact,torch.zeros_like(contact))
        active_lo=olo.masked_fill(~pairs[...,None],float('inf')).amin(2) if linked else other_lo
        active_hi=ohi.masked_fill(~pairs[...,None],-float('inf')).amax(2) if linked else other_hi
        bblo=torch.minimum(qlo,active_lo);bbhi=torch.maximum(qhi,active_hi)
        sides=torch.stack((qlo[...,0]-bblo[...,0],bbhi[...,0]-qhi[...,0],bbhi[...,1]-qhi[...,1],qlo[...,1]-bblo[...,1]),-1)/local[...,None]
        boundary=(sides*required).sum(-1)/required.sum(-1).clamp_min(1)
        bbox=(bbhi-bblo).prod(-1)/total[:,None]
        return torch.stack((load.log1p(),contact,boundary,bbox),-1)

    base=measurements(lo,wh);ratio=(wh[...,0]/wh[...,1].clamp_min(1e-12)).log()
    mib_weights=mib_peers.to(p.dtype)+torch.eye(n,device=p.device)[None]
    mib_weights=mib_weights*valid[:,None,:];mib_weights=mib_weights/mib_weights.sum(-1,keepdim=True).clamp_min(1)
    ratio=(mib_weights@ratio[...,None]).squeeze(-1)
    responses=[]
    for delta in DELTAS:
        r=(ratio+delta).clamp(-5,5)
        trial_wh=torch.stack(((area*r.exp()).clamp_min(1e-12).sqrt(),(area/r.exp()).clamp_min(1e-12).sqrt()),-1)
        # Same nonnegative-origin rule as the actual decoder. No F/P overwrite.
        trial_lo=(center-trial_wh/2).clamp_min(0)
        responses.append((measurements(trial_lo,trial_wh,linked=True)-base).asinh())
    return torch.cat(responses,-1)*valid[...,None]
