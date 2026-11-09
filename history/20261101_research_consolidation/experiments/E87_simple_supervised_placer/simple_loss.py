"""Three fixed groups: primary reference supervision, per-block collisions, public task."""
import torch
import torch.nn.functional as F
from .batches import decode
from .group_tree import minimum_group_tree

PUBLIC=('area','mask','valid','specs','constraints','group','mib','pin_coordinates','pin_weights','pin_valid','wire_b2b')
_BITS={}

def average(value,mask,dims):return (value*mask).sum(dims)/mask.sum(dims).clamp_min(1)

def contact_cost(p,scale):
    # Checked E70 primitive: corner-only contact is not abutment.
    lo=p[...,:2];hi=lo+p[...,2:]
    signed=torch.minimum(hi[:,:,None,:],hi[:,None,:,:])-torch.maximum(lo[:,:,None,:],lo[:,None,:,:])
    fx=torch.minimum((hi[:,:,None,0]-lo[:,None,:,0]).square(),(hi[:,None,:,0]-lo[:,:,None,0]).square())
    fy=torch.minimum((hi[:,:,None,1]-lo[:,None,:,1]).square(),(hi[:,None,:,1]-lo[:,:,None,1]).square())
    ty=(.01*torch.minimum(p[:,:,None,3],p[:,None,:,3])-signed[...,1]).clamp_min(0).square()
    tx=(.01*torch.minimum(p[:,:,None,2],p[:,None,:,2])-signed[...,0]).clamp_min(0).square()
    return torch.minimum((fx+ty)/scale.clamp_min(1e-6).square(),(fy+tx)/scale.clamp_min(1e-6).square())

def public_components(state,g):
    if set(g)-set(PUBLIC):raise ValueError('Public task API rejects labels/extra fields')
    valid=g['valid'];n=valid.shape[1];count=valid.sum(-1).clamp_min(1)
    area=g['area'].clamp_min(1e-12);local=area.sqrt()
    p=decode(state,g);lo=p[...,:2];hi=lo+p[...,2:];center=(lo+hi)/2
    delta=center[:,:,None,:]-center[:,None,:,:];ps=(local[:,:,None]+local[:,None,:])/2
    ids=torch.arange(n,device=state.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
    upper=pairs&(ids[:,None]<ids[None,:])[None]
    slack=(p[:,:,None,2:]+p[:,None,:,2:])/2-delta.abs()
    depth=torch.minimum(slack[...,0],slack[...,1]).clamp_min(0)
    penetration=((depth/ps.clamp_min(1e-6)).square()*upper).sum((1,2))/count
    intersection=(torch.minimum(hi[:,:,None,:],hi[:,None,:,:])-torch.maximum(lo[:,:,None,:],lo[:,None,:,:])).clamp_min(0)
    pair_area=intersection[...,0]*intersection[...,1]*pairs
    load=pair_area.sum(-1)/area*valid;load_mean=load.sum(-1)/count
    excess=(load-.10).clamp_min(0).masked_fill(~valid,0).amax(-1)
    overlap=penetration+load_mean+excess
    total_area=(area*valid).sum(-1).clamp_min(1e-12)
    wire=g['wire_b2b']*valid[:,:,None]*valid[:,None,:]
    pw=g['pin_weights']*g['pin_valid']*valid[:,:,None]
    wl=(wire*delta.abs().sum(-1)).sum((1,2))+(pw*(g['pin_coordinates']-center[:,:,None,:]).abs().sum(-1)).sum((1,2))
    weight=wire.sum((1,2))+pw.sum((1,2));wirelength=wl/(total_area.sqrt()*weight.clamp_min(1e-12))
    bblo=lo.masked_fill(~valid[...,None],float('inf')).amin(1);bbhi=hi.masked_fill(~valid[...,None],-float('inf')).amax(1)
    extent=bbhi-bblo;compactness=(extent[:,0]*extent[:,1]/total_area-1).clamp_min(0)
    peers=((g['group']>0)&pairs).sum(-1).to(state.dtype);ng=(peers/(peers+1)).sum(-1)
    grouping=minimum_group_tree(contact_cost(p,ps),g['group'],valid)
    sides=torch.stack((lo[...,0]-bblo[:,None,0],bbhi[:,None,0]-hi[...,0],bbhi[:,None,1]-hi[...,1],lo[...,1]-bblo[:,None,1]),-1)
    key=(str(state.device),g['constraints'].dtype)
    if key not in _BITS:_BITS[key]=torch.tensor((1,2,4,8),device=state.device,dtype=g['constraints'].dtype)
    required=((g['constraints'][...,4,None]&_BITS[key])!=0)&valid[...,None]
    edge=F.smooth_l1_loss(sides/local[...,None],torch.zeros_like(sides),reduction='none',beta=.1)
    nb=required.any(-1).sum(-1);bsum=((edge*required).sum(-1)/required.sum(-1).clamp_min(1)).sum(-1)
    same=(g['mib']>0)&pairs;mpeer=same.sum(-1).to(state.dtype);mw=mpeer/(mpeer+1)*valid
    logs=p[...,2:].clamp_min(1e-12).log()
    mean=((logs[:,:,None,:]*same[:,:,:,None]).sum(1)+logs)/(mpeer+1)[...,None]
    msum=((logs-mean).square().mean(-1)*mw).sum(-1);nm=mw.sum(-1)
    soft=(bsum+grouping*ng+msum)/(nb+ng+nm).clamp_min(1)
    return dict(penetration=penetration,module_load_mean=load_mean,module_max_excess10=excess,
        overlap_total=overlap,wirelength=wirelength,compactness=compactness,grouping=grouping,
        boundary=bsum/nb.clamp_min(1),mib=msum/nm.clamp_min(1),soft=soft,
        task=.5*(wirelength+compactness)+2*soft,
        pair_intersection_fraction=.5*pair_area.sum((1,2))/total_area,module_load_max=load.amax(-1))

def components(state,g,coefficients):
    out=public_components(state,{k:g[k] for k in PUBLIC});mask=g['mask']*g['valid'][...,None]
    local=g['area'].clamp_min(1e-12).sqrt()
    out['position']=average(F.smooth_l1_loss(state[...,:2]/local[...,None],g['target'][...,:2]/local[...,None],reduction='none',beta=.1),mask[...,:2],(1,2))
    out['shape']=average((state[...,2]-g['target'][...,2]).square(),mask[...,2],(1,))
    out['reference']=out['position']+.25*out['shape'];out['supervision']=out['reference']
    out['total']=out['reference']+coefficients['overlap']*out['overlap_total']+coefficients['task']*out['task']
    return out
