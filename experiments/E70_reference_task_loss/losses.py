"""Fixed reference position/shape plus eight strictly public task terms."""
import torch
import torch.nn.functional as F
from .batches import decode
from .group_tree import minimum_group_tree
WEIGHTS={'position':.25,'shape':.25,'penetration':1.,'intersection_area':1.,
         'actual_wirelength':.25,'compactness':.25,'group_connectivity':.5,
         'boundary':.25,'mib':.1,'parameter_domain':.1}
TASK_KEYS=tuple(k for k in WEIGHTS if k not in ('position','shape'))
PUBLIC_TASK_KEYS=('area','mask','valid','specs','constraints','group','mib',
                  'pin_coordinates','pin_weights','pin_valid','wire_b2b')
_BITS={}

def side_bits(device,dtype):
    key=(str(device),dtype)
    if key not in _BITS:_BITS[key]=torch.tensor((1,2,4,8),device=device,dtype=dtype)
    return _BITS[key]

def average(v,m,dims):return (v*m).sum(dims)/m.sum(dims).clamp_min(1)

def scales(g):
    local=g['area'].clamp_min(1e-12).sqrt();valid=g['valid']
    ordered=local.masked_fill(~valid,float('inf')).sort(-1).values
    count=valid.sum(-1).clamp_min(1)
    low=ordered.gather(1,((count-1)//2)[:,None]).squeeze(1)
    high=ordered.gather(1,(count//2)[:,None]).squeeze(1)
    return local,torch.maximum(local,.25*(low+high)[:,None])

def contact_cost(p,pair_scale):
    lo=p[...,:2];hi=lo+p[...,2:]
    signed=torch.minimum(hi[:,:,None,:],hi[:,None,:,:])-torch.maximum(lo[:,:,None,:],lo[:,None,:,:])
    fx=torch.minimum((hi[:,:,None,0]-lo[:,None,:,0]).square(),(hi[:,None,:,0]-lo[:,:,None,0]).square())
    fy=torch.minimum((hi[:,:,None,1]-lo[:,None,:,1]).square(),(hi[:,None,:,1]-lo[:,:,None,1]).square())
    ty=(.01*torch.minimum(p[:,:,None,3],p[:,None,:,3])-signed[...,1]).clamp_min(0).square()
    tx=(.01*torch.minimum(p[:,:,None,2],p[:,None,:,2])-signed[...,0]).clamp_min(0).square()
    return torch.minimum((fx+ty)/pair_scale.clamp_min(1e-6).square(),(fy+tx)/pair_scale.clamp_min(1e-6).square())

def public_components(state,g):
    if set(g)-set(PUBLIC_TASK_KEYS):raise ValueError('Task Loss rejects labels and extra fields')
    valid=g['valid'];b,n=valid.shape;local,t=scales(g)
    p=decode(state,g);center=p[...,:2]+p[...,2:]/2
    delta=center[:,:,None,:]-center[:,None,:,:];ps=(local[:,:,None]+local[:,None,:])/2
    ids=torch.arange(n,device=state.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]<ids[None,:])[None]
    count=valid.sum(1).clamp_min(1)
    slack=(p[:,:,None,2:]+p[:,None,:,2:])/2-delta.abs()
    depth=torch.minimum(slack[...,0],slack[...,1]).clamp_min(0)
    penetration=((depth/ps.clamp_min(1e-6)).square()*pairs).sum((1,2))/count
    lo=p[...,:2];hi=lo+p[...,2:]
    inter=(torch.minimum(hi[:,:,None,:],hi[:,None,:,:])-torch.maximum(lo[:,:,None,:],lo[:,None,:,:])).clamp_min(0)
    overlap=inter[...,0]*inter[...,1]*pairs
    area_total=(g['area']*valid).sum(-1).clamp_min(1e-12)
    amin=torch.minimum(g['area'][:,:,None],g['area'][:,None,:]).clamp_min(1e-12)
    area=.5*overlap.sum((1,2))/area_total+.5*(overlap/amin).sum((1,2))/count
    wire=g['wire_b2b']*valid[:,:,None]*valid[:,None,:]
    pw=g['pin_weights']*g['pin_valid']*valid[:,:,None]
    wl=(wire*delta.abs().sum(-1)).sum((1,2))+(pw*(g['pin_coordinates']-center[:,:,None,:]).abs().sum(-1)).sum((1,2))
    weight=wire.sum((1,2))+pw.sum((1,2));denom=torch.where(weight>0,weight,torch.ones_like(weight))
    wirelength=wl/(area_total.sqrt()*denom)
    bblo=lo.masked_fill(~valid[...,None],float('inf')).amin(1)
    bbhi=hi.masked_fill(~valid[...,None],-float('inf')).amax(1);extent=bbhi-bblo
    compactness=(extent[:,0]*extent[:,1]/area_total-1).clamp_min(0)
    group=minimum_group_tree(contact_cost(p,ps),g['group'],valid)
    sides=torch.stack((lo[...,0]-bblo[:,None,0],bbhi[:,None,0]-hi[...,0],bbhi[:,None,1]-hi[...,1],lo[...,1]-bblo[:,None,1]),-1)
    bits=side_bits(state.device,g['constraints'].dtype)
    required=((g['constraints'][...,4,None]&bits)!=0)&valid[...,None]
    edge=F.smooth_l1_loss(sides/t[...,None],torch.zeros_like(sides),reduction='none',beta=.1)
    boundary=average((edge*required).sum(-1)/required.sum(-1).clamp_min(1),required.any(-1),(1,))
    logs=p[...,2:].clamp_min(1e-12).log()
    peers=((g['mib']>0)&valid[:,:,None]&valid[:,None,:]).sum(-1).to(state.dtype)
    mean=((g['mib']@logs)*peers[...,None]+logs)/(peers+1)[...,None]
    mw=peers/(peers+1)*valid
    mib=((logs-mean).square().mean(-1)*mw).sum(-1)/mw.sum(-1).clamp_min(1)
    center_mask=g['mask'][...,:2]*valid[...,None];shape_mask=g['mask'][...,2]*valid
    raw_xy=state[...,:2]-p[...,2:]/2
    domain=average(((-raw_xy).clamp_min(0)/t[...,None]).square(),center_mask,(1,2))
    domain=domain+average((state[...,2].abs()-5).clamp_min(0).square(),shape_mask,(1,))
    return {'penetration':penetration,'intersection_area':area,'actual_wirelength':wirelength,
            'compactness':compactness,'group_connectivity':group,'boundary':boundary,'mib':mib,'parameter_domain':domain}

def reference_components(state,g,target):
    _,t=scales(g);valid=g['valid']
    pos=average(F.smooth_l1_loss(state[...,:2]/t[...,None],target[...,:2]/t[...,None],reduction='none',beta=.1),g['mask'][...,:2]*valid[...,None],(1,2))
    shape=average((state[...,2]-target[...,2]).square(),g['mask'][...,2]*valid,(1,))
    return {'position':pos,'shape':shape}

def components(state,g):
    public={k:g[k] for k in PUBLIC_TASK_KEYS}
    out=public_components(state,public);out.update(reference_components(state,public,g['target']))
    out['task_total']=sum(WEIGHTS[k]*out[k] for k in TASK_KEYS)
    out['supervision']=WEIGHTS['position']*out['position']+WEIGHTS['shape']*out['shape']
    out['total']=out['task_total']+out['supervision'];return out
