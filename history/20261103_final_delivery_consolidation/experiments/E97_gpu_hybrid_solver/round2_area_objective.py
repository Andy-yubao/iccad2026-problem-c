"""Release teacher-axis group topology; preserve axis-free group nonoverlap."""
import torch
import torch.nn.functional as F
from .round2_area_freedom import decode
from .round2_area_freedom import decode as decode85
from .round2_connectivity import Packer,attach,network_inputs,mst_mask,public_denominator,_BITS

def base_per_case(state,g,target,usable):
    mask=g['mask']*g['valid'][...,None];local=g['area'].clamp_min(1e-12).sqrt()
    center=torch.nn.functional.smooth_l1_loss(state[...,:2]/local[...,None],target[...,:2]/local[...,None],reduction='none',beta=.1)
    center=(center*mask[...,:2]).sum((1,2))/mask[...,:2].sum((1,2)).clamp_min(1)
    shape=((state[...,2]-target[...,2]).square()*mask[...,2]).sum(1)/mask[...,2].sum(1).clamp_min(1)
    delta=state[:,:,:2][:,:,None]-state[:,:,:2][:,None,:];wanted=target[:,:,:2][:,:,None]-target[:,:,:2][:,None,:]
    scale=(local[:,:,None]+local[:,None,:])/2;n=state.shape[1];ix=torch.arange(n,device=state.device)
    pairs=g['valid'][:,:,None]&g['valid'][:,None,:]&(ix[:,None]!=ix[None,:])[None]
    pairs=pairs&(mask[...,0].bool()[:,:,None]|mask[...,0].bool()[:,None,:])
    # Exact teacher dimensions are supervision only, retaining area-tolerance
    # geometry for near-pair/teacher-axis selection; NN inputs remain public.
    wh=g['teacher_wh']
    gap=wanted.abs()-(wh[:,:,None]+wh[:,None,:])/2
    near=(gap.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)&pairs
    grouping=g['constraints'][...,3]
    same_group=(grouping[:,:,None]==grouping[:,None,:])&(grouping[:,:,None]>0)&pairs
    retained_near=near&~same_group
    error=torch.nn.functional.smooth_l1_loss(delta/scale.clamp_min(1e-8)[...,None],wanted/scale.clamp_min(1e-8)[...,None],reduction='none',beta=.1).mean(-1)
    relative=(error*retained_near).sum((1,2))/retained_near.sum((1,2)).clamp_min(1)
    predicted=decode85(state,g)
    teacher_center=target[...,:2]
    teacher_delta=teacher_center[:,:,None]-teacher_center[:,None,:]
    extent=(wh[:,:,None]+wh[:,None,:])/2
    score=teacher_delta.abs()/extent.clamp_min(1e-8)-1
    axis_x=score[...,0]>=score[...,1]
    teacher_chosen=torch.where(axis_x,teacher_delta[...,0],teacher_delta[...,1])
    direction=torch.where(teacher_chosen>=0,1.,-1.)
    predicted_center=predicted[...,:2]+predicted[...,2:]/2
    predicted_delta=predicted_center[:,:,None]-predicted_center[:,None,:]
    predicted_extent=(predicted[...,2:][:,:,None]+predicted[...,2:][:,None,:])/2
    native=(predicted_delta/predicted_extent.clamp_min(1e-8)).clamp(-8,8)
    signed=direction*torch.where(axis_x,native[...,0],native[...,1])
    teacher_clearance=torch.where(retained_near,torch.relu(1-signed).square(),0.).amax((1,2))
    group_clearance=torch.where(same_group,torch.relu(1-native.abs().amax(-1)).square(),0.).amax((1,2))
    clearance=torch.maximum(teacher_clearance,group_clearance)
    return dict(teacher_center=center,teacher_shape=shape,teacher_relative=relative,teacher_clearance_tail=clearance,total=usable*(center+.25*shape+.25*relative+clearance))

def geometric_terms(state,g,target,usable):
    out=base_per_case(state,g,target,usable);p=decode(state,g);valid=g['valid'];local=g['area'].clamp_min(1e-12).sqrt();c=g['constraints']
    lo=p[...,:2];hi=lo+p[...,2:];minimum=lo.masked_fill(~valid[...,None],1e4).amin(1);maximum=hi.masked_fill(~valid[...,None],-1e4).amax(1)
    side=torch.stack((lo[...,0]-minimum[:,None,0],maximum[:,None,0]-hi[...,0],maximum[:,None,1]-hi[...,1],lo[...,1]-minimum[:,None,1]),-1)/local[...,None]
    if state.device not in _BITS:_BITS[state.device]=torch.tensor([1,2,4,8],device=state.device)
    bits=_BITS[state.device];required=((c[...,4,None]&bits)!=0)&valid[...,None]
    boundary=torch.where(required,F.smooth_l1_loss(side,torch.zeros_like(side),beta=.1,reduction='none'),0.).amax(-1).sum(-1)
    center=lo+p[...,2:]/2;delta=(center[:,:,None]-center[:,None,:]).abs();extent=(p[...,2:][:,:,None]+p[...,2:][:,None,:])/2
    normal=(delta/extent.clamp_min(1e-8)).clamp(0,8)
    touching=F.smooth_l1_loss(normal,torch.ones_like(normal),beta=.1,reduction='none')
    pair_local=(local[:,:,None]+local[:,None,:])/2
    tangent=torch.relu(.01-(extent-delta)/pair_local.clamp_min(1e-8)[...,None]).square()
    cost=torch.minimum(touching[...,0]+tangent[...,1],touching[...,1]+tangent[...,0])
    tree=mst_mask(cost.detach(),c[...,3],valid)
    denom=public_denominator(c,valid)
    out['public_boundary']=boundary/denom;out['public_group_contact']=(cost*tree).sum((1,2))/denom
    out['public_soft']=out['public_boundary']+out['public_group_contact']
    return out


def raw_terms(state,g,target,usable):
    out=geometric_terms(state,g,target,usable)
    free=g['mask'][...,2]*g['valid'];wanted=(g['teacher_wh'][...,0]*g['teacher_wh'][...,1]/g['area'].clamp_min(1e-12)-1)/.01
    fraction=state[...,3] if state.shape[-1]==4 else torch.zeros_like(state[...,0])
    area=F.smooth_l1_loss(fraction/.01,wanted,beta=.1,reduction='none')
    out['teacher_area']=usable*(area*free).sum(-1)/free.sum(-1).clamp_min(1)
    out['total']=out['total']+.25*out['teacher_area']
    return out
