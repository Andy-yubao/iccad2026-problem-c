"""Original fixed C3 loss and decoder, copied as independent offline analysis.
Labels are reporting targets only, never solver inputs.
"""
import torch

def decode(state, g):
    ratio = state[..., 2].clamp(-5, 5)
    wh = torch.stack(((g['area'] * ratio.exp()).clamp_min(1e-12).sqrt(),
                      (g['area'] / ratio.exp()).clamp_min(1e-12).sqrt()), -1)
    wh = torch.where((~g['mask'][..., 2].bool())[..., None], g['specs'][..., 2:], wh)
    xy = (state[..., :2] - wh / 2).clamp_min(0)
    return torch.where((~g['mask'][..., 0].bool())[..., None], g['specs'], torch.cat((xy, wh), -1))

decode85 = decode

def per_case(state,g,target,usable):
    mask=g['mask']*g['valid'][...,None];local=g['area'].clamp_min(1e-12).sqrt()
    center=torch.nn.functional.smooth_l1_loss(state[...,:2]/local[...,None],target[...,:2]/local[...,None],reduction='none',beta=.1)
    center=(center*mask[...,:2]).sum((1,2))/mask[...,:2].sum((1,2)).clamp_min(1)
    shape=((state[...,2]-target[...,2]).square()*mask[...,2]).sum(1)/mask[...,2].sum(1).clamp_min(1)
    delta=state[:,:,:2][:,:,None]-state[:,:,:2][:,None,:];wanted=target[:,:,:2][:,:,None]-target[:,:,:2][:,None,:]
    scale=(local[:,:,None]+local[:,None,:])/2;n=state.shape[1];ix=torch.arange(n,device=state.device)
    pairs=g['valid'][:,:,None]&g['valid'][:,None,:]&(ix[:,None]!=ix[None,:])[None]
    pairs=pairs&(mask[...,0].bool()[:,:,None]|mask[...,0].bool()[:,None,:])
    # Labels contain center/log-aspect; near geometry uses public nominal area
    # and locked WH, rather than an unrecorded teacher area-tolerance variable.
    wh=decode85(target,g)[...,2:]
    gap=wanted.abs()-(wh[:,:,None]+wh[:,None,:])/2
    near=(gap.clamp_min(0).square().sum(-1).sqrt()<=.25*scale)&pairs
    error=torch.nn.functional.smooth_l1_loss(delta/scale.clamp_min(1e-8)[...,None],wanted/scale.clamp_min(1e-8)[...,None],reduction='none',beta=.1).mean(-1)
    relative=(error*near).sum((1,2))/near.sum((1,2)).clamp_min(1)
    predicted=decode85(state,g);teacher=decode85(target,g)
    teacher_center=teacher[...,:2]+teacher[...,2:]/2
    teacher_delta=teacher_center[:,:,None]-teacher_center[:,None,:]
    extent=(teacher[...,2:][:,:,None]+teacher[...,2:][:,None,:])/2
    score=teacher_delta.abs()/extent.clamp_min(1e-8)-1
    axis_x=score[...,0]>=score[...,1]
    teacher_chosen=torch.where(axis_x,teacher_delta[...,0],teacher_delta[...,1])
    direction=torch.where(teacher_chosen>=0,1.,-1.)
    predicted_center=predicted[...,:2]+predicted[...,2:]/2
    predicted_delta=predicted_center[:,:,None]-predicted_center[:,None,:]
    predicted_extent=(predicted[...,2:][:,:,None]+predicted[...,2:][:,None,:])/2
    native=(predicted_delta/predicted_extent.clamp_min(1e-8)).clamp(-8,8)
    signed=direction*torch.where(axis_x,native[...,0],native[...,1])
    clearance=torch.where(near,torch.relu(1-signed).square(),0.).amax((1,2))
    return dict(teacher_center=center,teacher_shape=shape,teacher_relative=relative,teacher_clearance_tail=clearance,total=usable*(center+.25*shape+.25*relative+clearance))
