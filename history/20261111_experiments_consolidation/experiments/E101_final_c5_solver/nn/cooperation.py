"""Bounded joint descent of actual pair intersection area, public geometry only.
Eight simultaneous steps use the analytic rectangle-intersection center gradient.
Four predetermined backtracking scales and a per-layout no-increase check prevent
surrogate penetration forces from worsening true area. Boolean branch choices
have no straight-through estimator; accepted coordinates retain normal autograd.
Finite bounded descent does not certify legality or solve containment plateaus.
"""
import torch
from .base_model import decode
STEPS=8
BACKTRACKS=4

def spatial_centers(source,proposed,g,frame,max_correction=.15,*,steps=STEPS):
    p=decode(source,g);center=p[...,:2]+p[...,2:]/2;wh=p[...,2:];valid=g['valid']
    mobility=g['mask'][...,:2]*valid[...,None];local=g['area'].clamp_min(1e-12).sqrt()
    lower=torch.maximum(frame[0][:,None]+wh/2,center-max_correction*local[...,None])
    upper=torch.minimum(frame[1][:,None]-wh/2,center+max_correction*local[...,None])
    def bound(x):
        x=torch.minimum(torch.maximum(x,lower),upper)
        return torch.where(mobility.bool(),x,g['known'][...,:2])*valid[...,None]
    n=source.shape[1];ids=torch.arange(n,device=source.device)
    pairs=valid[:,:,None]&valid[:,None,:]&(ids[:,None]!=ids[None,:])[None]
    def intersection(x):
        low=x-wh/2;high=x+wh/2
        overlap=(torch.minimum(high[:,:,None],high[:,None,:])-torch.maximum(low[:,:,None],low[:,None,:])).clamp_min(0)
        cost=(overlap[...,0]*overlap[...,1]*pairs).sum((1,2))/2
        return cost,overlap,low,high
    current=bound(proposed)
    for _ in range(steps):
        cost,overlap,low,high=intersection(current)
        collision=(overlap[...,0]>0)&(overlap[...,1]>0)&pairs
        # Partial derivative of min(high_i,high_j)-max(low_i,low_j).
        # Exact ties split each min/max derivative equally, like torch.minimum.
        upper_grad=(high[:,:,None]<high[:,None,:]).to(source.dtype)+.5*(high[:,:,None]==high[:,None,:]).to(source.dtype)
        lower_grad=(low[:,:,None]>low[:,None,:]).to(source.dtype)+.5*(low[:,:,None]==low[:,None,:]).to(source.dtype)
        other_axis=torch.stack((overlap[...,1],overlap[...,0]),-1)
        gradient=((upper_grad-lower_grad)*other_axis*collision[...,None]).sum(2)
        # Count all potential interacting blocks, not only current collisions:
        # a move can create contacts with previously nonintersecting neighbours.
        degree=valid.sum(-1).clamp_min(1).to(source.dtype)[:,None,None]
        direction=-gradient*mobility/degree
        best=current;best_cost=cost
        for backtrack in range(BACKTRACKS):
            trial=bound(current+direction/(2**backtrack))
            trial_cost=intersection(trial)[0]
            improve=trial_cost<best_cost
            best=torch.where(improve[:,None,None],trial,best)
            best_cost=torch.where(improve,trial_cost,best_cost)
        current=best
    return current
