"""Differentiable containment inside a frozen public-predicted frame."""
import torch
from .base_model import decode


def predicted_frame(base_state, g):
    rectangles=decode(base_state,g)
    low=rectangles[...,:2].masked_fill(~g['valid'][...,None],float('inf')).amin(1)
    high=(rectangles[...,:2]+rectangles[...,2:]).masked_fill(~g['valid'][...,None],-float('inf')).amax(1)
    return low, high


def contain(state, g, frame):
    low,high=frame
    rectangles=decode(state,g)
    physical_centers=rectangles[...,:2]+rectangles[...,2:]/2
    size=(high-low).clamp_min(1e-12)
    area=g['area'].clamp_min(1e-12)
    # Area-preserving free shapes must fit both frame dimensions. Every
    # source rectangle fits its own bounding frame, hence this interval is
    # feasible for valid free shapes (apart from FP32 roundoff).
    lower=(area/size[:,None,1].square()).log().clamp_min(-5.)
    upper=(size[:,None,0].square()/area).log().clamp_max(5.)
    ratio=torch.minimum(torch.maximum(state[...,2].clamp(-5,5),lower),upper)
    wh=torch.stack(((area*ratio.exp()).sqrt(),(area/ratio.exp()).sqrt()),-1)
    wh=torch.where((~g['mask'][...,2].bool())[...,None],g['specs'][...,2:],wh)
    center_low=low[:,None]+wh/2
    center_high=high[:,None]-wh/2
    centers=torch.minimum(torch.maximum(physical_centers,center_low),center_high)
    canonical=torch.cat((centers,ratio[...,None]),-1)
    return torch.where(g['mask'].bool(),canonical,g['known'])*g['valid'][...,None]
