"""Public-group soft spanning-tree contact and public weighted b2b excess."""
import torch
from .batches import decode

CONTACT_SCALE=.01
TANGENTIAL_FRACTION=.05
RIDGE=1e-4


def public_group_contact(rectangles,g):
    """Matrix-tree log connectivity; normal touching + tangential overlap.

    Group membership is public. A group need only form a connected graph,
    unlike supervising every pair as a contact. Inverse-quadratic affinity
    avoids exp underflow and retains a gradient across separated components.
    Cholesky_ex avoids host synchronization in CUDA graph capture.
    """
    p=rectangles;valid=g['valid'];n=p.shape[1]
    eye=torch.eye(n,device=p.device,dtype=p.dtype)[None]
    same=(g['group']>0).to(p.dtype)
    grouped=(g['constraints'][...,3]>0)&valid
    within=(same+eye*grouped[...,None]).clamp_max(1)
    within=within*valid[:,:,None]*valid[:,None,:]
    counts=within.sum(-1).clamp_min(1)
    pair=same*valid[:,:,None]*valid[:,None,:]
    centers=p[...,:2]+p[...,2:]/2
    delta=(centers[:,:,None]-centers[:,None,:]).abs()
    extent=(p[:,:,None,2:]+p[:,None,:,2:])/2
    local=g['area'].clamp_min(1e-12).sqrt()
    scale=((local[:,:,None]+local[:,None,:])/2).clamp_min(1e-6)
    gap=(delta-extent)/scale[...,None]
    min_dim=torch.minimum(p[:,:,None,2:],p[:,None,:,2:])/scale[...,None]
    tangent_margin=TANGENTIAL_FRACTION*min_dim
    vertical=gap[...,0].square()+(gap[...,1]+tangent_margin[...,1]).clamp_min(0).square()
    horizontal=gap[...,1].square()+(gap[...,0]+tangent_margin[...,0]).clamp_min(0).square()
    cost=torch.minimum(vertical,horizontal)
    weight=pair/(1+cost/CONTACT_SCALE)/counts[:,:,None]
    laplacian=torch.diag_embed(weight.sum(-1))-weight
    nullspace=within/counts[:,:,None]
    inactive=(~grouped).to(p.dtype)
    matrix=(laplacian+nullspace+torch.diag_embed(inactive)+RIDGE*eye)/(1+RIDGE)
    factor,info=torch.linalg.cholesky_ex(matrix,check_errors=False)
    logdet=2*factor.diagonal(dim1=-2,dim2=-1).clamp_min(1e-30).log().sum(-1)
    freedom=(grouped.to(p.dtype).sum(-1)-(grouped/counts).sum(-1)).clamp_min(1)
    value=-logdet/freedom
    # A failed factorization must propagate NaN to the existing pre-Adam guard.
    return torch.where(info==0,value,torch.full_like(value,float('nan')))


def b2b_excess(rectangles,g):
    # features[13] is public log1p(weighted b2b degree). Recover exact
    # weighted adjacency up to its original FP32 normalization roundoff.
    degree=g['features'][...,13].expm1().clamp_min(0)
    weights=g['adjacency']*degree[...,None]
    weights=(weights+weights.transpose(1,2))/2
    weights=weights*g['valid'][:,:,None]*g['valid'][:,None,:]
    centers=rectangles[...,:2]+rectangles[...,2:]/2
    ref=g['reference'];reference=ref[...,:2]+ref[...,2:]/2
    length=((centers[:,:,None]-centers[:,None,:]).abs().sum(-1)*weights).sum((1,2))/2
    target=((reference[:,:,None]-reference[:,None,:]).abs().sum(-1)*weights).sum((1,2))/2
    ratio=length.clamp_min(1e-8)/target.clamp_min(1e-8)
    loss=ratio.log().clamp_min(0).square()
    return loss*(target>1e-8),ratio


def terms(state,g):
    p=decode(state,g)
    group=public_group_contact(p,g)
    wire,ratio=b2b_excess(p,g)
    return group,wire,ratio
