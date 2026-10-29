"""Positive reference-box expansion supervision; never an inference input."""
from .batches import decode
from .ops import product2

def expansion(state,g):
    p=decode(state,g);r=g['reference'];valid=g['valid'][...,None]
    lo=p[...,:2].masked_fill(~valid,1e4).amin(1)
    hi=(p[...,:2]+p[...,2:]).masked_fill(~valid,-1e4).amax(1)
    rlo=r[...,:2].masked_fill(~valid,1e4).amin(1)
    rhi=(r[...,:2]+r[...,2:]).masked_fill(~valid,-1e4).amax(1)
    ratio=product2(hi-lo).clamp_min(1e-8)/(rhi-rlo).prod(-1).clamp_min(1e-8)
    return ratio.log().clamp_min(0).square()
