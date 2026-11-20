"""Exact x/y transposition, with no reference-derived input transform parameters."""
import numpy as np
import torch

def boundary_swap(codes):
    return (codes & ~15) | ((codes&1)<<3) | ((codes&2)<<1) | ((codes&4)>>1) | ((codes&8)>>3)

def flags_for(seed, epoch, indices):
    rng = np.random.default_rng(np.random.SeedSequence([20261104,int(seed),int(epoch)]))
    return (rng.random(32000)<.5)[indices]

def axis_swap(batch):
    flags = batch['axis_swap']
    out = dict(batch)
    def select(key, changed):
        value = batch[key]
        mask = flags.reshape((-1,)+(1,)*(value.ndim-1))
        out[key] = torch.where(mask,changed,value)
    def reorder(value, order):
        # Python-list advanced indexing creates host index tensors during capture.
        return torch.stack(tuple(value[...,i] for i in order),-1)
    select('features',reorder(batch['features'],(0,2,1,4,3,5,6,10,9,8,7,11,12,13,14,16,15)))
    for key in ('known','target'):
        value = batch[key]
        select(key,torch.stack((value[...,1],value[...,0],-value[...,2]),-1))
    for key in ('specs','reference'):
        select(key,reorder(batch[key],(1,0,3,2)))
    select('mask',reorder(batch['mask'],(1,0,2)))
    constraints = batch['constraints'].clone()
    constraints[...,4] = boundary_swap(constraints[...,4])
    select('constraints',constraints)
    select('pin_coordinates',reorder(batch['pin_coordinates'],(1,0)))
    return out
