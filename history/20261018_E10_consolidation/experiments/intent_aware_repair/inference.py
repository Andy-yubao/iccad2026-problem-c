"""Strict public-input inference preparation; no reference argument or labels."""
import torch
from .data import graph_features


def prepare_inference(inputs,device='cuda'):
    base=graph_features(inputs,'cpu')
    known=torch.cat(((base['specs'][:,:2]+base['specs'][:,2:]/2)/base['scale'],
                     (base['specs'][:,2].clamp_min(1e-8)/base['specs'][:,3].clamp_min(1e-8)).log()[:,None]),1)
    known=known*(1-base['mask'])
    forward={'features':base['features'][None].to(device),
             'adjacency':base['adjacency'][None].to(device),
             'mib':base['relations'][0][None].to(device),'group':base['relations'][1][None].to(device),
             'mask':base['mask'][None].to(device),'known':known[None].to(device),
             'valid':torch.ones(1,len(known),device=device,dtype=torch.bool)}
    decoder={key:base[key].to(device) for key in ('areas','specs','locked','preplaced','scale')}
    return forward,decoder
