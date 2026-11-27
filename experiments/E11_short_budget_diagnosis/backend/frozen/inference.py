"""Public-only features match padded training preparation, including dimensions."""
import torch
from .data import graph_features


def prepare_inference(inputs,device='cuda'):
    g=graph_features(inputs);n=int(inputs[0]);s=g['scale'];locked=g['locked']
    known=torch.cat(((g['specs'][:,:2]+g['specs'][:,2:]/2)/s,
                     (g['specs'][:,2].clamp_min(1e-8)/g['specs'][:,3].clamp_min(1e-8)).log()[:,None]),-1)*(1-g['mask'])
    public={'features':g['features'],'adjacency':g['adjacency'],'mib':g['relations'][0],
            'group':g['relations'][1],'mask':g['mask'],'known':known,
            'area':torch.where(locked,g['specs'][:,2:].prod(1),g['areas'])/s.square(),
            'specs':torch.where(locked[:,None],g['specs']/s,torch.zeros_like(g['specs'])),
            'constraints':torch.as_tensor(inputs[5]).long(),'valid':torch.ones(n,dtype=torch.bool),
            'scale':s}
    return {key:value[None].to(device) for key,value in public.items()}
