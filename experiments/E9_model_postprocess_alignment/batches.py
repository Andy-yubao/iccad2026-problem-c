"""Experiment-local preparation and padded batching; no test loader."""
import torch
from .data import graph_features,encode_target

PUBLIC_KEYS=('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid')


def inputs_only(batch):
    """Keep supervision outside the network's input contract."""
    return {key:batch[key] for key in PUBLIC_KEYS if key in batch}


def sample(inputs,reference):
    g=graph_features(inputs)
    n=int(inputs[0]);scale=g['scale']
    locked=g['locked'];area=torch.where(locked,g['specs'][:,2:].prod(1),g['areas'])/scale.square()
    s={'features':g['features'],'adjacency':g['adjacency'],'mib':g['relations'][0],
       'group':g['relations'][1],'mask':g['mask'],'area':area,
       'known':torch.cat(((g['specs'][:,:2]+g['specs'][:,2:]/2)/scale,
                          (g['specs'][:,2].clamp_min(1e-8)/g['specs'][:,3].clamp_min(1e-8)).log()[:,None]),1),
       'target':encode_target(reference,g),'specs':g['specs']/scale,
       'reference':reference/scale,'constraints':torch.as_tensor(inputs[5]).long(),'scale':scale,'n':n}
    # known values are zero outside public locked channels: labels never enter inputs.
    s['known']=s['known']*(1-s['mask'])
    s['specs']=torch.where(g['locked'][:,None],s['specs'],torch.zeros_like(s['specs']))
    # Public weighted edge lists for the training objective, separate from the
    # row-normalized attention adjacency. Pin averages cannot reproduce HPWL.
    s['wire_b2b']=torch.as_tensor(inputs[2]).float().clone()
    s['wire_p2b']=torch.as_tensor(inputs[3]).float().clone()
    s['pins']=torch.as_tensor(inputs[4]).float().clone()/scale
    return s


def collate(samples,indices,device):
    rows=[samples[i] for i in indices];b=len(rows);n=max(r['n'] for r in rows)
    out={}
    node=('features','mask','area','known','target','specs','reference','constraints')
    for key in node:
        t=rows[0][key]
        shape=(b,n)+t.shape[1:]
        out[key]=torch.zeros(shape,dtype=t.dtype)
        for i,r in enumerate(rows):out[key][i,:r['n']]=r[key]
    for key in ('adjacency','mib','group'):
        out[key]=torch.zeros(b,n,n)
        for i,r in enumerate(rows):out[key][i,:r['n'],:r['n']]=r[key]
    for key in ('wire_b2b','wire_p2b','pins'):
        size=max(1,max(len(r[key]) for r in rows))
        out[key]=torch.full((b,size,rows[0][key].shape[-1]),-1. if key!='pins' else 0.)
        for i,r in enumerate(rows):out[key][i,:len(r[key])]=r[key]
    out['valid']=torch.arange(n)[None,:]<torch.tensor([r['n'] for r in rows])[:,None]
    out['scale']=torch.stack([r['scale'] for r in rows])
    return {k:v.to(device) for k,v in out.items()}


def decode(state,g):
    r=state[...,2].clamp(-5,5)
    wh=torch.stack(((g['area']*r.exp()).clamp_min(1e-12).sqrt(),
                    (g['area']/r.exp()).clamp_min(1e-12).sqrt()),dim=-1)
    locked=~g['mask'][...,2].bool()
    wh=torch.where(locked[...,None],g['specs'][...,2:],wh)
    xy=(state[...,:2]-wh/2).clamp_min(0)
    out=torch.cat((xy,wh),dim=-1)
    return torch.where((~g['mask'][...,0].bool())[...,None],g['specs'],out)
