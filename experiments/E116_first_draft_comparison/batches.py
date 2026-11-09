"""Experiment-local preparation and padded batching; no test loader."""
import torch
from .features import graph_features,encode_target

PUBLIC_KEYS=('features','adjacency','group','mask','area','known','specs','constraints','scale','valid')


def inputs_only(batch):
    """Keep supervision outside the network's input contract."""
    return {key:batch[key] for key in PUBLIC_KEYS if key in batch}


def sample(inputs,reference):
    g=graph_features(inputs)
    n=int(inputs[0]);scale=g['scale']
    area=g['areas']/scale.square()
    s={'features':g['features'],'adjacency':g['adjacency'],
       'group':g['relations'][0],'mask':g['mask'],'area':area,
       'known':torch.cat(((g['specs'][:,:2]+g['specs'][:,2:]/2)/scale,
                          (g['specs'][:,2].clamp_min(1e-8)/g['specs'][:,3].clamp_min(1e-8)).log()[:,None]),1),
       'target':encode_target(reference,g),'specs':g['specs']/scale,
       'reference':reference/scale,'constraints':torch.as_tensor(inputs[5]).long().clone(),'scale':scale,'n':n}
    # Public specifications are conditioning only; targets remain outside model inputs.
    s['known']=torch.cat((s['known'][:,:2]*g['preplaced'][:,None],
                          s['known'][:,2:]*g['locked'][:,None]),1)
    s['specs']=torch.where(g['locked'][:,None],s['specs'],torch.zeros_like(s['specs']))
    s['constraints'][:,2] = 0  # Raw MIB metadata stays in the archive only.
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
    for key in ('adjacency','group'):
        out[key]=torch.zeros(b,n,n)
        for i,r in enumerate(rows):out[key][i,:r['n'],:r['n']]=r[key]
    out['valid']=torch.arange(n)[None,:]<torch.tensor([r['n'] for r in rows])[:,None]
    out['scale']=torch.stack([r['scale'] for r in rows])
    return {k:v.to(device) for k,v in out.items()}


def decode(state,g):
    r=state[...,2].clamp(-5,5)
    wh=torch.stack(((g['area']*r.exp()).clamp_min(1e-12).sqrt(),
                    (g['area']/r.exp()).clamp_min(1e-12).sqrt()),dim=-1)
    xy=(state[...,:2]-wh/2).clamp_min(0)
    out=torch.cat((xy,wh),dim=-1)
    return out  # No fixed-shape or preplaced overwrite.
