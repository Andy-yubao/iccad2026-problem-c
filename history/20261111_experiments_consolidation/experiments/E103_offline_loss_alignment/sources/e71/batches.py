"""Experiment-local preparation and padded batching; no test loader."""
import torch
from .features import graph_features,encode_target
from .ports import public_ports

PUBLIC_KEYS=('features','adjacency','mib','group','mask','area','known','specs','constraints','scale','valid',
             'pin_coordinates','pin_weights','pin_valid')


def inputs_only(batch):
    """Keep supervision outside the network's input contract."""
    result={key:batch[key] for key in PUBLIC_KEYS if key in batch}
    # Frozen shared cache retains17 channels; these three historical pin
    # summaries are excluded at the public network boundary, not merely zeroed.
    result['features']=result['features'][...,:14]
    return result


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
    s.update(public_ports(inputs,scale))
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
    for key in ('adjacency','mib','group','wire_b2b'):
        out[key]=torch.zeros(b,n,n)
        for i,r in enumerate(rows):out[key][i,:r['n'],:r['n']]=r[key]
    out['valid']=torch.arange(n)[None,:]<torch.tensor([r['n'] for r in rows])[:,None]
    out['scale']=torch.stack([r['scale'] for r in rows])
    p=max(1,max(r['pin_coordinates'].shape[1] for r in rows))
    out['pin_coordinates']=torch.zeros(b,n,p,2)
    out['pin_weights']=torch.zeros(b,n,p)
    out['pin_valid']=torch.zeros(b,n,p,dtype=torch.bool)
    for i,r in enumerate(rows):
        count=r['pin_coordinates'].shape[1]
        out['pin_coordinates'][i,:r['n'],:count]=r['pin_coordinates']
        out['pin_weights'][i,:r['n'],:count]=r['pin_weights']
        out['pin_valid'][i,:r['n'],:count]=r['pin_valid']
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
