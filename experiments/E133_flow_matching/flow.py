"""Linear conditional paths, fixed common-random-number monitoring, Euler solve."""
import hashlib
import json
import torch
from .batches import inputs_only

def path(z0,z1,t):return (1-t[:,None,None])*z0+t[:,None,None]*z1,z1-z0

def objective(predicted,target,valid):
    channel=((predicted-target).square()*valid[...,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
    return dict(total=channel.mean(-1),position=channel[:,:2].mean(-1),shape=channel[:,2])

def member_seed(member,domain,draw=0):
    raw=json.dumps([20261112133,domain,member,draw],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8],'little')%(2**63-1)

def fixed_bank(rows,nodes,draw=0,domain='monitor',device='cuda'):
    noise=[];times=[]
    for row in rows:
        generator=torch.Generator(device='cpu').manual_seed(member_seed(list(row['member']),domain,draw))
        z=torch.zeros(nodes,3);z[:row['graph']['n']]=torch.randn(row['graph']['n'],3,generator=generator)
        noise.append(z);times.append((draw+torch.rand((),generator=generator).item())/4)
    return torch.stack(noise).to(device),torch.tensor(times,device=device)

@torch.no_grad()
def integrate(model,g,z0,steps=8,trace=False):
    condition=model.encode(inputs_only(g));valid=g['valid'];z=z0*valid[...,None]
    states=[model.unstandardize(z).clone()] if trace else None
    for k in range(steps):
        t=torch.full((z.shape[0],),k/steps,device=z.device)
        z=(z+model.velocity(condition,z,t,valid)/steps)*valid[...,None]
        if trace:states.append(model.unstandardize(z).clone())
    state=model.unstandardize(z)
    return (state,states) if trace else state

def training_statistics(rows):
    # Equal-case contribution; x/y pooled and logratio symmetrized to make
    # the frozen affine transform commute with exact axis-swap augmentation.
    moments=torch.zeros(4,dtype=torch.float64)
    for row in rows:
        target=row['graph']['target'].double()
        moments+=torch.stack((target[:,:2].mean(),target[:,:2].square().mean(),
                              target[:,2].square().mean(),torch.tensor(1.,dtype=torch.float64)))
    moments/=len(rows)
    m=moments[0];s=(moments[1]-m*m).clamp_min(.0025).sqrt();r=moments[2].clamp_min(.0025).sqrt()
    return dict(mean=[float(m),float(m),0.],std=[float(s),float(s),float(r)],
        count=len(rows),source='train16k cached target only;equal-case moments;axis-symmetric',dev_used=False,test_read=False)
