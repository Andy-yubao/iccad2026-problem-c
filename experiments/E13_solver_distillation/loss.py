"""Teacher geometry and optional structure; no original reference supervision."""
import torch
import torch.nn.functional as F

def mean_mask(x,mask):return (x*mask).sum()/mask.sum().clamp_min(1)
def balanced_binary(logits,target,mask):
    values=F.binary_cross_entropy_with_logits(logits,target,reduction='none');parts=[]
    for flag in (False,True):
        selected=mask&(target.bool()==flag)
        if selected.any():parts.append(values[selected].mean())
    return torch.stack(parts).mean() if parts else logits.sum()*0

def losses(out,g,t,structure):
    valid=g['valid'];mask=g['mask'].bool()&valid[...,None]
    local=g['area'].clamp_min(1e-12).sqrt()
    center=(out['geometry'][...,:2]-t['geometry'][...,:2])/local[...,None]
    ratio=out['geometry'][...,2]-t['geometry'][...,2]
    cm=mask[...,:2];rm=mask[...,2]
    center_mse=mean_mask(center.square(),cm);ratio_mse=mean_mask(ratio.square(),rm)
    result={'center_mse':center_mse,'ratio_mse':ratio_mse,'geometry':center_mse+ratio_mse}
    metrics={'center_sq_sum':float((center.square()*cm).sum().detach()),'center_count':int(cm.sum()),
      'ratio_sq_sum':float((ratio.square()*rm).sum().detach()),'ratio_count':int(rm.sum())}
    if structure:
        n=valid.shape[1];pairs=valid[:,:,None]&valid[:,None,:]&~torch.eye(n,device=valid.device,dtype=torch.bool)[None]
        group=(g['group']>0)&pairs;mib=(g['mib']>0)&pairs
        requested=((g['constraints'][...,4,None]&torch.tensor([1,2,4,8],device=valid.device))!=0)&valid[...,None]
        boundary=balanced_binary(out['boundary'],t['boundary'],requested)
        presence=torch.logsumexp(out['contact'][...,:4],-1)-out['contact'][...,4]
        positive=(t['contact']<4)&group
        contact=balanced_binary(presence,(t['contact']<4).float(),group)
        if positive.any():contact=contact+F.cross_entropy(out['contact'][...,:4][positive],t['contact'][positive])
        same=(out['mib']+out['mib'].transpose(1,2))/2
        mib_loss=balanced_binary(same,t['mib'],mib)
        result.update(boundary=boundary,contact=contact,mib=mib_loss)
        prediction=(presence>0)&group
        for prefix,pred,target,allowed in [('contact',prediction,t['contact']<4,group),
                  ('boundary',out['boundary']>0,t['boundary'].bool(),requested),
                  ('mib',same>0,t['mib'].bool(),mib)]:
            metrics.update({prefix+'_tp':int((pred&target&allowed).sum()),prefix+'_fp':int((pred&~target&allowed).sum()),
              prefix+'_fn':int((~pred&target&allowed).sum())})
    result['total']=result['geometry']+(sum(result[k] for k in ('boundary','contact','mib'))*.1 if structure else 0.)
    return result,metrics
