"""Bound each trainable module's joint AdamW weight/bias displacement.

All fresh Linear/LayerNorm groups, including graph-local,
use a fixed .5% relative step bound; forward and Loss stay fixed.
"""
import torch

class GraphStepControl:
    def __init__(self,model,max_relative_step=.005):
        entries=[(name,tuple(p for p in module.parameters(recurse=False) if p.requires_grad)) for name,module in model.named_modules()]
        entries=[(name,group) for name,group in entries if group]
        self.names=[name for name,group in entries]
        self.groups=[group for name,group in entries]
        self.previous=[tuple(torch.empty_like(p) for p in group) for group in self.groups]
        self.max_relative_step=max_relative_step
        device=self.groups[0][0].device
        self.clipped=torch.zeros(len(self.groups),dtype=torch.long,device=device)
        self.steps=0
        self.last_scales=torch.ones(len(self.groups),device=device)

    @torch.no_grad()
    def before(self):
        for group,old in zip(self.groups,self.previous):
            for p,v in zip(group,old):v.copy_(p)

    @torch.no_grad()
    def after(self):
        scales=[]
        for index,(group,old) in enumerate(zip(self.groups,self.previous)):
            norm=torch.stack([v.square().sum() for v in old]).sum().sqrt()
            changes=[p-v for p,v in zip(group,old)]
            update=torch.stack([d.square().sum() for d in changes]).sum().sqrt()
            scale=(self.max_relative_step*norm.clamp_min(1e-8)/update.clamp_min(1e-12)).clamp_max(1.)
            for p,v,d in zip(group,old,changes):p.copy_(torch.where(scale<1.,v+scale*d,p))
            self.clipped[index]+=(scale<1.).long();scales.append(scale)
        self.last_scales=torch.stack(scales);self.steps+=1

    def summary(self):
        return {'policy':'AdamW proposal then per-trainable-module joint displacement projection; all randomly initialized graph-local tensors trainable',
            'max_relative_step':self.max_relative_step,'steps':self.steps,
            'group_names':self.names,'clipped_steps_per_group':self.clipped.cpu().tolist(),'last_scales':self.last_scales.cpu().tolist()}
