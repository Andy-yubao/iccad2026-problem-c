"""Project only graph-aggregation AdamW parameter displacements.

Forward architecture, graph inputs and the fixed objective are unchanged. Each
local Linear's joint weight/bias step is bounded by 1% of its pre-step norm.
This controls Adam's actual displacement, rather than its incoming gradient.
"""
import torch

class GraphStepControl:
    def __init__(self,model,max_relative_step=.01):
        self.groups=[tuple(layer.local.parameters()) for layer in model.layers]
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
        return {'policy':'AdamW proposal then joint graph-local weight/bias displacement projection',
            'max_relative_step':self.max_relative_step,'steps':self.steps,
            'clipped_steps_per_layer':self.clipped.cpu().tolist(),'last_scales':self.last_scales.cpu().tolist()}
