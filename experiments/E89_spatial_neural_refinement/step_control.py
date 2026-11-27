"""Fresh corrector uses ordinary clipped AdamW; frozen placer cannot move.

Retain the training loop's resumable step accounting API without imposing the
legacy displacement bound relative to zero-initialized output-head weights.
"""
import torch


class GraphStepControl:
    def __init__(self,model,max_relative_step=.005):
        entries=[(name,tuple(p for p in module.parameters(recurse=False) if p.requires_grad))
                 for name,module in model.named_modules()]
        entries=[(name,group) for name,group in entries if group]
        self.names=[name for name,_ in entries]
        self.clipped=torch.zeros(len(entries),dtype=torch.long,device=entries[0][1][0].device)
        self.last_scales=torch.ones(len(entries),device=self.clipped.device)
        self.steps=0

    def before(self):pass
    def after(self):self.steps+=1

    def summary(self):
        return dict(policy='fresh corrector AdamW with global gradient clipping1; no parameter projection; placer frozen',
            steps=self.steps,group_names=self.names,clipped_steps_per_group=self.clipped.cpu().tolist(),
            last_scales=self.last_scales.cpu().tolist())
