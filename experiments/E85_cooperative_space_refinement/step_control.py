"""Track adapter-only Adam updates; source encoder and output bounds are fixed."""
import torch
class AdapterStepTracker:
    def __init__(self,model):
        self.names=[name for name,module in model.named_modules() if any(p.requires_grad for p in module.parameters(recurse=False))]
        device=next(model.parameters()).device;self.clipped=torch.zeros(len(self.names),dtype=torch.long,device=device);self.steps=0
    def before(self):pass
    def after(self):self.steps+=1
    def summary(self):return {'policy':'Plain AdamW all warm-widened F, P42/P74 frozen/eval, fixed source-owned frames and per-stage residual caps','steps':self.steps,'group_names':self.names,'clipped_steps_per_group':self.clipped.cpu().tolist(),'max_relative_step':None}
