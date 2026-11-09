"""Same-case frozen-parent relative gates plus original absolute numerical gates."""
import torch

def safe(total,parent_total,norm=None,finite=None):
    ok=torch.isfinite(total).all()&torch.isfinite(parent_total).all()
    if finite is not None:ok=ok&finite
    if norm is not None:ok=ok&torch.isfinite(norm)&(norm<10000.)
    mean=total.mean();base_mean=parent_total.mean()
    case_limit=torch.maximum(10*parent_total,parent_total+20.)
    mean_limit=torch.maximum(3*base_mean,base_mean+5.)
    return ok&(total.amax()<1000.)&(mean<1000.)&(total<case_limit).all()&(mean<mean_limit)

def details(total,parent_total,norm=None):
    return dict(mean_total=float(total.mean()),max_case=float(total.max()),
        parent_mean=float(parent_total.mean()),parent_max=float(parent_total.max()),
        relative_mean_limit=float(torch.maximum(3*parent_total.mean(),parent_total.mean()+5.)),
        max_relative_case_ratio=float((total/parent_total.clamp_min(1e-6)).max()),
        gradient_norm=None if norm is None else float(norm),absolute_case_limit=1000.,gradient_limit=10000.)

class LossExplosion(RuntimeError):
    def __init__(self,message,diagnostics=None):
        super().__init__(message)
        self.diagnostics=diagnostics
