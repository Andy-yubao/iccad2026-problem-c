"""Eager guarded updates; no graph warmup, probes or disposable updates."""
import torch
from .public import inputs_only
from .loss import per_case
from .step_control import GraphStepControl
from .augmentation import axis_swap


class CapturedStep:
    # Same trainer interface; deliberately no model/backward execution in init.
    def __init__(self,model,optimizer,sample,phase):
        self.model,self.optimizer,self.phase=model,optimizer,phase
        self.inputs={k:v.clone() for k,v in sample.items()}
        self.inputs['axis_swap']=torch.zeros(sample['valid'].shape[0],device=sample['valid'].device,dtype=torch.bool)
        self.projection=GraphStepControl(model)

    def replay(self):
        self.optimizer.zero_grad(set_to_none=True)
        batch=axis_swap(self.inputs);output=self.model(inputs_only(batch))
        per=per_case(output,batch,self.phase);values={k:v.mean() for k,v in per.items()}
        values['total'].backward()
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.,error_if_nonfinite=False,foreach=True)
        finite=torch.isfinite(output).all()&torch.isfinite(torch.stack(list(values.values()))).all()&torch.isfinite(norm)&torch.isfinite(per['total']).all()
        if not bool(finite&(per['total'].amax()<1000.)&(norm<10000.)):
            raise RuntimeError('Numerical guard rejected update before AdamW')
        self.projection.before();self.optimizer.step();self.projection.after()
        return values,norm


class EagerStep(CapturedStep):
    def __init__(self,model,optimizer,sample,phase,projection):
        super().__init__(model,optimizer,sample,phase)
        self.projection=projection
