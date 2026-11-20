"""Registered eager FP32 FM update; no backward CUDA graph on this Windows build."""
import torch
from .batches import inputs_only
from .augmentation import axis_swap
from .flow import path,objective
from .step_control import GraphStepControl

def augmented(batch):return axis_swap(batch)

class ProtectedStep:
    def __init__(self,model,optimizer,sample,phase='fm',updates_authorized=False):
        if updates_authorized:
            from .control import assert_training_authorized
            assert_training_authorized(phase)
        self.updates_authorized=updates_authorized;self.model=model;self.optimizer=optimizer
        self.inputs={k:v.clone() for k,v in sample.items()}
        self.inputs['axis_swap']=torch.zeros(len(sample['valid']),dtype=torch.bool,device='cuda')
        self.projection=GraphStepControl(model)
    def backward_only(self):
        self.optimizer.zero_grad(set_to_none=True)
        batch=augmented(self.inputs);valid=batch['valid']
        z1=self.model.standardize(batch['target'])*valid[...,None]
        self.z0=torch.randn_like(z1)*valid[...,None]
        self.t=torch.rand(z1.shape[0],device=z1.device)
        zt,target=path(self.z0,z1,self.t)
        out=self.model(inputs_only(batch),zt,self.t)
        self.per_layout=objective(out,target,valid)
        self.values={k:v.mean() for k,v in self.per_layout.items()}
        self.values['total'].backward()
        self.norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.,error_if_nonfinite=False,foreach=True)
        self.max_case=self.per_layout['total'].amax()
        finite=torch.isfinite(torch.stack(list(self.values.values()))).all()&torch.isfinite(out).all()&torch.isfinite(self.norm)&torch.isfinite(self.per_layout['total']).all()
        if not bool(finite&(self.max_case<100.)&(self.values['total']<100.)&(self.norm<10000.)):
            raise RuntimeError('FM numerical guard rejected before AdamW')
        return self.values,self.norm
    def replay(self):
        if not self.updates_authorized:raise PermissionError('Preparation only')
        values,norm=self.backward_only();self.projection.before();self.optimizer.step();self.projection.after()
        return values,norm
