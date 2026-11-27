"""Same fixed-G CUDA computation, with targeted actual graph-step protection."""
import torch
from .cuda_step import CapturedStep as OriginalCapturedStep
from .step_control import GraphStepControl

class CapturedStep(OriginalCapturedStep):
    def __init__(self,model,optimizer,sample,arm):
        super().__init__(model,optimizer,sample,arm)
        self.hidden_ref=model.last_hidden_max
        self.output_ref=model.last_output_max
        self.projection=GraphStepControl(model,.01)

    def replay(self,batch=None):
        if batch is not None:
            for k,v in batch.items():self.inputs[k].copy_(v)
        self.graph.replay()
        health=torch.stack((self.hidden_ref,self.output_ref))
        if not bool(self.finite & torch.isfinite(health).all()):raise RuntimeError('Nonfinite update rejected before AdamW')
        mature=self.projection.steps>=2000
        # Original non-prenorm160 initialization has dev G~265; do not mistake
        # its known finite initial scale for a mature-training blow-up.
        limit=5. if mature else 1e4
        grad_limit=1000. if mature else 1e7
        if not bool((self.values['total']<limit)&(self.norm<grad_limit)&(self.hidden_ref<1000.)&(self.output_ref<1000.)):
            raise RuntimeError('Explosive finite forward/gradient rejected before AdamW: '+str({'step':self.projection.steps+1,
                'G':float(self.values['total']),'grad_norm':float(self.norm),'hidden':float(self.hidden_ref),'state':float(self.output_ref)}))
        self.projection.before();self.optimizer.step();self.projection.after()
        return self.values,self.norm
