"""Replay a fixed-size CUDA Graph with an active-case mask for the tail."""
import torch
from .public import inputs_only
from .loss import per_case
from .captured_projection import CapturedProjection
from .augmentation import axis_swap


class CapturedStep:
    def __init__(self,model,optimizer,sample):
        self.model,self.optimizer=model,optimizer
        self.stage,self.joint=model.stage,model.prefix_unfrozen
        self.inputs={k:v.clone() for k,v in sample.items()}
        self.inputs['axis_swap']=torch.zeros(sample['valid'].shape[0],device=sample['valid'].device,dtype=torch.bool)
        self.active=torch.ones_like(self.inputs['axis_swap'])
        self.trainable=[p for p in model.parameters() if p.requires_grad]
        rng=torch.cuda.get_rng_state_all()
        optimizer.zero_grad(set_to_none=True)
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.forward_backward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.values,self.norm,self.accepted=self.forward_backward()
        torch.cuda.synchronize()
        torch.cuda.set_rng_state_all(rng)
        self.projection=CapturedProjection(model)

    def set_case_count(self,count):
        self.active[:count].fill_(True)
        self.active[count:].fill_(False)

    def forward_backward(self):
        # Keep the captured gradient tensor addresses stable.
        self.optimizer.zero_grad(set_to_none=False)
        batch=axis_swap(self.inputs);output=self.model(inputs_only(batch))
        per=per_case(output,batch,self.stage,self.joint)
        total_cases=self.active.sum()
        values={k:torch.where(self.active,v,0.).sum()/total_cases for k,v in per.items()}
        values['total'].backward()
        norm=torch.nn.utils.clip_grad_norm_(self.trainable,1.,error_if_nonfinite=False,foreach=True)
        finite_output=torch.isfinite(output).all(dim=(0,2,3))|~self.active
        case_max=per['total'].masked_fill(~self.active,-torch.inf).amax()
        finite=finite_output.all()&torch.isfinite(torch.stack(list(values.values()))).all()&torch.isfinite(norm)
        return values,norm,finite&(case_max<1000.)&(norm<10000.)

    def update(self,accepted):
        if not bool(accepted):
            raise RuntimeError('Numerical guard rejected update before AdamW')
        self.projection.before();self.optimizer.step();self.projection.after()

    def replay(self):
        self.graph.replay()
        self.update(self.accepted)
        return self.values,self.norm
