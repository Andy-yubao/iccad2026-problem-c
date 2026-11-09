"""Fixed coefficient vector per branch; same FP32 model and bounded Adam updates."""
import torch
from ..batches import inputs_only
from ..simple_loss import components
from ..loss_config import A,B
from .captured_projection import CapturedProjection
from .augmentation import axis_swap

def per_case(state,graph,coefficients):
    values=components(state,graph,dict(overlap=A,task=B))
    values['legacy_total']=values['total']
    values['total']=sum(coefficients[k]*values[k] for k in coefficients)
    return values

class CapturedStep:
    def __init__(self,model,optimizer,sample,coefficients,projection=None):
        self.model,self.optimizer=model,optimizer
        self.coefficients=dict(coefficients)
        self.inputs={k:v.clone() for k,v in sample.items()}
        self.inputs['axis_swap']=torch.zeros(sample['valid'].shape[0],device=sample['valid'].device,dtype=torch.bool)
        rng=torch.cuda.get_rng_state_all();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                augmented=axis_swap(self.inputs)
                values=per_case(model(inputs_only(augmented)),augmented,self.coefficients)
                values['total'].mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            optimizer.zero_grad(set_to_none=False)
            augmented=axis_swap(self.inputs)
            out=model(inputs_only(augmented));self.per_case=per_case(out,augmented,self.coefficients)
            self.values={k:v.mean() for k,v in self.per_case.items()}
            self.values['total'].backward()
            self.norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=True)
            self.max_case=self.per_case['total'].amax()
            self.finite=torch.isfinite(out).all() & torch.isfinite(torch.stack(list(self.values.values()))).all() & torch.isfinite(self.norm)
        torch.cuda.synchronize();torch.cuda.set_rng_state_all(rng)
        self.projection=CapturedProjection(model)
        if projection:
            assert projection['group_names']==self.projection.names
            self.projection.steps=projection['steps']
            self.projection.clipped.copy_(torch.tensor(projection['clipped_steps_per_group'],device='cuda'))
            self.projection.last_scales.copy_(torch.tensor(projection['last_scales'],device='cuda'))

    def replay(self,batch=None):
        if batch is not None:
            for k,v in batch.items():self.inputs[k].copy_(v)
        self.graph.replay()
        if not bool(self.finite & (self.max_case<1000) & (self.values['total']<1000) & (self.norm<10000)):
            raise RuntimeError('Weighted branch numerical guard rejected before AdamW')
        self.projection.before();self.optimizer.step();self.projection.after()
        return self.values,self.norm
