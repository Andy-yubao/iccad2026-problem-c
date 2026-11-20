"""Same FP32 CUDA arithmetic and projection; explicit phase and per-case guard."""
import torch
from .batches import inputs_only
from .loss import per_case
from .captured_projection import CapturedProjection
from .augmentation import axis_swap

class CapturedStep:
    def __init__(self, model, optimizer, sample, phase='C'):
        self.model,self.optimizer,self.phase=model,optimizer,phase
        self.inputs={k:v.clone() for k,v in sample.items()}
        self.inputs['axis_swap']=torch.zeros(sample['valid'].shape[0],device=sample['valid'].device,dtype=torch.bool)
        rng=torch.cuda.get_rng_state_all()
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                augmented=axis_swap(self.inputs)
                values=per_case(model(inputs_only(augmented)),augmented,phase)
                values['total'].mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False,foreach=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            optimizer.zero_grad(set_to_none=False)
            augmented=axis_swap(self.inputs)
            output=model(inputs_only(augmented))
            self.per_layout=per_case(output,augmented,phase)
            self.values={k:v.mean() for k,v in self.per_layout.items()}
            self.values['total'].backward()
            self.norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False,foreach=True)
            self.max_case=self.per_layout['total'].amax()
            self.output_max=output.abs().amax()
            self.finite=torch.isfinite(torch.stack(list(self.values.values()))).all() & torch.isfinite(self.norm) & torch.isfinite(output).all() & torch.isfinite(self.per_layout['total']).all()
        torch.cuda.synchronize()
        torch.cuda.set_rng_state_all(rng)
        self.projection=CapturedProjection(model)

    def replay(self,batch=None):
        if batch is not None:
            for k,v in batch.items():self.inputs[k].copy_(v)
        self.graph.replay()
        if not bool(self.finite & (self.values['total']<1000.) & (self.max_case<1000.) & (self.norm<10000.)):
            raise RuntimeError('Numerical guard rejected candidate before AdamW')
        self.projection.before()
        self.optimizer.step()
        self.projection.after()
        return self.values,self.norm
