"""FP32 CUDA graph, frozen parent and clean trust; strict gates before every AdamW."""
import torch
from .loss import compute
from .captured_projection import CapturedProjection
from .augmentation import axis_swap
from .guards import safe,details,LossExplosion

class CapturedStep:
    def __init__(self,model,optimizer,sample,parent,phase='baseline'):
        self.model,self.optimizer,self.parent=model,optimizer,parent
        self.inputs={k:v.clone() for k,v in sample.items()}
        self.inputs['axis_swap']=torch.zeros(sample['valid'].shape[0],device='cuda',dtype=torch.bool)
        rng=torch.cuda.get_rng_state_all();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                output,values,baseline=compute(model,parent,axis_swap(self.inputs),training=True)
                finite=torch.isfinite(torch.stack(list(values.values()))).all() & torch.isfinite(output).all()
                if not bool(safe(values['total'],baseline['total'],finite=finite)):
                    raise LossExplosion('Loss explosion during graph warmup',details(values['total'],baseline['total']))
                values['total'].mean().backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False,foreach=True)
                if not bool(safe(values['total'],baseline['total'],norm=norm,finite=finite)):
                    raise LossExplosion('Gradient explosion during graph warmup',details(values['total'],baseline['total'],norm))
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            optimizer.zero_grad(set_to_none=False)
            output,self.per_layout,baseline=compute(model,parent,axis_swap(self.inputs),training=True)
            self.parent_total=baseline['total']
            self.values={k:v.mean() for k,v in self.per_layout.items()}
            self.values['total'].backward()
            self.norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=False,foreach=True)
            self.max_case=self.per_layout['total'].amax();self.output_max=output.abs().amax()
            self.finite=torch.isfinite(torch.stack(list(self.per_layout.values()))).all() & torch.isfinite(output).all()
            self.ok=safe(self.per_layout['total'],self.parent_total,self.norm,self.finite)
            self.mean_limit=torch.maximum(3*self.parent_total.mean(),self.parent_total.mean()+5.)
        torch.cuda.synchronize();torch.cuda.set_rng_state_all(rng)
        self.projection=CapturedProjection(model)

    def diagnostics(self):
        return dict(**details(self.per_layout['total'],self.parent_total,self.norm),
            per_case_total=self.per_layout['total'].detach().cpu(),parent_total=self.parent_total.detach().cpu(),
            mean_terms={k:float(v) for k,v in self.values.items()},finite=bool(self.finite))

    def replay(self,batch=None):
        if batch is not None:
            for k,v in batch.items():self.inputs[k].copy_(v)
        self.graph.replay()
        return self.step_if_safe()

    def step_if_safe(self):
        if not bool(self.ok):
            raise LossExplosion('Loss explosion: rejected before AdamW',self.diagnostics())
        self.projection.before();self.optimizer.step();self.projection.after()
        return self.values,self.norm
