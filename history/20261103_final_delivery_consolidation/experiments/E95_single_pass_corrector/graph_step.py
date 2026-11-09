"""Static FP32 efficient attention graph; guarded original AdamW outside capture."""
import gc
import torch
from .loss import per_case
from .step_engine import finite_parameters

class CapturedStep:
    def __init__(self,model,optimizer,sample,draft):
        self.model,self.optimizer=model,optimizer
        self.parameters=list(model.corrector.parameters())
        self.inputs={k:v.clone() for k,v in sample.items()};self.draft=draft.clone()
        # Release an unused RMS diagnostic graph before assigning capture streams.
        model.corrector.last_hidden_rms=None
        optimizer.zero_grad(set_to_none=True);gc.collect()
        random_state=torch.cuda.get_rng_state_all();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream());model.train()
        with torch.cuda.stream(stream):
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction=model.corrector(self.inputs,self.draft)
                values=per_case(prediction,self.inputs,self.draft)
                values['total'].mean().backward()
                torch.nn.utils.clip_grad_norm_(self.parameters,1.,foreach=True)
                del prediction,values
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        gc.collect();torch.cuda.empty_cache()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            optimizer.zero_grad(set_to_none=True)
            prediction=model.corrector(self.inputs,self.draft)
            cases=per_case(prediction,self.inputs,self.draft)
            self.values={k:v.mean() for k,v in cases.items()}
            self.values['total'].backward()
            self.norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=False,foreach=True)
            self.hidden=model.corrector.last_hidden_rms.amax()
            finite=torch.isfinite(torch.stack(list(self.values.values()))).all() & torch.isfinite(self.norm) & torch.isfinite(prediction).all() & torch.isfinite(self.draft).all()
            self.gate=torch.stack((finite.to(torch.float32),cases['total'].amax(),self.norm,self.hidden))
        torch.cuda.synchronize();torch.cuda.set_rng_state_all(random_state)

    def replay(self,batch=None,draft=None):
        if batch is not None:
            for key,value in batch.items():self.inputs[key].copy_(value,non_blocking=True)
        if draft is not None:self.draft.copy_(draft,non_blocking=True)
        self.graph.replay()
        gate=self.gate.detach().cpu().tolist()
        if gate[0]!=1 or gate[1]>=1000 or gate[2]>=10000 or not gate[3]<1000:
            raise RuntimeError('Numerical/hidden-scale guard rejected graph candidate before AdamW')
        self.optimizer.step()
        if not bool(finite_parameters(self.parameters)):
            raise RuntimeError('Nonfinite graph parameters after AdamW; stop without retry')
        return self.values,gate[2],gate[3]
