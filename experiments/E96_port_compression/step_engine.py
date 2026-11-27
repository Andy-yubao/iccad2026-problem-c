"""Dynamic FP32 eager updates with combined pre-Adam guards."""
import torch
from .loss import per_case

def finite_parameters(parameters):
    return torch.isfinite(torch.stack(torch._foreach_norm(parameters))).all()

class PackedStep:
    def __init__(self,model,optimizer):
        self.model,self.optimizer=model,optimizer
        self.parameters=list(model.corrector.parameters())

    def replay(self,batch,draft):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        prediction=self.model.corrector(batch,draft)
        cases=per_case(prediction,batch,draft)
        values={k:v.mean() for k,v in cases.items()}
        values['total'].backward()
        norm=torch.nn.utils.clip_grad_norm_(self.parameters,1.,error_if_nonfinite=False,foreach=True)
        hidden=self.model.corrector.last_hidden_rms.amax()
        finite=torch.isfinite(torch.stack(list(values.values()))).all() & torch.isfinite(norm) & torch.isfinite(prediction).all() & torch.isfinite(draft).all()
        gate=torch.stack((finite.to(torch.float32),cases['total'].amax(),norm,hidden)).detach().cpu().tolist()
        if gate[0]!=1 or gate[1]>=1000 or gate[2]>=10000 or not gate[3]<1000:
            raise RuntimeError('Numerical/hidden-scale guard rejected candidate before AdamW')
        self.optimizer.step()
        if not bool(finite_parameters(self.parameters)):
            raise RuntimeError('Nonfinite parameters after AdamW; no automatic retry')
        return values,gate[2],gate[3]
