"""FP32 CUDA graph for fixed-shape forward, unified Loss, backward and clipping."""
import torch
from .batches import inputs_only
from .loss import losses


class CapturedStep:
    def __init__(self, model, optimizer, sample, arm='geometry96'):
        self.model, self.optimizer, self.arm = model, optimizer, arm
        rng = torch.cuda.get_rng_state_all()
        self.inputs = {k: v.clone() for k, v in sample.items()}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                value = losses(model(inputs_only(self.inputs)), self.inputs, None, arm, 0.)
                value['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            optimizer.zero_grad(set_to_none=False)
            state=model(inputs_only(self.inputs))
            self.values=losses(state,self.inputs,None,arm,0.)
            self.health=torch.stack((state.abs().amax(),model.last_hidden_max))
            self.values['total'].backward()
            self.norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
            self.finite = torch.isfinite(torch.stack(list(self.values.values()))).all() & torch.isfinite(self.norm) & torch.isfinite(self.health).all()
            self.stable=(self.values['total']<1e4)&(self.norm<1e6)&(self.health[0]<1e4)&(self.health[1]<1e5)
        torch.cuda.synchronize()
        torch.cuda.set_rng_state_all(rng)

    def replay(self, batch=None):
        if batch is not None:
            for k, v in batch.items(): self.inputs[k].copy_(v)
        self.graph.replay()
        if not bool(self.finite): raise RuntimeError('Nonfinite loss or gradient; optimizer not advanced')
        if not bool(self.stable):raise RuntimeError('Finite-but-explosive numerical guard; optimizer not advanced: '+str({'total':float(self.values['total']),'grad_norm':float(self.norm),'state_hidden_max':self.health.tolist()}))
        self.optimizer.step()
        return self.values, self.norm
