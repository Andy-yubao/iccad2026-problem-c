"""FP32 CUDA graph for fixed-shape forward, unified Loss, backward and clipping."""
import torch
from .batches import inputs_only
from .loss import losses
from .curriculum import position_weight


class CapturedStep:
    def __init__(self, model, optimizer, sample, arm='full_ports'):
        self.model, self.optimizer, self.arm = model, optimizer, arm
        rng = torch.cuda.get_rng_state_all()
        self.inputs = {k: v.clone() for k, v in sample.items()}
        self.position_coefficient = torch.tensor(position_weight(1), device=sample['area'].device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                value = losses(model(inputs_only(self.inputs)), self.inputs, None, arm,
                               position_coefficient=self.position_coefficient)
                value['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            optimizer.zero_grad(set_to_none=False)
            self.values = losses(model(inputs_only(self.inputs)), self.inputs, None, arm,
                                 position_coefficient=self.position_coefficient)
            self.values['total'].backward()
            self.norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
            self.finite = torch.isfinite(torch.stack(list(self.values.values()))).all() & torch.isfinite(self.norm)
        torch.cuda.synchronize()
        torch.cuda.set_rng_state_all(rng)

    def replay(self, batch=None, *, epoch=1):
        if batch is not None:
            for k, v in batch.items(): self.inputs[k].copy_(v)
        self.position_coefficient.fill_(position_weight(epoch))
        self.graph.replay()
        if not bool(self.finite): raise RuntimeError('Nonfinite loss or gradient; optimizer not advanced')
        self.optimizer.step()
        return self.values, self.norm
