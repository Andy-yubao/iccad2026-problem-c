"""FP32 CUDA graph for fixed-shape forward, unified Loss, backward and clipping."""
import torch
from .batches import inputs_only
from .loss import losses


class CapturedStep:
    def __init__(self, model, optimizer, sample, arm='geometry'):
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
            self.values = losses(model(inputs_only(self.inputs)), self.inputs, None, arm, 0.)
            self.values['total'].backward()
            self.norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
            self.finite = torch.isfinite(torch.stack(list(self.values.values()))).all() & torch.isfinite(self.norm)
        torch.cuda.synchronize()
        torch.cuda.set_rng_state_all(rng)

    def replay(self, batch=None):
        if batch is not None:
            for k, v in batch.items(): self.inputs[k].copy_(v)
        self.graph.replay()
        if not bool(self.finite): raise RuntimeError('Nonfinite loss or gradient; optimizer not advanced')
        self.optimizer.step()
        return self.values, self.norm
