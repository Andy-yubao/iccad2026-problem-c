"""FP32 graph capture; preparation replays gradients without Adam updates."""
import torch
from .batches import inputs_only
from .loss import per_case
from .augmentation import axis_swap
from .captured_projection import CapturedProjection


class CapturedStep:
    def __init__(self, model, optimizer, sample, phase, updates_authorized=False):
        if updates_authorized:
            from .control import assert_training_authorized
            assert_training_authorized(phase)
        self.updates_authorized = updates_authorized
        self.model, self.optimizer, self.phase = model, optimizer, phase
        self.inputs = {k: v.clone() for k, v in sample.items()}
        self.inputs['axis_swap'] = torch.zeros(sample['valid'].shape[0], device='cuda', dtype=torch.bool)
        rng = torch.cuda.get_rng_state_all()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                optimizer.zero_grad(set_to_none=False)
                batch = axis_swap(self.inputs)
                per_case(model(inputs_only(batch)), batch, phase)['total'].mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            optimizer.zero_grad(set_to_none=False)
            batch = axis_swap(self.inputs)
            output = model(inputs_only(batch))
            self.per_layout = per_case(output, batch, phase)
            self.values = {k: v.mean() for k, v in self.per_layout.items()}
            self.values['total'].backward()
            self.norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False, foreach=True)
            self.max_case = self.per_layout['total'].amax()
            self.finite = (torch.isfinite(torch.stack(list(self.values.values()))).all()
                           & torch.isfinite(self.norm) & torch.isfinite(output).all()
                           & torch.isfinite(self.per_layout['total']).all())
        torch.cuda.synchronize()
        torch.cuda.set_rng_state_all(rng)
        self.projection = CapturedProjection(model)

    def backward_only(self):
        """No optimizer or projection mutation; this is the only preflight path."""
        self.graph.replay()
        if not bool(self.finite & (self.values['total'] < 1000.) & (self.max_case < 1000.) & (self.norm < 10000.)):
            raise RuntimeError('Numerical guard rejected candidate before AdamW')
        return self.values, self.norm

    def replay(self):
        # Deliberate defense against accidentally calling the training path in preparation.
        if not self.updates_authorized:
            raise PermissionError('Preparation capture: optimizer updates are disabled')
        values, norm = self.backward_only()
        self.projection.before()
        self.optimizer.step()
        self.projection.after()
        return values, norm
