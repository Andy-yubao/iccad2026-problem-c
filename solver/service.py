"""Frozen model inference, accelerated repair and independent hard validation."""
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from ._checks import independent_hard_checks
from ._gpu import Predictor
from ._identity import sha
from ._model.batches import decode
from ._model.model import GeometryModel
from ._postprocess.optimizer import optimize
from ._public_inputs import public_graph, repair_inputs


CHECKPOINT = Path(__file__).with_name('model.pt')
CHECKPOINT_SHA256 = 'ed2e627a87fb287bc6662586df4ffa86b3f344ee79994f1575d0fce6e8c88339'


class Solver:
    """Resident single-request solver for public inputs with at most 120 blocks."""

    def __init__(self, device='cuda'):
        if device not in ('cpu', 'cuda'):
            raise ValueError('Choose cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable. Select --device cpu to run the same solver on CPU.')
        self.device = device
        self.thread_limits = threadpool_limits(limits=1)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        if device == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            self.predictor = Predictor(CHECKPOINT, CHECKPOINT_SHA256)
        else:
            if sha(CHECKPOINT) != CHECKPOINT_SHA256:
                raise ValueError('Frozen E169 checkpoint changed')
            state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
            self.model = GeometryModel(**state['model_config']).eval()
            self.model.load_state_dict(state['model'], strict=True)

    @torch.inference_mode()
    def _draft(self, inputs):
        if self.device == 'cuda':
            return self.predictor.predict(inputs)[0]
        graph = public_graph(inputs)
        state = self.model(graph)[-1]
        layout = decode(state, graph)[0, :inputs[0]] * graph['scale'][0]
        return layout.numpy().astype(np.float64)

    def solve(self, public_inputs):
        """Return finite, independently verified rectangles in public units."""
        inputs = repair_inputs(public_inputs)
        if not 1 <= inputs[0] <= 120:
            raise ValueError('Expected 1–120 modules')
        draft = self._draft(inputs)
        if not np.isfinite(draft).all():
            raise ValueError('Nonfinite E169 draft')
        final, _ = optimize(draft, inputs)
        hard = independent_hard_checks(final.tolist(), inputs[1], inputs[5], inputs[6])
        if not hard['is_feasible'] or (final[:, :2] < 0).any():
            raise ValueError(f'Output failed independent hard-constraint checks: {hard}')
        return final
