"""Frozen E7 attention draft followed by the selected E8 conic solver."""
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import torch
from .frozen.model import GeometryModel
from .frozen.inference import prepare_inference
from .frozen.batches import decode
from .optimizer import optimize

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent / 'E10_integrated_baseline/runs/20261017_factorial_seed0/attention_geometry/checkpoints/best.pt'
CHECKPOINT_SHA256 = '47df667ed0b3994b60e554012255bcef1f7ec5cc76874b9e7493a30216f26c28'


class Baseline:
    """Loads once; solve accepts only the public masked problem tuple."""
    def __init__(self, device='cpu'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        if not CHECKPOINT.is_file():
            raise FileNotFoundError(f'Local frozen checkpoint required: {CHECKPOINT}; see README.md')
        if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError('Frozen checkpoint hash mismatch')
        self.device = device
        state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
        self.model = GeometryModel(**state['model_config']).to(device)
        self.model.load_state_dict(state['model'])
        self.model.eval()

    def draft(self, solver_inputs):
        public = prepare_inference(solver_inputs, self.device)
        with torch.no_grad():
            positions = (decode(self.model(public), public)[0] * public['scale'][0]).cpu().numpy().astype(float)
        constraints, specs = np.asarray(solver_inputs[5]), np.asarray(solver_inputs[6])
        locked = constraints[:, :2].any(1)
        positions[locked, 2:] = specs[locked, 2:]
        positions[constraints[:, 1] != 0] = specs[constraints[:, 1] != 0]
        return positions

    def solve(self, solver_inputs, seconds=5., extra_seconds=16.):
        if seconds <= 0:
            raise ValueError('seconds must be positive')
        start = time.perf_counter()
        draft = self.draft(solver_inputs)
        inference_seconds = time.perf_counter() - start
        positions, stats = optimize(draft, solver_inputs, seconds=seconds)
        from .stable import polish
        positions,stable=polish(draft,positions,solver_inputs,stats,seconds=extra_seconds)
        return positions, {**stats, 'inference_seconds': inference_seconds,
                           'stability':stable,
                           'end_to_end_seconds': time.perf_counter() - start}
