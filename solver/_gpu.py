"""Compute E169 drafts with one resident GPU model and reusable CUDA Graphs."""
import time

import numpy as np
import torch

from ._public_inputs import public_graph
from ._model.batches import decode
from ._identity import sha
from ._model.model import GeometryModel


MODULE_SLOTS = 120
PORT_AXES = dict(port_features=1, port_valid=1, role_edges=2, role_group=2, role_hops=2)


class CapturedForward:
    def __init__(self, model, graph, ports, neighbors):
        self.host = {}
        for key, value in graph.items():
            shape = list(value.shape)
            if key in PORT_AXES:
                shape[PORT_AXES[key]] += ports - graph['port_valid'].shape[1]
            self.host[key] = torch.empty(shape, dtype=value.dtype, pin_memory=True)
        self.device = {key: torch.empty_like(value, device='cuda') for key, value in self.host.items()}
        self.copy_inputs(graph)

        model.shape_neighbor_count = neighbors
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.forward(model)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.capture, stream=stream):
            self.output = self.forward(model)
        torch.cuda.synchronize()

    def forward(self, model):
        state = model(self.device)[-1]
        return decode(state, self.device)[0] * self.device['scale'][0]

    def copy_inputs(self, graph):
        for key, value in graph.items():
            host = self.host[key]
            if key in PORT_AXES:
                axis = PORT_AXES[key]
                host.fill_(-1 if key == 'role_hops' else 0)
                host.narrow(axis, 0, value.shape[axis]).copy_(value)
            else:
                host.copy_(value)
            self.device[key].copy_(host, non_blocking=True)

    def predict(self, graph, n):
        self.copy_inputs(graph)
        self.capture.replay()
        # The blocking download also completes the input copies before host
        # buffers are reused by the next single-case request.
        return self.output[:n].cpu().numpy().astype(np.float64)


class Predictor:
    """Single-dispatcher, fixed FP32 inference with the original public graph."""

    def __init__(self, checkpoint, expected_hash):
        if sha(checkpoint) != expected_hash:
            raise ValueError('Frozen E169 checkpoint changed')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = GeometryModel(**state['model_config']).eval().cuda()
        self.model.load_state_dict(state['model'], strict=True)
        self.captures = {}

    @torch.inference_mode()
    def predict(self, inputs):
        n = int(inputs[0])
        if not 1 <= n <= MODULE_SLOTS:
            raise ValueError('Module count exceeds the registered E169 inference shape')
        start = time.perf_counter()
        graph = public_graph(inputs, pad=MODULE_SLOTS)
        ports = graph['port_valid'].shape[1]
        # Padding only adds invalid port slots. Large inputs use another
        # capture instead of truncating ports or same-group contacts.
        port_slots = 1 << max(9, (ports - 1).bit_length())
        max_neighbors = int((graph['group'] > 0).sum(-1).max())
        neighbor_slots = 16 if max_neighbors <= 16 else MODULE_SLOTS
        signature = port_slots, neighbor_slots
        graph_seconds = time.perf_counter() - start
        capture_seconds = 0.
        capture_cpu_seconds = 0.
        if signature not in self.captures:
            start = time.perf_counter()
            cpu_start = time.process_time()
            self.captures[signature] = CapturedForward(self.model, graph, *signature)
            capture_seconds = time.perf_counter() - start
            capture_cpu_seconds = time.process_time() - cpu_start
        start = time.perf_counter()
        draft = self.captures[signature].predict(graph, n)
        if not np.isfinite(draft).all():
            raise ValueError('Nonfinite E169 draft')
        return draft, dict(graph_seconds=graph_seconds, forward_seconds=time.perf_counter() - start,
                           capture_seconds=capture_seconds, capture_cpu_seconds=capture_cpu_seconds)
