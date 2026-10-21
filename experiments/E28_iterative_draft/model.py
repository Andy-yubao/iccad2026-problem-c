"""Matched-parameter 2x2 controls: terminal representation x recurrent steps."""
import torch
from torch import nn
from .base_model import GeometryModel, GeometryBlock, decode
from .batches import PUBLIC_KEYS as BASE_KEYS

ARMS = {
    'single_pooled': {'full_pins': False, 'steps': 1},
    'single_full': {'full_pins': True, 'steps': 1},
    'recurrent_pooled': {'full_pins': False, 'steps': 4},
    'recurrent_full': {'full_pins': True, 'steps': 4},
}
EXTRA_KEYS = ('pin_xy', 'pin_weights', 'pin_valid', 'pin_indices', 'pin_link_weights', 'latent', 'draft')


def inputs_only(graph):
    return {k: graph[k] for k in (*BASE_KEYS, *EXTRA_KEYS) if k in graph}


def feedback(state, graph):
    """Public collision feedback, with no solver, reference or gradient oracle."""
    boxes = decode(state, graph)
    center = boxes[..., :2] + boxes[..., 2:] / 2
    delta = center[:, :, None] - center[:, None, :]
    depth = (boxes[:, :, None, 2:] + boxes[:, None, :, 2:]) / 2 - delta.abs()
    valid = graph['valid']
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs = pairs & ~torch.eye(state.shape[1], dtype=torch.bool, device=state.device)[None]
    penetration = depth.amin(-1).clamp_min(0) * pairs
    scale = graph['area'].sqrt().clamp_min(1e-6)
    load = (penetration / scale[:, :, None]).sum(-1)
    push = (delta.sign() * penetration[..., None]).sum(2) / scale[..., None]
    return torch.cat((push, load[..., None], boxes[..., 2:].log().clamp(-20, 20)), -1)


class DraftModel(nn.Module):
    def __init__(self, arm='recurrent_full', width=96):
        super().__init__()
        if arm not in ARMS or width % 4:
            raise ValueError('Unknown arm or width not divisible by four')
        self.arm, self.width = arm, width
        self.steps, self.full_pins = ARMS[arm]['steps'], ARMS[arm]['full_pins']
        self.base = GeometryModel(width=96, layers=4)
        self.base.requires_grad_(False)
        # Same modules and RNG initialization in all arms, including pooled controls.
        self.input = nn.Sequential(nn.Linear(27, width), nn.SiLU(), nn.LayerNorm(width))
        self.pin_encoder = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([GeometryBlock(width) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @torch.no_grad()
    def initial(self, graph):
        return self.base({k: graph[k] for k in BASE_KEYS})

    def terminal_message(self, graph, state):
        # Pooled controls expose exactly one weighted centroid, through the same encoder.
        weights = graph['pin_weights']
        weight_sum = weights.sum(-1, keepdim=True)
        boxes = decode(state, graph)
        center = boxes[..., :2] + boxes[..., 2:] / 2
        if self.full_pins:
            index = graph['pin_indices']
            batch = torch.arange(state.shape[0], device=state.device)[:, None, None]
            linked_xy = graph['pin_xy'][batch, index]
            delta = linked_xy - center[:, :, None]
            tokens = torch.cat((delta, delta.abs()), -1)
            return (self.pin_encoder(tokens) * graph['pin_link_weights'][..., None]).sum(2) / weight_sum.clamp_min(1e-8)
        centroid = (weights[..., None] * graph['pin_xy'][:, None]).sum(2) / weight_sum.clamp_min(1e-8)
        delta = centroid - center
        return self.pin_encoder(torch.cat((delta, delta.abs()), -1)) * (weight_sum > 0)

    def forward(self, graph, *, return_trajectory=False):
        if set(graph) - set((*BASE_KEYS, *EXTRA_KEYS)):
            raise ValueError('Supervision or unregistered input entered model')
        state = graph['draft'].detach() if 'draft' in graph else self.initial(graph)
        trajectory = [state]
        for _ in range(self.steps):
            observed = torch.cat((graph['features'], state, feedback(state, graph), graph['latent']), -1)
            h = self.input(observed) + self.terminal_message(graph, state)
            for block in self.blocks:
                h = block(h, graph, state)
            delta = self.head(h).tanh()
            local = graph['area'].sqrt().clamp_min(1e-6)
            # Match the maximum cumulative displacement in one- and four-step arms.
            gain = torch.stack((.5 * local, .5 * local, torch.full_like(local, .5)), -1) / self.steps
            state = state + delta * gain * graph['mask'] * graph['valid'][..., None]
            trajectory.append(state)
        return trajectory if return_trajectory else state


def trainable_state(model):
    return {k: v for k, v in model.state_dict().items() if not k.startswith('base.')}


def load_trainable(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith('base.') for key in missing):
        raise ValueError('Refiner checkpoint model structure differs')
