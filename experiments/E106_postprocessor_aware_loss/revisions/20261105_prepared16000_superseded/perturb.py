"""Two mature centers and three symmetric, hard-mask-preserving local pairs."""
import torch
from .batches import decode


def structural_direction(state, graph, kind, generator):
    active = graph['mask'][0] * graph['valid'][0, :, None]
    n = state.shape[0]
    direction = torch.zeros_like(state)
    positions = decode(state[None], graph)[0]
    center = positions[:, :2] + positions[:, 2:] / 2
    if kind == 'group':
        groups = graph['constraints'][0, :, 3]
        candidates = torch.unique(groups[groups > 0])
        if len(candidates):
            ids = groups == candidates[int(torch.randint(len(candidates), (1,), generator=generator))]
            direction[ids, :2] = torch.randn(2, generator=generator)
    elif kind == 'boundary':
        codes = graph['constraints'][0, :, 4]
        low, high = positions[:, :2].amin(0), (positions[:, :2]+positions[:, 2:]).amax(0)
        for i in range(n):
            bits = int(codes[i])
            if bits & 1: direction[i, 0] -= positions[i, 0]-low[0]
            if bits & 2: direction[i, 0] += high[0]-positions[i, 0]-positions[i, 2]
            if bits & 8: direction[i, 1] -= positions[i, 1]-low[1]
            if bits & 4: direction[i, 1] += high[1]-positions[i, 1]-positions[i, 3]
    elif kind == 'cluster':
        i = int(torch.randint(n, (1,), generator=generator))
        ids = (center-center[i]).square().sum(-1).argsort()[:max(2, n//8)]
        direction[ids, :2] = center[ids]-center[ids].mean(0)
    elif kind == 'shape':
        direction[:, 2] = torch.randn(n, generator=generator)
    direction *= active
    if float(direction.square().sum()) < 1e-16:
        direction = torch.randn(state.shape, generator=generator) * active
    return direction


def generate(centers, graph, spec):
    generator = torch.Generator().manual_seed(spec['perturb_seed'])
    active = graph['mask'][0] * graph['valid'][0, :, None]
    if active.sum() == 0:
        raise ValueError('Registered problem has no movable state channels')
    unit = torch.cat((graph['area'][0].sqrt()[:, None].expand(-1, 2),
                      torch.ones_like(graph['area'][0, :, None])), -1)
    states = [centers[0].detach().clone(), centers[1].detach().clone()]
    pairs, directions = [], []
    kinds = ('group', 'boundary', 'cluster', 'shape')
    for j in range(3):
        root = 0 if j < 2 else 1
        center = centers[root]
        kind = 'gaussian' if j != 1 else kinds[spec['index'] % 4]
        radius = (.005 if spec['index'] % 2 == 0 else .020) if j == 0 else .020 if j == 1 else (.010 if spec['index'] % 2 == 0 else .030)
        direction = torch.randn(center.shape, generator=generator) * active if kind == 'gaussian' else structural_direction(center, graph, kind, generator)
        rms = (direction.square().sum() / active.sum()).sqrt().clamp_min(1e-12)
        direction = direction / rms
        plus, minus = center + radius*unit*direction, center - radius*unit*direction
        # Decoder restores exact public locked values; don't change any active mask.
        assert torch.equal(plus * (1-active), center * (1-active))
        assert torch.equal(minus * (1-active), center * (1-active))
        pairs.append(dict(plus=len(states), minus=len(states)+1, radius=radius, root=root, kind=kind))
        states.extend((plus, minus))
        directions.append(direction)
    return torch.stack(states), pairs, torch.stack(directions)
