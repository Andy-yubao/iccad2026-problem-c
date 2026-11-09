"""Whole-problem batching keeps every local pair and split together."""
import torch
from .common import RUN
from .batches import PUBLIC_KEYS


def load_case(index):
    row = torch.load(RUN/f'data/{index:05d}.pt', map_location='cpu', weights_only=False)
    if not row['complete'] or len(row['metrics']) != 5:
        raise ValueError('Incomplete five-draft record')
    return row


def batch_cases(rows, device='cpu'):
    count, variants = len(rows), 5
    n = max(r['n'] for r in rows)
    graph = {}
    for key in PUBLIC_KEYS:
        t = rows[0]['graph'][key]
        if key in ('adjacency', 'mib', 'group'):
            out = torch.zeros((count*variants, n, n), dtype=t.dtype)
            for i, row in enumerate(rows):
                size = row['n']
                out[i*variants:(i+1)*variants, :size, :size] = row['graph'][key]
        elif key == 'scale':
            out = torch.cat([r['graph'][key].repeat(variants) for r in rows])
        else:
            shape = (count*variants, n) + t.shape[2:]
            out = torch.zeros(shape, dtype=t.dtype)
            for i, row in enumerate(rows):
                out[i*variants:(i+1)*variants, :row['n']] = row['graph'][key]
        graph[key] = out.to(device)
    state = torch.zeros(count*variants, n, 3)
    for i, row in enumerate(rows):
        state[i*variants:(i+1)*variants, :row['n']] = row['states']
    return dict(graph=graph, states=state.to(device),
                Q=torch.stack([r['Q'] for r in rows]).float().to(device),
                auxiliary=torch.stack([r['auxiliary'] for r in rows]).float().to(device),
                n=torch.tensor([r['n'] for r in rows], dtype=torch.float32, device=device),
                radius=torch.tensor([r['pairs'][0]['radius'] for r in rows], device=device))
