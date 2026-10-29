"""Pack existing public/supervision tensors once; batch membership/order unchanged."""
import torch


class ResidentRows:
    def __init__(self, rows, device, max_nodes=None):
        self.rows = rows
        self.device = torch.device(device)
        self.max_nodes = max_nodes or max(r['graph']['n'] for r in rows)
        self.indices = {id(row): i for i, row in enumerate(rows)}
        self.tensors = {}
        for key in ('features', 'mask', 'area', 'known', 'target', 'specs', 'reference', 'constraints',
                    'adjacency', 'mib', 'group'):
            first = rows[0]['graph'][key]
            shape = ((len(rows), self.max_nodes, self.max_nodes) if key in ('adjacency', 'mib', 'group') else
                     (len(rows), self.max_nodes)+first.shape[1:])
            value = torch.zeros(shape, dtype=first.dtype)
            for i, row in enumerate(rows):
                n = row['graph']['n']
                if key in ('adjacency', 'mib', 'group'): value[i, :n, :n] = row['graph'][key]
                else: value[i, :n] = row['graph'][key]
            self.tensors[key] = value.to(self.device)
        self.tensors['scale'] = torch.stack([r['graph']['scale'] for r in rows]).to(self.device)
        self.tensors['valid'] = (torch.arange(self.max_nodes)[None] <
                                torch.tensor([r['graph']['n'] for r in rows])[:, None]).to(self.device)
        self.bytes = sum(t.numel()*t.element_size() for t in self.tensors.values())

    def batch(self, indices, out=None):
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        if out is None:
            return {k: torch.index_select(v, 0, indices) for k, v in self.tensors.items()}
        for k, v in self.tensors.items(): torch.index_select(v, 0, indices, out=out[k])
        return out

    def row_indices(self, rows):
        return [self.indices[id(r)] for r in rows]
