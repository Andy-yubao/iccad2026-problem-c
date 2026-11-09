"""Extra token inputs derived exclusively from public solver specifications."""
import numpy as np
import torch
from .batches import inputs_only as original_inputs_only

EXTRA_KEYS = ('terminal_xy', 'terminal_weight', 'terminal_valid', 'terminal_edge_index', 'terminal_edge_weight',
              'constraint_member', 'constraint_type', 'constraint_valid')


def inputs_only(graph):
    return {**original_inputs_only(graph), **{key: graph[key] for key in EXTRA_KEYS if key in graph}}


def one(inputs, scale):
    n, _, _, edges, pins, constraints, _ = inputs
    edges = np.asarray(edges, dtype=float).reshape(-1, 3)
    if len(edges):
        edges = edges[(edges[:, :2] >= 0).all(1) & (edges[:, 2] > 0)]
    active = np.unique(edges[:, 0].astype(int)) if len(edges) else np.zeros(0, dtype=int)
    xy = np.asarray(pins, dtype=float)[active, :2]/float(scale)
    weights = np.zeros((n, len(active)), dtype=np.float32)
    if len(edges):
        columns = np.searchsorted(active, edges[:, 0].astype(int))
        np.add.at(weights, (edges[:, 1].astype(int), columns), edges[:, 2])
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-6)
    c = np.asarray(constraints)
    members = []; kinds = []
    for kind, column in enumerate((2, 3)):
        for group in np.unique(c[:, column]):
            ids = c[:, column] == group
            if group > 0 and ids.sum() > 1:
                members.append(ids.astype(np.float32)); kinds.append(kind)
    return {'xy': xy, 'weights': weights, 'members': members, 'kinds': kinds}


def augment(graph, solver_inputs, scales, device):
    values = [one(inputs, scale) for inputs, scale in zip(solver_inputs, scales)]
    b, n = graph['valid'].shape
    p = max(1, max(len(v['xy']) for v in values)); k = max(1, max(len(v['members']) for v in values))
    extra = {'terminal_xy': torch.zeros(b, p, 2), 'terminal_weight': torch.zeros(b, n, p),
             'terminal_valid': torch.zeros(b, p, dtype=torch.bool),
             'constraint_member': torch.zeros(b, k, n), 'constraint_type': torch.zeros(b, k, dtype=torch.long),
             'constraint_valid': torch.zeros(b, k, dtype=torch.bool)}
    for index, value in enumerate(values):
        nrows = value['weights'].shape[0]; npins = len(value['xy']); ngroups = len(value['members'])
        extra['terminal_xy'][index, :npins] = torch.as_tensor(value['xy'], dtype=torch.float32)
        extra['terminal_weight'][index, :nrows, :npins] = torch.as_tensor(value['weights'])
        extra['terminal_valid'][index, :npins] = True
        if ngroups:
            extra['constraint_member'][index, :ngroups, :nrows] = torch.as_tensor(np.asarray(value['members']))
            extra['constraint_type'][index, :ngroups] = torch.as_tensor(value['kinds'])
            extra['constraint_valid'][index, :ngroups] = True
    edge_index = extra['terminal_weight'].nonzero(as_tuple=False)
    extra['terminal_edge_index'] = edge_index
    extra['terminal_edge_weight'] = extra['terminal_weight'][tuple(edge_index.T)]
    graph.update({key: value.to(device) for key, value in extra.items()})
    return graph
