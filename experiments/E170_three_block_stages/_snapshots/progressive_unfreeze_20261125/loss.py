"""E170 per-outlet supervision over the registered active phase objective."""
import torch
from .base_loss import B, TOTAL_COMPONENTS, base_per_case
from .blockage import components
from .loss_config import (BLOCK_SCALE, OVERLAP_SCALE, REFERENCE_SCALE,
                          SOFT_COUNT_SCALE, SOFT_COUNT_TAU, exit_weights)


def blockage_weight(graph, dtype):
    valid = graph['valid']
    pairs = valid[:, :, None] & valid[:, None, :]
    ids = torch.arange(valid.shape[1], device=valid.device)
    peers = ((graph['group'] > 0) & pairs & (ids[:, None] != ids[None, :])).sum(-1).to(dtype)
    grouped = (peers / (peers + 1)).sum(-1)
    boundary = ((graph['constraints'][..., 4] != 0) & valid).sum(-1)
    return 2 * B * BLOCK_SCALE * grouped / (grouped + boundary).clamp_min(1)


def exit_loss(state, graph, index, weight):
    count_scale = SOFT_COUNT_SCALE[index]
    terms, boxes, tree = base_per_case(state, graph, soft_count_tau=SOFT_COUNT_TAU if count_scale else None)
    terms['base_full_total'] = terms['total']
    terms['reference_weighted'] = REFERENCE_SCALE[index] * terms['reference_weighted']
    terms['overlap_weighted'] = OVERLAP_SCALE[index] * terms['overlap_weighted']
    terms.update(components(boxes, tree, graph))
    terms['group_block_weighted'] = weight * terms['group_block']
    terms['total'] = sum(terms[key] for key in TOTAL_COMPONENTS) + terms['group_block_weighted']
    if count_scale:
        terms['group_count_weighted'] = count_scale * terms['group_count_proxy']
        terms['boundary_count_weighted'] = count_scale * terms['boundary_count_proxy']
        terms['total'] = terms['total'] + terms['group_count_weighted'] + terms['boundary_count_weighted']
    return terms


def per_case(states, graph, stage, joint):
    if states.shape[0] != stage + 1:
        raise ValueError('Training output does not match the registered stage')
    weight = blockage_weight(graph, states.dtype)
    weights = exit_weights(stage, joint)
    exits = {i: exit_loss(states[i], graph, i, weight) for i in weights}
    fields = ('total', *TOTAL_COMPONENTS, 'group_block_weighted', 'group_block_edges',
              'group_block_coverage', 'group_count_proxy', 'boundary_count_proxy',
              'group_count_weighted', 'boundary_count_weighted')
    rows = {f's{i}_{key}': row[key] for i, row in exits.items()
            for key in fields if key in row}
    return dict(rows, total=sum(weights[i] * row['total'] for i, row in exits.items()),
                final_base_full_total=exits[stage]['base_full_total'],
                final_total=exits[stage]['total'])
