"""E165's joint exit objective with count proxies for grouping and boundary."""
import torch
from .base_loss import B, TOTAL_COMPONENTS, base_per_case
from .blockage import components
from .loss_config import (ARM, BLOCK_SCALE, EXIT_WEIGHTS, L0_AUX_WEIGHT,
                          OVERLAP_SCALE, REFERENCE_SCALE, SOFT_COUNT_SCALE, SOFT_COUNT_TAU)


def blockage_weight(graph, dtype):
    valid = graph['valid']
    pairs = valid[:, :, None] & valid[:, None, :]
    ids = torch.arange(valid.shape[1], device=valid.device)
    peers = ((graph['group'] > 0) & pairs & (ids[:, None] != ids[None, :])).sum(-1).to(dtype)
    grouped = (peers / (peers + 1)).sum(-1)
    boundary = ((graph['constraints'][..., 4] != 0) & valid).sum(-1)
    return 2 * B * BLOCK_SCALE * grouped / (grouped + boundary).clamp_min(1)


def exit_loss(state, graph, index, weight):
    count_scale = SOFT_COUNT_SCALE.get(index)
    terms, boxes, tree = base_per_case(state, graph, soft_count_tau=SOFT_COUNT_TAU if count_scale else None)
    terms['base_full_total'] = terms['total']
    terms['reference_weighted'] = REFERENCE_SCALE[index] * terms['reference_weighted']
    terms['overlap_weighted'] = OVERLAP_SCALE[index] * terms['overlap_weighted']
    terms.update(components(boxes, tree, graph))
    terms['group_block_weighted'] = weight * terms['group_block']
    terms['total'] = sum(terms[key] for key in TOTAL_COMPONENTS) + terms['group_block_weighted']
    if count_scale is not None:
        terms['group_count_weighted'] = count_scale * terms['group_count_proxy']
        terms['boundary_count_weighted'] = count_scale * terms['boundary_count_proxy']
        terms['total'] = terms['total'] + terms['group_count_weighted'] + terms['boundary_count_weighted']
    return terms


def per_case(states, graph, phase):
    if phase != ARM or states.shape[0] != 13:
        raise ValueError('Expected the E166 backend and 13 readouts')
    weight = blockage_weight(graph, states.dtype)
    initial = exit_loss(states[0], graph, 0, weight)
    exits = {index: exit_loss(states[index], graph, index, weight) for index in EXIT_WEIGHTS}
    out = {f'l{index}_{key}': row[key]
           for index, row in {0: initial, **exits}.items()
           for key in ('total', *TOTAL_COMPONENTS, 'group_block_weighted', 'group_block_edges', 'group_block_coverage')}
    joint = sum(EXIT_WEIGHTS[index] * row['total'] for index, row in exits.items())
    for index in SOFT_COUNT_SCALE:
        for key in ('group_count_proxy', 'boundary_count_proxy', 'group_count_weighted', 'boundary_count_weighted'):
            out[f'l{index}_{key}'] = exits[index][key]
    return dict(out, total=joint + L0_AUX_WEIGHT * initial['total'],
                final_base_full_total=exits[12]['base_full_total'],
                final_total=exits[12]['total'])
