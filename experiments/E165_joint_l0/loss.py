"""E151's exit objective plus E161's selected-MST blockage and L0 anchor."""
import torch
from .base_loss import B, TOTAL_COMPONENTS, base_per_case
from .blockage import components
from .loss_config import ARMS, BLOCK_SCALE, EXIT_WEIGHTS, L0_AUX_WEIGHT, OVERLAP_SCALE


def blockage_weight(graph, dtype):
    valid = graph['valid']
    pairs = valid[:, :, None] & valid[:, None, :]
    ids = torch.arange(valid.shape[1], device=valid.device)
    peers = ((graph['group'] > 0) & pairs & (ids[:, None] != ids[None, :])).sum(-1).to(dtype)
    grouped = (peers / (peers + 1)).sum(-1)
    boundary = ((graph['constraints'][..., 4] != 0) & valid).sum(-1)
    return 2 * B * BLOCK_SCALE * grouped / (grouped + boundary).clamp_min(1)


def exit_loss(state, graph, index, weight):
    terms, boxes, tree = base_per_case(state, graph)
    terms['base_full_total'] = terms['total']
    terms['overlap_weighted'] = OVERLAP_SCALE[index] * terms['overlap_weighted']
    terms.update(components(boxes, tree, graph))
    terms['group_block_weighted'] = weight * terms['group_block']
    terms['total'] = sum(terms[key] for key in TOTAL_COMPONENTS) + terms['group_block_weighted']
    return terms


def per_case(states, graph, phase):
    if phase not in ARMS or states.shape[0] != (1 if phase == ARMS[0] else 13):
        raise ValueError('Unexpected E165 readouts')
    weight = blockage_weight(graph, states.dtype)
    initial = exit_loss(states[0], graph, 0, weight)
    if phase == ARMS[0]:
        return initial
    exits = {index: exit_loss(states[index], graph, index, weight) for index in EXIT_WEIGHTS}
    out = {f'l{index}_{key}': row[key]
           for index, row in {0: initial, **exits}.items()
           for key in ('total', *TOTAL_COMPONENTS, 'group_block_weighted', 'group_block_edges', 'group_block_coverage')}
    joint = sum(EXIT_WEIGHTS[index] * row['total'] for index, row in exits.items())
    return dict(out, total=joint + L0_AUX_WEIGHT * initial['total'],
                final_base_full_total=exits[12]['base_full_total'],
                final_total=exits[12]['total'])
