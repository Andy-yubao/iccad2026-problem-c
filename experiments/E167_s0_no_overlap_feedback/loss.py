"""E166 L4 exit objective for E167 S0 training."""
import torch
from .base_loss import B, TOTAL_COMPONENTS, base_per_case
from .blockage import components
from .loss_config import ARM, BLOCK_SCALE, EXIT_WEIGHTS, OVERLAP_SCALE


def blockage_weight(graph, dtype):
    valid = graph['valid']
    pairs = valid[:, :, None] & valid[:, None, :]
    ids = torch.arange(valid.shape[1], device=valid.device)
    peers = ((graph['group'] > 0) & pairs & (ids[:, None] != ids[None, :])).sum(-1).to(dtype)
    grouped = (peers / (peers + 1)).sum(-1)
    boundary = ((graph['constraints'][..., 4] != 0) & valid).sum(-1)
    return 2 * B * BLOCK_SCALE * grouped / (grouped + boundary).clamp_min(1)


def per_case(states, graph, phase):
    if phase != ARM or states.shape[0] != 5:
        raise ValueError('Expected E167 S0 and the L0–L4 readouts')
    terms, boxes, tree = base_per_case(states[4], graph)
    base_full_total = terms['total']
    terms['overlap_weighted'] = OVERLAP_SCALE[4] * terms['overlap_weighted']
    terms.update(components(boxes, tree, graph))
    terms['group_block_weighted'] = blockage_weight(graph, states.dtype) * terms['group_block']
    final_total = sum(terms[key] for key in TOTAL_COMPONENTS) + terms['group_block_weighted']
    fields = (*TOTAL_COMPONENTS, 'group_block_weighted', 'group_block_edges', 'group_block_coverage')
    out = {f'l4_{key}': terms[key] for key in fields}
    return dict(out, l4_total=final_total, total=EXIT_WEIGHTS[4] * final_total,
                final_base_full_total=base_full_total, final_total=final_total)
