"""Per-module coverage burden with sublinear area and continuous count weights.

Only public input area and decoded rectangles are used. Pair sums deliberately
retain multiplicity; these loads are not union-covered fractions.
"""
import torch
from .batches import decode
from .ops import product2

LOSS_WEIGHT = .25
AREA_EXPONENT = .5
COUNT_TAU = .05
COUNT_STRENGTH = 1.


def definition():
    return {'weight_controlled_by':'loss_config.configuration(arm)', 'area_exponent': AREA_EXPONENT,
            'count_tau': COUNT_TAU, 'count_strength': COUNT_STRENGTH,
            'area_weight': 'sqrt(public input area), normalized by sum of node weights',
            'load': 'sum_j intersection(i,j) / public input area(i); multiplicity retained',
            'soft_count': 'sum_j r_ij / (r_ij + .05)',
            'count_multiplier': '1 + soft_count / (1 + soft_count); in [1,2)',
            'objective': '.5 * area-weighted mean(load * count_multiplier)',
            'masks': 'exclude padding/self/immutable-immutable pairs; include free-preplaced pairs',
            'selection': 'same fixed total as training; all splits and epochs share one definition'}


def from_rectangles(rectangles, area, valid, mutable):
    """Return one value per layout; area weights cannot be changed by the model."""
    xy, hi = rectangles[..., :2], rectangles[..., :2] + rectangles[..., 2:]
    intersection = product2((torch.minimum(hi[:, :, None], hi[:, None, :]) -
                            torch.maximum(xy[:, :, None], xy[:, None, :])).clamp_min(0))
    n = rectangles.shape[1]
    pairs = (valid[:, :, None] & valid[:, None, :] &
             ~torch.eye(n, dtype=torch.bool, device=rectangles.device)[None] &
             (mutable[:, :, None] | mutable[:, None, :]))
    relative = intersection * pairs / area.clamp_min(1e-12)[:, :, None]
    load = relative.sum(-1)
    count = (relative / (relative + COUNT_TAU)).sum(-1)
    factor = 1. + COUNT_STRENGTH * count / (1. + count)
    node_weights = area.clamp_min(1e-12).pow(AREA_EXPONENT) * valid
    denominator = node_weights.sum(-1).clamp_min(1e-12)
    base = .5 * (node_weights * load).sum(-1) / denominator
    counted = .5 * (node_weights * load * factor).sum(-1) / denominator
    node_denominator = valid.sum(-1).clamp_min(1)
    return {'module_overlap_base': base, 'module_overlap_counted': counted,
            'module_count_extra': counted - base,
            'module_load_mean': (load * valid).sum(-1) / node_denominator,
            'module_load_max': load.masked_fill(~valid, 0).amax(-1),
            'module_load_gt05_fraction': ((load > .05) & valid).sum(-1) / node_denominator,
            'module_soft_count_mean': (count * valid).sum(-1) / node_denominator,
            'module_count_factor_mean': (factor * valid).sum(-1) / node_denominator}


def components(state, graph):
    return from_rectangles(decode(state, graph), graph['area'], graph['valid'],
                           graph['mask'][..., 0].bool())
