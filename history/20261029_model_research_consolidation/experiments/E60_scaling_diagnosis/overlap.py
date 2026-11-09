"""Raw relative separation lengths; aggregation stays with the experiment Loss.

The centerline branch uses its actual derivative. Its scalar is not monotone
under every componentwise outward displacement; see check_overlap evidence.
"""
import torch


def axis_distance(delta, extent):
    gap = extent - delta.abs()
    return torch.minimum(gap[..., 0], gap[..., 1]).clamp_min(0)


def centerline_distance(delta, extent, pair_scale, near_zero=1e-4):
    absolute = delta.abs()
    radius2 = delta.square().sum(-1)
    epsilon = (pair_scale.clamp_min(1e-8) * near_zero).clamp_min(1e-12)
    # The inactive near-coincident branch must also have finite derivatives.
    radius = radius2.clamp_min(epsilon.square()).sqrt()
    direction = absolute / radius[..., None]
    gaps = extent - absolute
    exits = gaps / direction.clamp_min(1e-12)
    radial = torch.minimum(exits[..., 0], exits[..., 1]).clamp_min(0)
    regular = radius2 >= epsilon.square()
    return torch.where(regular, radial, axis_distance(delta, extent))


def overlap_per_layout(delta, extent, pair_scale, pairs, valid, kind):
    if kind == 'original':
        distance = axis_distance(delta, extent)
    elif kind == 'centerline_v1':
        distance = centerline_distance(delta, extent, pair_scale)
    else:
        raise ValueError(kind)
    penetration = distance / pair_scale.clamp_min(1e-8)
    return (penetration.square() * pairs).sum((1, 2)) / valid.sum(1).clamp_min(1)
