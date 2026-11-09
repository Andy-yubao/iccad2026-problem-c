"""Exact pair-intersection area for training; no reference positions or grid."""
import torch
from .batches import decode

AREA_WEIGHT = .25


def overlap_area(state, graph):
    """Sum unordered mutable pair areas / total valid input block area.

    Same quantity as occupancy.overlap_area, without its grid/void computation.
    Includes free-preplaced collisions; excludes immutable-immutable pairs.
    Zero-gap contacts incur zero penalty, with no added clearance requirement.
    """
    p = decode(state, graph)
    xy, hi = p[..., :2], p[..., :2] + p[..., 2:]
    inter = (torch.minimum(hi[:, :, None], hi[:, None, :]) -
             torch.maximum(xy[:, :, None], xy[:, None, :])).clamp_min(0).prod(-1)
    valid = graph['valid']
    mutable = graph['mask'][..., 0].bool()
    n = state.shape[1]
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs = pairs & ~torch.eye(n, device=state.device, dtype=torch.bool)[None]
    pairs = pairs & (mutable[:, :, None] | mutable[:, None, :])
    total = (graph['area'] * valid).sum(1).clamp_min(1e-8)
    return (inter * pairs).sum((1, 2)) / (2 * total)
