"""Detached trial-shape responses using same-group contact neighbors."""

from dataclasses import dataclass
import torch

DELTAS = (.2, -.2, .8, -.8)
METRICS = ('overlap', 'group_contact', 'boundary', 'bbox')
FEATURE_NAMES = tuple(f'shape_{d:+.1f}_{metric}' for d in DELTAS for metric in METRICS)
_BITS = {}


@dataclass(frozen=True)
class ShapeFeedbackContext:
    batch_index: torch.Tensor
    neighbor_index: torch.Tensor
    same_group: torch.Tensor
    pairs: torch.Tensor
    valid: torch.Tensor
    area: torch.Tensor
    local: torch.Tensor
    scale_squared: torch.Tensor
    required: torch.Tensor
    total_area: torch.Tensor


def max_same_group_neighbors(rows):
    """Determine a fixed width for the complete training and monitoring caches."""
    maximum = 1
    for row in rows:
        group = row['graph']['group']
        count = (group > 0).sum(-1) - (group.diagonal() > 0).to(torch.int64)
        maximum = max(maximum, int(count.max()))
    return maximum


@torch.no_grad()
def prepare_shape_feedback(graph, neighbor_count=None):
    """Prepare layout-independent tensors once per model forward."""
    valid = graph['valid']
    batch_size, n = valid.shape
    ids = torch.arange(n, device=valid.device)
    pairs = valid[:, :, None] & valid[:, None, :] & (ids[:, None] != ids[None, :])
    same_group = (graph['group'] > 0) & pairs
    # CUDA Graph replay requires a fixed top-k width. The trainer computes it
    # from all resident rows. Standalone calls can accept any group size.
    width = n if neighbor_count is None else min(neighbor_count, n)
    neighbors = same_group.to(torch.int32).topk(width, dim=-1, sorted=False).indices
    batch = torch.arange(batch_size, device=valid.device)[:, None, None]
    area = graph['area'].clamp_min(1e-12)
    local = area.sqrt()
    scale_squared = ((local[:, :, None] + local[batch, neighbors]) / 2).clamp_min(1e-6).square()
    constraints = graph['constraints']
    key = (str(valid.device), constraints.dtype)
    if key not in _BITS:
        _BITS[key] = torch.tensor((1, 2, 4, 8), device=valid.device, dtype=constraints.dtype)
    required = ((constraints[..., 4, None] & _BITS[key]) != 0) & valid[..., None]
    return ShapeFeedbackContext(
        batch, neighbors, same_group.gather(-1, neighbors), pairs, valid, area,
        local, scale_squared, required, (area * valid).sum(-1).clamp_min(1e-12),
    )


def other_extrema(coordinates, valid, least):
    """Extremum of every axis after excluding the current rectangle."""
    fill = torch.inf if least else -torch.inf
    extrema = torch.min if least else torch.max
    reduce = torch.amin if least else torch.amax
    masked = coordinates.masked_fill(~valid[..., None], fill)
    first, winner = extrema(masked, dim=1, keepdim=True)
    ids = torch.arange(coordinates.shape[1], device=coordinates.device)[None, :, None]
    first_here = winner == ids
    second = reduce(masked.masked_fill(first_here, fill), dim=1, keepdim=True)
    return torch.where(first_here, second, first).masked_fill(~valid[..., None], fill)


@torch.no_grad()
def shape_features(p, context, overlap_feedback=True):
    lo = p[..., :2]
    wh = p[..., 2:]
    hi = lo + wh
    center = lo + wh / 2
    other_lo = other_extrema(lo, context.valid, least=True)
    other_hi = other_extrema(hi, context.valid, least=False)

    batch, neighbors = context.batch_index, context.neighbor_index
    near_lo, near_hi, near_wh = lo[batch, neighbors], hi[batch, neighbors], wh[batch, neighbors]
    has_group = context.same_group.any(-1)

    def measurements(qlo, qwh):
        qhi = qlo + qwh
        if overlap_feedback:
            signed = torch.minimum(qhi[:, :, None], hi[:, None]) - torch.maximum(qlo[:, :, None], lo[:, None])
            overlap = ((signed.clamp_min(0).prod(-1) * context.pairs).sum(-1) / context.area).log1p()
        else:
            overlap = torch.zeros_like(context.area)
        signed = torch.minimum(qhi[:, :, None], near_hi) - torch.maximum(qlo[:, :, None], near_lo)
        fx = torch.minimum((qhi[:, :, None, 0] - near_lo[..., 0]).square(),
                           (near_hi[..., 0] - qlo[:, :, None, 0]).square())
        fy = torch.minimum((qhi[:, :, None, 1] - near_lo[..., 1]).square(),
                           (near_hi[..., 1] - qlo[:, :, None, 1]).square())
        ty = (.01 * torch.minimum(qwh[:, :, None, 1], near_wh[..., 1]) - signed[..., 1]).clamp_min(0).square()
        tx = (.01 * torch.minimum(qwh[:, :, None, 0], near_wh[..., 0]) - signed[..., 0]).clamp_min(0).square()
        contact = torch.minimum((fx + ty) / context.scale_squared, (fy + tx) / context.scale_squared)
        contact = contact.masked_fill(~context.same_group, torch.inf).amin(-1)
        contact = torch.where(has_group, contact, torch.zeros_like(contact))
        bblo = torch.minimum(qlo, other_lo)
        bbhi = torch.maximum(qhi, other_hi)
        sides = torch.stack((qlo[..., 0] - bblo[..., 0], bbhi[..., 0] - qhi[..., 0],
                             bbhi[..., 1] - qhi[..., 1], qlo[..., 1] - bblo[..., 1]), -1) / context.local[..., None]
        boundary = (sides * context.required).sum(-1) / context.required.sum(-1).clamp_min(1)
        bbox = (bbhi - bblo).prod(-1) / context.total_area[:, None]
        return torch.stack((overlap, contact, boundary, bbox), -1)

    base = measurements(lo, wh)
    ratio = (wh[..., 0] / wh[..., 1].clamp_min(1e-12)).log()
    responses = []
    for delta in DELTAS:
        r = (ratio + delta).clamp(-5, 5)
        trial_wh = torch.stack(((context.area * r.exp()).clamp_min(1e-12).sqrt(),
                                (context.area / r.exp()).clamp_min(1e-12).sqrt()), -1)
        trial_lo = (center - trial_wh / 2).clamp_min(0)
        responses.append((measurements(trial_lo, trial_wh) - base).asinh())
    return torch.cat(responses, -1) * context.valid[..., None]
