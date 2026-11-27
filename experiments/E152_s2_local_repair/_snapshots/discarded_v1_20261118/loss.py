"""Matched L12 objectives with a tolerant displacement budget relative to L8."""
import torch
import torch.nn.functional as F

from .base_loss import TOTAL_COMPONENTS, base_per_case
from .batches import decode
from .loss_config import (CENTER_TOLERANCE, DRIFT_WEIGHT, HUBER_BETA, OVERLAP_DRIFT_WEIGHT,
                          OVERLAP_THRESHOLD, RATIO_TOLERANCE, RATIO_WEIGHT, REF_WEIGHT)
from .simple_loss import PUBLIC, public_components


def incoming_module_overlap(state, batch):
    """Each module's cumulative overlap area divided by its own public area."""
    p = decode(state, batch)
    lo, hi = p[..., :2], p[..., :2] + p[..., 2:]
    intersection = (torch.minimum(hi[:, :, None], hi[:, None, :])
                    - torch.maximum(lo[:, :, None], lo[:, None, :])).clamp_min(0).prod(-1)
    indices = torch.arange(p.shape[1], device=p.device)
    valid = batch['valid']
    pair = valid[:, :, None] & valid[:, None, :] & (indices[:, None] != indices[None, :])
    return (intersection * pair).sum(-1) / batch['area'].clamp_min(1e-12)


def drift_terms(incoming, outgoing, batch):
    incoming = incoming.detach()
    displacement = torch.linalg.vector_norm(outgoing[..., :2] - incoming[..., :2], dim=-1)
    center_excess = (displacement / batch['area'].clamp_min(1e-12).sqrt() - CENTER_TOLERANCE).clamp_min(0)
    shape_excess = ((outgoing[..., 2] - incoming[..., 2]).abs() - RATIO_TOLERANCE).clamp_min(0)
    weights = torch.where(incoming_module_overlap(incoming, batch) > OVERLAP_THRESHOLD, OVERLAP_DRIFT_WEIGHT, 1.)
    valid = batch['valid']
    constraints = batch['constraints']
    free_centers = valid & (constraints[..., 1] == 0)
    free_shapes = free_centers & (constraints[..., 0] == 0)
    losses = []
    for excess, eligible in ((center_excess, free_centers), (shape_excess, free_shapes)):
        penalty = F.smooth_l1_loss(excess, torch.zeros_like(excess), beta=HUBER_BETA, reduction='none')
        losses.append((penalty * weights * eligible).sum(-1) / eligible.sum(-1).clamp_min(1))
    return losses


def per_case(states, batch, phase, diagnostics=False):
    if states.shape[0] != 13:
        raise ValueError('Expected L0–L12 readouts')
    full = base_per_case(states[12], batch)
    reference = full['reference_weighted']
    public = {key: full[key] for key in TOTAL_COMPONENTS if key != 'reference_weighted'}
    task = sum(public.values())
    center, ratio = drift_terms(states[8], states[12], batch)
    drift = center + RATIO_WEIGHT * ratio
    total = task + DRIFT_WEIGHT * drift
    if REF_WEIGHT[phase]:
        total = total + reference
    out = dict(total=total, task=task, drift=drift, drift_center=center, drift_ratio=ratio, **public)
    if diagnostics:
        public_inputs = {key: batch[key] for key in PUBLIC}
        out.update(reference=reference, original_full_L12=full['total'])
        for layer in (8, 10, 12):
            costs = full if layer == 12 else public_components(states[layer], public_inputs)
            out[f'L{layer}_overlap'] = costs['overlap_total']
            out[f'L{layer}_max_module_overlap'] = costs['module_load_max']
    return out
