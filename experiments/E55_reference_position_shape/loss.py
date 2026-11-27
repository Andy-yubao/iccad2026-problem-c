"""Only reference center and raw log-aspect supervision; no geometric penalties."""
import torch
import torch.nn.functional as F
from .loss_config import configuration


def weights(arm, epoch, start_epoch=None):
    return configuration(arm)['weights']


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=None):
    w = configuration(arm)['weights']
    target, mask = graph['target'], graph['mask']
    local = graph['area'].clamp_min(1e-12).sqrt()
    center_errors = F.smooth_l1_loss(state[..., :2]/local[..., None],
                                   target[..., :2]/local[..., None], reduction='none', beta=.1)
    center = (center_errors*mask[..., :2]).sum((1, 2))/mask[..., :2].sum((1, 2)).clamp_min(1)
    shape_errors = (state[..., 2]-target[..., 2]).square()
    shape = (shape_errors*mask[..., 2]).sum(1)/mask[..., 2].sum(1).clamp_min(1)
    # Form per-layout totals first; monitoring preserves this exact aggregation.
    total = (w['center_local']*center+w['shape']*shape).mean()
    return {'total': total, 'center_local': center.mean(), 'shape': shape.mean(),
            'supervision': total}
