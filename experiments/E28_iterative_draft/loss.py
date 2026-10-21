"""Public geometry energy, exact weighted wire proxy, weak compatible anchors."""
import torch
import torch.nn.functional as F
from .base_model import decode

WEIGHTS = {'penetration': 2., 'overlap_area': 2., 'crowding': .25, 'contact': .5,
           'boundary': .5, 'mib': .25, 'wire_ratio': .1, 'bbox_ratio': .1}


def average(value, mask, dims):
    return (value * mask).sum(dims) / mask.sum(dims).clamp_min(1)


def wire(boxes, graph):
    center = boxes[..., :2] + boxes[..., 2:] / 2
    internal = ((center[:, :, None] - center[:, None]).abs().sum(-1) * graph['block_weights']).sum((1, 2)) / 2
    external = ((center[:, :, None] - graph['pin_xy'][:, None]).abs().sum(-1) * graph['pin_weights']).sum((1, 2))
    return internal + external


def components(state, graph):
    boxes = decode(state, graph); valid = graph['valid']; n = state.shape[1]
    area = graph['area']; local = area.sqrt().clamp_min(1e-6)
    pairs = valid[:, :, None] & valid[:, None, :]
    pairs = pairs & ~torch.eye(n, device=state.device, dtype=torch.bool)[None]
    mutable = graph['mask'][..., 0].bool()
    train_pairs = pairs & (mutable[:, :, None] | mutable[:, None, :])
    scale = ((local[:, :, None] + local[:, None, :]) / 2).clamp_min(1e-6)
    xy, wh = boxes[..., :2], boxes[..., 2:]; hi = xy + wh; center = xy + wh / 2
    delta = center[:, :, None] - center[:, None, :]
    margin = delta.abs() - (wh[:, :, None] + wh[:, None, :]) / 2
    penetration = (-margin.amax(-1)).clamp_min(0) / scale
    count = valid.sum(1).clamp_min(1); total_area = (area * valid).sum(1).clamp_min(1e-8)
    intersection = (torch.minimum(hi[:, :, None], hi[:, None]) - torch.maximum(xy[:, :, None], xy[:, None])).clamp_min(0).prod(-1)
    overlap_area = (intersection * train_pairs).sum((1, 2)) / (2 * total_area)
    load = (intersection * train_pairs / area[:, :, None].clamp_min(1e-8)).sum(-1)
    tx = (margin[..., 0] / scale).square() + (margin[..., 1] / scale + .01).clamp_min(0).square()
    ty = (margin[..., 1] / scale).square() + (margin[..., 0] / scale + .01).clamp_min(0).square()
    lo = xy.masked_fill(~valid[..., None], 1e4).amin(1)
    upper = hi.masked_fill(~valid[..., None], -1e4).amax(1)
    side = torch.stack((xy[..., 0] - lo[:, None, 0], upper[:, None, 0] - hi[..., 0],
                        upper[:, None, 1] - hi[..., 1], xy[..., 1] - lo[:, None, 1]), -1)
    required = ((graph['constraints'][..., 4, None] & torch.tensor([1, 2, 4, 8], device=state.device)) != 0) & valid[..., None]
    mib_pairs = (graph['mib'] > 0) & pairs
    shape_diff = (wh.clamp_min(1e-8).log()[:, :, None] - wh.clamp_min(1e-8).log()[:, None]).square().mean(-1)
    initial_boxes = decode(graph['draft'], graph).detach()
    values = {
        'penetration': (penetration.square() * train_pairs).sum((1, 2)) / count,
        'overlap_area': overlap_area,
        'crowding': ((load - .5).clamp_min(0).square() * valid).sum(1) / count,
        'contact': average(torch.minimum(tx, ty), graph['contact_tree'], (1, 2)),
        'boundary': average(F.smooth_l1_loss(side / local[..., None], torch.zeros_like(side), reduction='none', beta=.1), required, (1, 2)),
        'mib': average(shape_diff, mib_pairs, (1, 2)),
        'wire_ratio': wire(boxes, graph) / wire(initial_boxes, graph).clamp_min(1e-6),
        'bbox_ratio': (upper - lo).prod(-1) / total_area,
        'center_local': average(F.smooth_l1_loss(state[..., :2] / local[..., None], graph['target'][..., :2] / local[..., None], reduction='none', beta=.1), graph['mask'][..., :2] * valid[..., None], (1, 2)),
        'shape_anchor': average((state[..., 2] - graph['target'][..., 2]).square(), graph['shape_anchor_mask'] * graph['mask'][..., 2] * valid, (1,)),
    }
    values['physics'] = sum(WEIGHTS[key] * values[key] for key in WEIGHTS)
    values['total'] = values['physics'] + .1 * values['center_local'] + .025 * values['shape_anchor']
    return values


def loss(state, graph):
    values = components(state, graph)
    return {key: value.mean() for key, value in values.items()}
