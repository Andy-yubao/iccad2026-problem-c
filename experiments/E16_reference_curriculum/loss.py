"""Coordinate supervision and label-coordinate-independent quality proxies.

Official discrete Q is used only for evaluation. These continuous quantities
are deliberately named proxies; no gradient through the numerical repair.
"""
import numpy as np
import torch
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
from .model import decode


def weights(arm, epoch, start_epoch=6):
    if start_epoch < 1:
        raise ValueError('Curriculum start epoch must be positive')
    if arm == 'supervised_small':
        return {'supervision': 1., 'quality': 0., 'overlap': 0.}
    phase = min(1., max(0., (epoch - start_epoch + 1) / 15))
    return {'supervision': 1. - .95 * phase, 'quality': .05 + .95 * phase, 'overlap': 1. + 9. * phase}


def geometry_loss(state, graph, targets):
    centers, shapes = [], []
    for i, target in enumerate(targets):
        n = len(target['geometry'])
        mask = graph['mask'][i, :n]
        delta = state[i, :n] - target['geometry']
        centers.append(((delta[:, :2] / graph['area'][i, :n].sqrt()[:, None]).square() * mask[:, :2]).sum()
                       / mask[:, :2].sum().clamp_min(1))
        shapes.append((delta[:, 2].square() * mask[:, 2]).sum() / mask[:, 2].sum().clamp_min(1))
    return torch.stack(centers), torch.stack(shapes)


def quality_terms(positions, graph, targets):
    components = {name: [] for name in ('wire_ratio', 'bbox_ratio', 'boundary', 'grouping', 'mib', 'overlap')}
    for b, target in enumerate(targets):
        n = len(target['geometry'])
        p = positions[b, :n]
        xy, wh = p[:, :2], p[:, 2:]
        upper, center = xy + wh, xy + wh / 2
        zero = p.sum() * 0
        wire = zero
        for name in ('b2b', 'p2b'):
            edge = target[name]
            if len(edge):
                i, j, w = edge[:, 0].long(), edge[:, 1].long(), edge[:, 2]
                left = center[i] if name == 'b2b' else target['pins'][i, :2]
                wire = wire + ((left - center[j]).abs().sum(-1) * w).sum()
        lo, hi = xy.amin(0), upper.amax(0)
        bbox = (hi - lo).prod()
        components['wire_ratio'].append(wire / max(target['wire_anchor'], 1e-8))
        components['bbox_ratio'].append(bbox / max(target['area_anchor'], 1e-8))
        # Count each overlapping pair once; normalize per block, avoiding N^2
        # dilution. Public locked-locked pairs have no trainable coordinates.
        intersection = (torch.minimum(upper[:, None], upper[None]) - torch.maximum(xy[:, None], xy[None])).clamp_min(0)
        intersection_area = intersection.prod(-1)
        area = wh.prod(-1)
        pairs = torch.triu(torch.ones((n, n), device=p.device, dtype=torch.bool), diagonal=1)
        components['overlap'].append((intersection_area / torch.minimum(area[:, None], area[None]).clamp_min(1e-10))[pairs].sum() / n)
        c = graph['constraints'][b, :n]
        requested = (c[:, 4, None] & torch.tensor([1, 2, 4, 8], device=p.device)) != 0
        gaps = torch.stack((xy[:, 0] - lo[0], hi[0] - upper[:, 0],
                            hi[1] - upper[:, 1], xy[:, 1] - lo[1]), -1)
        local = area.sqrt().clamp_min(1e-8)
        boundary = ((gaps / local[:, None]).square() * requested).sum()
        grouping, mib = zero, zero
        for ids in target['groups']:
            if len(ids) < 2:
                continue
            # Any connected touching arrangement is acceptable. Select a
            # current-layout MST, rather than inheriting a teacher contact tree.
            a, u, length = xy[ids], upper[ids], local[ids]
            separation = (torch.maximum(a[:, None], a[None]) - torch.minimum(u[:, None], u[None])).clamp_min(0)
            distance = (separation / torch.minimum(length[:, None], length[None])[..., None]).square().sum(-1)
            detached = distance.detach().cpu().double().numpy() + 1e-9
            np.fill_diagonal(detached, 0)
            tree = minimum_spanning_tree(csr_matrix(detached)).tocoo()
            grouping = grouping + distance[tree.row, tree.col].sum()
        for ids in target['mibs']:
            if len(ids) < 2:
                continue
            logs = wh[ids].log()
            mib = mib + (logs - logs.mean(0)).square().mean() * (len(ids) - 1)
        nsoft = int((c[:, 4] != 0).sum()) + sum(max(0, len(ids)-1) for ids in target['groups'] + target['mibs'])
        for name, value in (('boundary', boundary), ('grouping', grouping), ('mib', mib)):
            components[name].append(value / max(1, nsoft))
    return {name: torch.stack(values) for name, values in components.items()}


def losses(state, graph, targets, arm, epoch, *, include_quality=True, start_epoch=6):
    center, shape = geometry_loss(state, graph, targets)
    supervision = center + shape
    if not include_quality:
        if arm != 'supervised_small':
            raise ValueError('Quality objectives cannot be disabled for curriculum training')
        return {'total': supervision.mean(), 'supervision': supervision.mean(),
                'center_mse': center.mean(), 'shape_mse': shape.mean()}
    terms = quality_terms(decode(state, graph), graph, targets)
    # Ratios deliberately stay unclipped: they can reward improvement beyond
    # a stored layout, whereas official Q has floors and discrete penalties.
    quality = .5 * (terms['wire_ratio'] + terms['bbox_ratio']) + 2 * (terms['boundary'] + terms['grouping'] + terms['mib'])
    w = weights(arm, epoch, start_epoch)
    total = (w['supervision'] * supervision + w['quality'] * quality + w['overlap'] * terms['overlap']).mean()
    return {'total': total, 'supervision': supervision.mean(), 'center_mse': center.mean(),
            'shape_mse': shape.mean(), 'quality_proxy': quality.mean(),
            **{k: v.mean() for k, v in terms.items()}}
