"""Candidate reference-relative topology metric, not a registered training loss.

Only identities, public areas and centers enter the metric. Layout widths and
heights enter solely to recover their geometric centers. No model/data loader.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TopologyTarget:
    centers: np.ndarray
    scale: np.ndarray
    eligible: np.ndarray
    near: np.ndarray
    ambiguous_membership: np.ndarray
    axes: np.ndarray
    signs: np.ndarray
    unit_delta: np.ndarray
    unit_distance: np.ndarray
    reference_rms: np.ndarray
    k: int
    axis_ratio: float
    tie_rtol: float


def neighbors(distance, eligible, k, tie_rtol):
    """Directed kNN, including all ties at the kth radius; never use IDs to break ties."""
    near = np.zeros_like(eligible)
    for i in range(len(distance)):
        candidates = np.flatnonzero(eligible[i])
        if not len(candidates):
            continue
        count = min(k, len(candidates))
        radius = np.partition(distance[i, candidates], count-1)[count-1]
        near[i, candidates] = distance[i, candidates] <= radius+tie_rtol*max(radius, 1e-12)
    return near


def normalized(centers, scale, eligible):
    delta = (centers[None, :, :]-centers[:, None, :])/scale[..., None]
    distance = np.linalg.norm(delta, axis=-1)
    rms = np.sqrt((distance**2*eligible).sum(1)/np.maximum(eligible.sum(1), 1))
    rms = np.maximum(rms, 1e-8)
    return delta/rms[:, None, None], distance/rms[:, None], distance


def target(reference_centers, public_areas, preplaced=None, k=6, axis_ratio=.1, tie_rtol=1e-6):
    centers = np.asarray(reference_centers, dtype=np.float64)
    areas = np.asarray(public_areas, dtype=np.float64)
    n = len(centers)
    if centers.shape != (n,2) or areas.shape != (n,) or n < 2:
        raise ValueError('Need at least two centers and matching public areas')
    if not np.isfinite(centers).all() or not np.isfinite(areas).all() or not (areas>0).all():
        raise ValueError('Nonfinite centers or invalid public areas')
    if k < 1 or not 0 < axis_ratio <= 1:
        raise ValueError('Invalid metric configuration')
    eligible = ~np.eye(n, dtype=bool)
    if preplaced is not None:
        preplaced = np.asarray(preplaced, bool)
        if preplaced.shape != (n,):
            raise ValueError('Invalid preplaced mask')
        eligible &= ~(preplaced[:, None]&preplaced[None, :])
    if not eligible.any():
        raise ValueError('No eligible non-P-P relations; report metric as unavailable')
    scale = (np.sqrt(areas[:, None])+np.sqrt(areas[None, :]))/2
    delta = centers[None, :, :]-centers[:, None, :]
    dominant = np.abs(delta).max(-1)
    if ((dominant == 0)&eligible).any():
        raise ValueError('Coincident reference centers have undefined directional topology')
    axes = (np.abs(delta) >= axis_ratio*dominant[..., None])&eligible[..., None]
    unit_delta, unit_distance, distance = normalized(centers, scale, eligible)
    reference_rms = np.sqrt((distance**2*eligible).sum(1)/np.maximum(eligible.sum(1), 1))
    near = neighbors(distance, eligible, k, tie_rtol)
    ambiguous = np.zeros_like(eligible)
    for i in range(n):
        quota = min(k,int(eligible[i].sum()))
        if int(near[i].sum()) > quota:
            radius = np.partition(distance[i,eligible[i]],quota-1)[quota-1]
            ambiguous[i] = eligible[i]&(np.abs(distance[i]-radius)<=tie_rtol*max(radius,1e-12))
    return TopologyTarget(centers.copy(), scale, eligible, near, ambiguous, axes, np.sign(delta),
                          unit_delta, unit_distance, reference_rms, k, axis_ratio, tie_rtol)


def balanced_pairs(values, goal):
    per_node = []
    for i in range(len(values)):
        parts = []
        for mask in (goal.near[i], goal.eligible[i]&~goal.near[i]):
            if mask.any():
                parts.append(float(values[i,mask].mean()))
        if parts:
            per_node.append(float(np.mean(parts)))
    return float(np.mean(per_node))


def similarity(predicted_centers, goal):
    centers = np.asarray(predicted_centers, dtype=np.float64)
    if centers.shape != goal.centers.shape or not np.isfinite(centers).all():
        raise ValueError('Invalid predicted centers')
    delta = centers[None, :, :]-centers[:, None, :]
    distance = np.linalg.norm(delta/goal.scale[..., None], axis=-1)
    near = neighbors(distance, goal.eligible, goal.k, goal.tie_rtol)
    correct_axis = goal.signs*delta > 0
    direction = ((~goal.axes)|correct_axis).all(-1)&goal.eligible
    neighborhood = ((near == goal.near)|goal.ambiguous_membership)&goal.eligible
    joint = direction&neighborhood
    return dict(topology_similarity=balanced_pairs(joint,goal),
        direction_match=balanced_pairs(direction,goal),
        neighborhood_match=balanced_pairs(neighborhood,goal),
        reference_near_directed_edges=int(goal.near.sum()),
        predicted_near_directed_edges=int(near.sum()),
        eligible_directed_edges=int(goal.eligible.sum()))


def layout_similarity(predicted_xywh, reference_xywh, public_areas, **kwargs):
    predicted = np.asarray(predicted_xywh, float)
    reference = np.asarray(reference_xywh, float)
    if predicted.shape != reference.shape or predicted.ndim!=2 or predicted.shape[1]!=4:
        raise ValueError('Expected matching N x 4 layouts')
    if not (predicted[:,2:]>0).all() or not (reference[:,2:]>0).all():
        raise ValueError('Invalid rectangle dimensions')
    goal = target(reference[:,:2]+reference[:,2:]/2, public_areas, **kwargs)
    return similarity(predicted[:,:2]+predicted[:,2:]/2, goal)


def surrogate_loss(predicted_centers, goal, margin_fraction=.1, rms_floor_fraction=.05):
    """Unbatched differentiable reference-consistent ordinal surrogate, prototype.

    This is not -T: exact ranks and correctness indicators have zero derivatives
    almost everywhere. Both loss families equal zero at the reference. Query
    RMS normalization prevents reducing the loss by global expansion alone.
    """
    import torch
    p = predicted_centers
    if p.ndim!=2 or tuple(p.shape)!=goal.centers.shape or not 0 < margin_fraction <= 1:
        raise ValueError('Invalid centers/margin')
    tensor = lambda value: torch.as_tensor(value, device=p.device, dtype=p.dtype)
    scale = tensor(goal.scale)
    eligible = torch.as_tensor(goal.eligible,device=p.device)
    delta = (p[None,:,:]-p[:,None,:])/scale[...,None]
    square_distance = delta.square().sum(-1)
    floor = (rms_floor_fraction*tensor(goal.reference_rms)).clamp_min(1e-8)
    mean_square = (square_distance*eligible).sum(1)/eligible.sum(1).clamp_min(1)
    rms = torch.maximum(mean_square,floor.square()).sqrt()
    unit_delta = delta/rms[:,None,None]
    # norm() defines a finite zero subgradient for coincident predicted centers.
    unit_distance = torch.linalg.vector_norm(delta,dim=-1)/rms[:,None]
    axes = torch.as_tensor(goal.axes,device=p.device)
    order_hinge = (margin_fraction*tensor(np.abs(goal.unit_delta))-tensor(goal.signs)*unit_delta).clamp_min(0)
    pair_loss = (order_hinge*axes).sum(-1)/axes.sum(-1).clamp_min(1)
    node_order=[]
    node_neighbor=[]
    for i in range(len(p)):
        parts=[]
        for mask in (goal.near[i],goal.eligible[i]&~goal.near[i]):
            if mask.any():
                parts.append(pair_loss[i,torch.as_tensor(mask,device=p.device)].mean())
        if parts:
            node_order.append(torch.stack(parts).mean())
        positive = np.flatnonzero(goal.near[i]&~goal.ambiguous_membership[i])
        negative = np.flatnonzero(goal.eligible[i]&~goal.near[i])
        if len(positive) and len(negative):
            ref_gap=goal.unit_distance[i,negative][None,:]-goal.unit_distance[i,positive][:,None]
            candidate_gap=unit_distance[i,tensor(negative).long()][None,:]-unit_distance[i,tensor(positive).long()][:,None]
            node_neighbor.append((margin_fraction*tensor(ref_gap)-candidate_gap).clamp_min(0).mean())
    zero=p.sum()*0
    order=torch.stack(node_order).mean() if node_order else zero
    neighborhood=torch.stack(node_neighbor).mean() if node_neighbor else zero
    return dict(total=.5*(order+neighborhood),order=order,neighborhood=neighborhood)
