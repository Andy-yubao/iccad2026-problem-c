"""E6 input-only features, initially copied from E1 on 2026-10-17.
Independent maintenance; only datasets is shared.
"""
import torch

def graph_features(inputs, device="cpu"):
    n, areas, b2b, p2b, pins, constraints, specs = inputs
    areas, b2b, p2b, pins, c, s = [torch.as_tensor(v, device=device, dtype=torch.float32)
                                   for v in (areas, b2b, p2b, pins, constraints, specs)]
    n = int(n)
    if areas.shape != (n,) or c.shape[0] != n or s.shape != (n, 4) or (areas <= 0).any():
        raise ValueError("Invalid solver tuple")
    if not all(torch.isfinite(v).all() for v in (areas, b2b, p2b, pins, c, s)):
        raise ValueError("Nonfinite solver input")
    # Official rows are unpadded in the observed member. Accept only explicit
    # all-minus-one edge padding; reject negative, fractional or out-of-range IDs.
    cleaned = []
    for edges, first_limit in ((b2b, n), (p2b, len(pins))):
        if edges.ndim != 2 or edges.shape[1] != 3:
            raise ValueError("Expected edge triples")
        edges = edges[~(edges == -1).all(1)]
        ids = edges[:, :2]
        if ((ids < 0).any() or (ids != ids.long()).any() or (ids[:, 0] >= first_limit).any()
                or (ids[:, 1] >= n).any() or (edges[:, 2] < 0).any()):
            raise ValueError("Invalid edge endpoint or weight")
        cleaned.append(edges)
    b2b, p2b = cleaned
    fixed, pre = c[:, 0] != 0, c[:, 1] != 0
    locked = fixed | pre
    if (s[locked, 2:] <= 0).any() or (s[pre, :2] < 0).any():
        raise ValueError("Missing hard specifications")
    # Require masking, so accidental free reference geometry cannot leak in.
    if (s[~locked] != -1).any() or (s[fixed & ~pre, :2] != -1).any():
        raise ValueError("Unmasked ordinary block geometry is forbidden")
    effective = torch.where(locked, s[:, 2:].prod(1), areas)
    scale = effective.sum().sqrt().clamp_min(1e-6)
    known_wh = torch.where(locked[:, None], s[:, 2:] / scale, torch.zeros_like(s[:, 2:]))
    known_center = torch.where(pre[:, None], (s[:, :2] + s[:, 2:] / 2) / scale, torch.zeros_like(s[:, :2]))
    adjacency = torch.zeros((n, n), device=device)
    if b2b.numel():
        i, j = b2b[:, 0].long(), b2b[:, 1].long()
        adjacency.index_put_((i, j), b2b[:, 2].clamp_min(0), accumulate=True)
        adjacency.index_put_((j, i), b2b[:, 2].clamp_min(0), accumulate=True)
    degree = adjacency.sum(1)
    relation = []
    counts = []
    for col in (2, 3):
        same = (c[:, col, None] == c[None, :, col]) & (c[:, col, None] > 0)
        same.fill_diagonal_(False)
        relation.append(same.float() / same.sum(1, keepdim=True).clamp_min(1))
        counts.append(same.sum(1, keepdim=True).float() / max(n, 1))
    pin_weight = torch.zeros(n, device=device)
    pin_center = torch.zeros((n, 2), device=device)
    if p2b.numel():
        pi, bi = p2b[:, 0].long(), p2b[:, 1].long()
        weight = p2b[:, 2].clamp_min(0)
        pin_weight.index_add_(0, bi, weight)
        pin_center.index_add_(0, bi, pins[pi, :2] / scale * weight[:, None])
        pin_center /= pin_weight[:, None].clamp_min(1e-6)
    boundary = ((c[:, 4].long()[:, None] & torch.tensor([1, 2, 4, 8], device=device)) != 0).float()
    features = torch.cat((torch.log(effective / effective.mean()).unsqueeze(1), known_wh, known_center,
                          fixed[:, None].float(), pre[:, None].float(), boundary, *counts,
                          torch.log1p(degree)[:, None], torch.log1p(pin_weight)[:, None], pin_center), dim=1)
    adj = adjacency / degree[:, None].clamp_min(1e-6)
    mask = torch.stack((~pre, ~pre, ~locked), dim=1).float()
    return dict(features=features, adjacency=adj, relations=relation, areas=areas, specs=s,
                scale=scale, fixed=fixed, preplaced=pre, locked=locked, mask=mask)


def encode_target(reference, graph):
    r = torch.as_tensor(reference, device=graph["features"].device)
    return torch.cat(((r[:, :2] + r[:, 2:] / 2) / graph["scale"],
                      torch.log(r[:, 2] / r[:, 3])[:, None]), dim=1)


def decode_layout(state, graph):
    ratio = state[:, 2].clamp(-5, 5)
    wh = torch.stack(((graph["areas"] * ratio.exp()).sqrt(),
                      (graph["areas"] / ratio.exp()).sqrt()), dim=1)
    wh = torch.where(graph["locked"][:, None], graph["specs"][:, 2:], wh)
    xy = (state[:, :2] * graph["scale"] - wh / 2).clamp_min(0)
    out = torch.cat((xy, wh), dim=1)
    return torch.where(graph["preplaced"][:, None], graph["specs"], out)
