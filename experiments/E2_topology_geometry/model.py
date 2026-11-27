"""Small graph encoder and pair-direction model; no reference input features."""
import numpy as np
import torch
from torch import nn


def features(p, dims, device="cpu"):
    scale = max(p.scale, 1e-8)
    flags = p.constraints.astype(float).copy()
    flags[:, 2:4] = flags[:, 2:4] > 0
    flags[:, 4] /= 15.
    node = np.column_stack((np.log(np.maximum(p.areas / scale**2, 1e-12)),
                            dims / scale, p.anchors / scale, flags))
    adjacency = p.adjacency.copy()
    adjacency += np.eye(p.n)
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1.)
    pairs = np.column_stack(np.triu_indices(p.n, 1))
    i, j = pairs.T
    extra = np.column_stack(((p.anchors[j] - p.anchors[i]) / scale,
                             (p.constraints[i, 2] == p.constraints[j, 2]) & (p.constraints[i, 2] > 0),
                             (p.constraints[i, 3] == p.constraints[j, 3]) & (p.constraints[i, 3] > 0)))
    return (torch.tensor(node, dtype=torch.float32, device=device),
            torch.tensor(adjacency, dtype=torch.float32, device=device),
            torch.tensor(pairs, dtype=torch.long, device=device),
            torch.tensor(extra, dtype=torch.float32, device=device))


class RelationModel(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.embed = nn.Linear(10, width)
        self.layers = nn.ModuleList([nn.Linear(width * 2, width) for _ in range(2)])
        self.pair = nn.Sequential(nn.Linear(width * 3 + 4, width), nn.SiLU(), nn.Linear(width, 4))

    def forward(self, node, adjacency, pairs, extra):
        h = torch.nn.functional.silu(self.embed(node))
        for layer in self.layers:
            h = h + torch.nn.functional.silu(layer(torch.cat((h, adjacency @ h), dim=-1)))
        i, j = pairs.T
        return self.pair(torch.cat((h[i], h[j], h[i] - h[j], extra), dim=-1))


def ambiguous_targets(reference):
    """All separations true in the training label are acceptable targets."""
    rect = np.asarray(reference)
    i, j = np.triu_indices(len(rect), 1)
    tol = 1e-5
    valid = np.column_stack((rect[i, 0] + rect[i, 2] <= rect[j, 0] + tol,
                             rect[j, 0] + rect[j, 2] <= rect[i, 0] + tol,
                             rect[i, 1] + rect[i, 3] <= rect[j, 1] + tol,
                             rect[j, 1] + rect[j, 3] <= rect[i, 1] + tol))
    return torch.tensor(valid, dtype=torch.bool)


def ambiguity_loss(logits, valid):
    # Marginal likelihood of the set of valid directions; never choose an
    # arbitrary single orientation for diagonally separated rectangles.
    keep = valid.any(dim=1)
    if not keep.any():
        raise ValueError("Training label has no valid separation pairs")
    logits, valid = logits[keep], valid[keep]
    return (torch.logsumexp(logits, dim=1) - torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1)).mean()


def predict(p, dims, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("kind") != "topology_geometry_relations_v1":
        raise ValueError("Not a trained E2 relation checkpoint")
    model = RelationModel()
    model.load_state_dict(state["model"])
    model.eval()
    data = features(p, dims)
    with torch.no_grad():
        logits = model(*data).numpy()
    return data[2].numpy(), logits
