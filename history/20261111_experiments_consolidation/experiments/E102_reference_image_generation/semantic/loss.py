"""Image-label matching only: equal-module CE, Dice, auxiliary pixel edges."""
import torch
from torch.nn import functional as F

from .model import CLASSES


def boundaries(labels):
    edges = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    edges[:, :, 1:] |= horizontal
    edges[:, :, :-1] |= horizontal
    edges[:, 1:, :] |= vertical
    edges[:, :-1, :] |= vertical
    return edges


def semantic_loss(output, target, corrupted):
    b = len(target)
    flat = target.flatten(1)
    logprob = F.log_softmax(output["logits"], dim=1)
    matched = logprob.gather(1, target[:, None])[:, 0].flatten(1)
    selected = corrupted.flatten(1).float()
    zeros = torch.zeros(b, CLASSES, device=target.device)
    selected_counts = zeros.scatter_add(1, flat, selected)
    class_ce = zeros.scatter_add(1, flat, -matched * selected) / selected_counts.clamp_min(1)
    visible = selected_counts[:, 1:] > 0
    ce_fg = (class_ce[:, 1:] * visible).sum(1) / visible.sum(1).clamp_min(1)
    ce = ce_fg + .25 * class_ce[:, 0]
    counts = zeros.scatter_add(1, flat, torch.ones_like(matched))
    probabilities = logprob.exp()
    predicted_mass = probabilities.flatten(2).sum(2)
    intersection = zeros.scatter_add(1, flat, matched.exp())
    dice_values = 1 - (2 * intersection + 1e-5) / (predicted_mass + counts + 1e-5)
    present = counts[:, 1:] > 0
    dice = (dice_values[:, 1:] * present).sum(1) / present.sum(1).clamp_min(1)
    edge_terms = []
    for first, second, truth in (
        (probabilities[:, :, :, :-1], probabilities[:, :, :, 1:], target[:, :, :-1] != target[:, :, 1:]),
        (probabilities[:, :, :-1, :], probabilities[:, :, 1:, :], target[:, :-1, :] != target[:, 1:, :])):
        different = (1 - (first * second).sum(1)).clamp(1e-6, 1 - 1e-6)
        edge_loss = F.binary_cross_entropy(different, truth.float(), reduction="none").flatten(1)
        truth = truth.flatten(1)
        pos = (edge_loss * truth).sum(1) / truth.sum(1).clamp_min(1)
        neg = (edge_loss * ~truth).sum(1) / (~truth).sum(1).clamp_min(1)
        edge_terms.append(.5 * (pos + neg))
    boundary = torch.stack(edge_terms).mean(0)
    total = ce + dice + .2 * boundary
    return total, {"ce": ce, "dice": dice, "boundary": boundary}


@torch.no_grad()
def image_metrics(prediction, target):
    values = []
    accuracies = []
    for p, t in zip(prediction, target):
        ids = torch.unique(t)
        ids = ids[ids > 0]
        scores = []
        for module in ids:
            truth, guess = t == module, p == module
            scores.append((truth & guess).sum().float() / (truth | guess).sum().clamp_min(1))
        values.append(torch.stack(scores).mean())
        fg = t > 0
        accuracies.append(((p == t) & fg).sum().float() / fg.sum().clamp_min(1))
    return {"module_mean_IoU": torch.stack(values), "foreground_pixel_accuracy": torch.stack(accuracies)}
