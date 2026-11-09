"""Pixel-space masking and MaskGIT-style generation; no reference at sampling."""
import math

import torch

from .model import MASK


def corrupt(target, generator):
    device = target.device
    phase = torch.rand(len(target), device=device, generator=generator)
    ratio = torch.cos(phase * math.pi / 2).clamp_min(.05)
    ratio = torch.where(torch.rand(len(target), device=device, generator=generator) < .25, 1., ratio)
    hidden = torch.rand(target.shape, device=device, generator=generator) < ratio[:, None, None]
    return torch.where(hidden, MASK, target), ratio, hidden


@torch.inference_mode()
def sample(model, public, resolution=512, steps=16, seed=0):
    """Public conditioning only. No reference, area projection, or rectangle fit."""
    device = public["features"].device
    b = len(public["features"])
    tokens = torch.full((b, resolution, resolution), MASK, device=device, dtype=torch.long)
    generator = torch.Generator(device=device).manual_seed(seed)
    total = resolution * resolution
    for step in range(steps):
        unknown = tokens == MASK
        ratio = unknown.float().flatten(1).mean(1)
        output = model(tokens, ratio, public)
        probability = output["logits"].softmax(1)
        # Temperature falls to zero: early stochastic proposals, final argmax.
        temperature = .6 * (1 - (step + 1) / steps)
        if temperature > 0:
            uniform = torch.rand(output["logits"].shape, device=device, generator=generator).clamp_(1e-6, 1 - 1e-6)
            proposal = (output["logits"] - temperature * (-uniform.log()).log()).argmax(1)
        else:
            proposal = output["logits"].argmax(1)
        confidence = probability.gather(1, proposal[:, None])[:, 0]
        keep_masked = int(total * math.cos((step + 1) / steps * math.pi / 2))
        if step == steps - 1:
            tokens = proposal
            break
        rank = confidence.flatten(1)
        next_mask = torch.zeros_like(rank, dtype=torch.bool)
        for i in range(b):
            number = min(keep_masked, max(0, int(unknown[i].sum()) - 1))
            if number:
                next_mask[i, rank[i].topk(number, largest=False).indices] = True
        tokens = torch.where(next_mask.reshape_as(tokens), MASK, proposal)
    return tokens
