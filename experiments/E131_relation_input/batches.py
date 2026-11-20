"""Public input boundary and unchanged all-module area-preserving decode."""
import torch

PUBLIC_KEYS = ('features', 'adjacency', 'group', 'constraints', 'valid', 'address',
               'mask', 'area', 'known', 'specs', 'scale')


def inputs_only(batch):
    return {key: batch[key] for key in PUBLIC_KEYS if key in batch}


def decode(state, g):
    r = state[..., 2].clamp(-5, 5)
    wh = torch.stack(((g['area'] * r.exp()).clamp_min(1e-12).sqrt(),
                      (g['area'] / r.exp()).clamp_min(1e-12).sqrt()), -1)
    xy = (state[..., :2] - wh / 2).clamp_min(0)
    return torch.cat((xy, wh), -1)  # No F/P specification overwrite.
