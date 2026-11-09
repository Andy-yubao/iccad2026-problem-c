"""Analytic multiscale rectangle-density moments; no reference or raster sampling."""
import math
import torch

BANDS = (4, 8)
_MODES = {}


def modes(device, dtype):
    key = (str(device), dtype)
    if key not in _MODES:
        pairs = [(i, j) for i in range(BANDS[-1]+1)
                 for j in range(BANDS[-1]+1) if i or j]
        k = torch.tensor(pairs, device=device, dtype=dtype)
        multiplicity = (1+(k[:, 0] > 0).to(dtype))*(1+(k[:, 1] > 0).to(dtype))
        inverse = multiplicity/k.square().sum(-1)
        masks = torch.stack([((k[:, 0] <= band) & (k[:, 1] <= band)).to(dtype)
                             for band in BANDS])
        weights = masks*inverse[None]
        weights = weights/weights.sum(-1, keepdim=True)
        _MODES[key] = k, weights
    return _MODES[key]


def density_energy(positions, areas, valid, *, bounds=None):
    """Mass-normalized cosine energy versus uniform occupation in a rectangle.

    Each rectangle contributes its exact integral, weighted by public a_i/A.
    No B-dependent density amplitude can dilute the objective as B grows.
    Bounds normally equal the actual predicted bbox, with gradients through it;
    explicit fixed bounds are useful for mechanism checks only. Truncated modes
    and uniformity are proxies, not a guarantee of packing or zero overlap.
    """
    lo = positions[..., :2]
    hi = lo+positions[..., 2:]
    if bounds is None:
        lower = lo.masked_fill(~valid[..., None], float('inf')).amin(1)
        upper = hi.masked_fill(~valid[..., None], -float('inf')).amax(1)
    else:
        lower, upper = bounds
    extent = (upper-lower).clamp_min(1e-6)
    center = (lo+positions[..., 2:]/2-lower[:, None])/extent[:, None]
    half_width = positions[..., 2:]/(2*extent[:, None])
    frequencies, weights = modes(positions.device, positions.dtype)
    phase = center[:, :, None, :]*frequencies[None, None]
    half_phase = half_width[:, :, None, :]*frequencies[None, None]
    factors = (phase*math.pi).cos()*torch.sinc(half_phase)
    basis = factors[..., 0]*factors[..., 1]
    mass = areas*valid
    mass = mass/mass.sum(-1, keepdim=True).clamp_min(1e-12)
    coefficients = (mass[..., None]*basis).sum(1)
    return (coefficients.square()[:, None, :]*weights[None]).sum(-1).mean(-1)
