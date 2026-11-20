"""Learned public problem encoding and exactly one standard Transformer block."""
import math
import torch
from torch import nn

PUBLIC = ('area', 'specs', 'constraints', 'valid', 'scale', 'wire_b2b',
          'pin_coordinates', 'pin_weights', 'pin_valid', 'address')


def inputs_only(batch):
    return {key: batch[key] for key in PUBLIC}


def public_descriptor(g):
    """20 node, 640 addressed relation and 8 global public factual channels."""
    if set(g) != set(PUBLIC):
        raise ValueError('The model accepts only the explicit public input contract')
    valid = g['valid']; dtype = g['area'].dtype
    b, n = valid.shape
    if n > 128 or g['address'].shape != (b, n):
        raise ValueError('Unsupported module/address shape')
    v = valid.to(dtype); count = v.sum(1).clamp_min(1)
    active = valid[:, :, None] & valid[:, None, :]
    eye = torch.eye(n, dtype=torch.bool, device=v.device)[None]
    wire = g['wire_b2b'] * active
    linked = (wire > 0) & active & ~eye
    labels = g['constraints'][..., 3]
    same = ((labels[:, :, None] == labels[:, None, :]) &
            (labels[:, :, None] > 0) & active & ~eye)
    fixed = (g['constraints'][..., 0] != 0) & valid
    pre = (g['constraints'][..., 1] != 0) & valid
    locked = fixed | pre
    spec = g['specs']
    wh = torch.where(locked[..., None], spec[..., 2:], torch.zeros_like(spec[..., 2:]))
    center = torch.where(pre[..., None], spec[..., :2] + spec[..., 2:] / 2,
                         torch.zeros_like(spec[..., :2]))
    safe_wh = torch.where(locked[..., None], spec[..., 2:], torch.ones_like(wh)).clamp_min(1e-12)
    ratio = (safe_wh[..., 0] / safe_wh[..., 1]).log()
    # Device construction avoids a host-to-device constant copy inside CUDA capture.
    bits = torch.ones(4, dtype=g['constraints'].dtype, device=v.device) << torch.arange(4, device=v.device)
    boundary = ((g['constraints'][..., 4, None] & bits) != 0).to(dtype)
    pins = g['pin_valid'] & valid[..., None] & (g['pin_weights'] > 0)
    weights = g['pin_weights'] * pins
    pin_sum = weights.sum(-1)
    degree = wire.sum(-1)
    fraction = g['area'].clamp_min(1e-12)
    own = torch.cat((fraction[..., None], (fraction * count[:, None]).log()[..., None],
        fixed[..., None].to(dtype), pre[..., None].to(dtype), boundary, wh, center, ratio[..., None],
        degree.log1p()[..., None], (linked.sum(-1) / (count - 1).clamp_min(1)[:, None])[..., None],
        (same.sum(-1) / (count - 1).clamp_min(1)[:, None])[..., None], pin_sum.log1p()[..., None],
        pins.sum(-1).to(dtype).log1p()[..., None], linked.any(-1)[..., None].to(dtype),
        pins.any(-1)[..., None].to(dtype)), -1) * v[..., None]
    identity = (g['address'][..., None] == torch.arange(128, device=v.device)).to(dtype) * v[..., None]
    occupied = identity.sum(1)
    inverse = identity.argmax(1)[:, None, :].expand(-1, n, -1)
    def to_slots(matrix):
        return matrix.gather(2, inverse) * occupied[:, None, :]
    relation = torch.cat((identity, occupied[:, None, :].expand(-1, n, -1),
        to_slots(wire / degree.clamp_min(1e-12)[..., None]),
        to_slots(wire.log1p()), to_slots(same.to(dtype))), -1) * v[..., None]
    upper = torch.triu(torch.ones(n, n, device=v.device, dtype=torch.bool), diagonal=1)
    global_info = torch.stack((count / 128, count.log(), 2 * g['scale'].log(),
        wire.sum((1, 2)).log1p(), (linked & upper).sum((1, 2)) / (count * (count - 1) / 2).clamp_min(1),
        pin_sum.sum(1).log1p(), fixed.sum(1) / count, pre.sum(1) / count), -1)
    if own.shape[-1] != 20 or relation.shape[-1] != 640:
        raise ValueError('Registered public descriptor changed')
    return own, relation, global_info[:, None, :].expand(-1, n, -1), pins, weights


class StandardBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(256); self.norm2 = nn.LayerNorm(256)
        self.qkv = nn.Linear(256, 768); self.output = nn.Linear(256, 256)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Dropout(.1), nn.Linear(1024, 256))
        self.residual_dropout = nn.Dropout(.05)

    def forward(self, h, valid):
        b, n, _ = h.shape
        q, k, v = self.qkv(self.norm1(h)).view(b, n, 3, 4, 64).permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(64)
        scores = scores.masked_fill(~valid[:, None, None, :], -1e4)
        attended = (scores.softmax(-1) @ v).transpose(1, 2).reshape(b, n, 256)
        h = h + self.residual_dropout(self.output(attended))
        return (h + self.residual_dropout(self.ff(self.norm2(h)))) * valid[..., None]


class GeometryModel(nn.Module):
    def __init__(self, architecture='standard256'):
        super().__init__()
        if architecture != 'standard256':
            raise ValueError('Only the registered E137 architecture is supported')
        # Construction order is part of the frozen seed0 initialization.
        self.port_encoder = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 64))
        self.input_encoder = nn.Sequential(nn.Linear(796, 512), nn.GELU(), nn.Linear(512, 256), nn.LayerNorm(256))
        self.block = StandardBlock()
        self.final_norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, 3)

    def forward(self, g):
        own, relation, global_info, pins, weights = public_descriptor(g)
        port_input = torch.cat((g['pin_coordinates'], g['pin_weights'].clamp_min(0).log1p()[..., None]), -1)
        encoded = self.port_encoder(port_input)
        mean = (encoded * weights[..., None]).sum(-2) / weights.sum(-1).clamp_min(1e-12)[..., None]
        maximum = encoded.masked_fill(~pins[..., None], torch.finfo(encoded.dtype).min).amax(-2)
        maximum = torch.where(pins.any(-1)[..., None], maximum, torch.zeros_like(maximum))
        h = self.input_encoder(torch.cat((own, relation, mean, maximum, global_info), -1))
        h = h * g['valid'][..., None]
        h = self.block(h, g['valid'])
        return self.head(self.final_norm(h)) * g['valid'][..., None]
