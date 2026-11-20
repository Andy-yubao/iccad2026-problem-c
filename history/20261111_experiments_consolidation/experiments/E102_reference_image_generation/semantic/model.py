"""Identity-aware U-Net with multiscale module attention and pixel mask logits."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..data import FEATURES, MAX_MODULES
from ..model import RelationAttention

CLASSES = MAX_MODULES + 1
MASK = CLASSES


class Residual(nn.Module):
    def __init__(self, incoming, outgoing, condition=256):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, incoming)
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, outgoing)
        self.affine = nn.Linear(condition, 2 * outgoing)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.skip = nn.Conv2d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x, condition):
        h = self.conv1(F.silu(self.norm1(x)))
        a, b = self.affine(F.silu(condition)).chunk(2, 1)
        h = self.norm2(h) * (1 + a[:, :, None, None]) + b[:, :, None, None]
        return self.skip(x) + self.conv2(F.silu(h))


class SpatialCondition(nn.Module):
    def __init__(self, channels, dim=192, inner=128, heads=4):
        super().__init__()
        self.heads, self.inner = heads, inner
        self.norm = nn.GroupNorm(8, channels)
        self.query = nn.Conv2d(channels, inner, 1)
        self.key_value = nn.Linear(dim, 2 * inner)
        self.output = nn.Conv2d(inner, channels, 1)

    def forward(self, x, modules, valid):
        b, _, h, w = x.shape
        q = self.query(self.norm(x)).flatten(2).transpose(1, 2)
        q = q.reshape(b, h * w, self.heads, self.inner // self.heads).transpose(1, 2)
        k, v = self.key_value(modules).reshape(b, -1, 2, self.heads, self.inner // self.heads).permute(2, 0, 3, 1, 4)
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=valid[:, None, None, :])
        context = context.transpose(1, 2).reshape(b, h * w, self.inner).transpose(1, 2).reshape(b, self.inner, h, w)
        return x + self.output(context)


class SemanticGenerator(nn.Module):
    def __init__(self, widths=(48, 64, 128, 192, 256), dim=192, module_layers=6, descriptor=32):
        super().__init__()
        self.config = dict(widths=list(widths), dim=dim, module_layers=module_layers, descriptor=descriptor)
        self.input_features = nn.Linear(FEATURES, dim)
        self.module_identity = nn.Embedding(MAX_MODULES, dim)
        self.module_encoder = nn.ModuleList([RelationAttention(dim, 4) for _ in range(module_layers)])
        self.token_embedding = nn.Embedding(CLASSES + 1, 16)
        self.input = nn.Conv2d(19, widths[0], 3, padding=1)
        self.time = nn.Sequential(nn.Linear(32, 256), nn.SiLU(), nn.Linear(256, 256))
        self.pooled_condition = nn.Linear(dim, 256)
        self.down, self.downsample, self.down_context = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for i, channels in enumerate(widths):
            self.down.append(nn.ModuleList([Residual(channels, channels), Residual(channels, channels)]))
            self.down_context.append(SpatialCondition(channels, dim) if i >= 2 else nn.Identity())
            if i < len(widths) - 1:
                self.downsample.append(nn.Conv2d(channels, widths[i + 1], 3, stride=2, padding=1))
        self.up, self.up_context = nn.ModuleList(), nn.ModuleList()
        current = widths[-1]
        for i in range(len(widths) - 2, -1, -1):
            self.up.append(nn.ModuleList([Residual(current + widths[i], widths[i]), Residual(widths[i], widths[i])]))
            self.up_context.append(SpatialCondition(widths[i], dim) if i >= 2 else nn.Identity())
            current = widths[i]
        self.pixel_descriptor = nn.Sequential(nn.GroupNorm(8, widths[0]), nn.SiLU(), nn.Conv2d(widths[0], descriptor, 1))
        self.mask_key = nn.Linear(dim, descriptor)
        self.mask_bias = nn.Linear(dim, 1)
        self.background = nn.Parameter(torch.randn(dim) * .02)

    def encode(self, public):
        x = self.input_features(public["features"]) + self.module_identity.weight[None]
        for layer in self.module_encoder:
            x = layer(x, public["relations"], public["valid"])
        return x

    def forward(self, noisy_ids, mask_fraction, public):
        modules = self.encode(public)
        valid = public["valid"]
        pooled = (modules * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        frequencies = torch.exp(torch.linspace(0, math.log(1000), 16, device=noisy_ids.device))
        phase = mask_fraction[:, None] * frequencies[None]
        condition = self.time(torch.cat((phase.sin(), phase.cos()), dim=1)) + self.pooled_condition(pooled)
        b, h, w = noisy_ids.shape
        yy, xx = torch.meshgrid(torch.linspace(1, -1, h, device=noisy_ids.device),
                                torch.linspace(-1, 1, w, device=noisy_ids.device), indexing="ij")
        coordinates = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        token_image = self.token_embedding(noisy_ids).permute(0, 3, 1, 2)
        image = self.input(torch.cat((token_image, coordinates, (noisy_ids == MASK)[:, None].float()), dim=1))
        skips = []
        for i, blocks in enumerate(self.down):
            for block in blocks:
                image = block(image, condition)
            if i >= 2:
                image = self.down_context[i](image, modules, valid)
            skips.append(image)
            if i < len(self.downsample):
                image = self.downsample[i](image)
        for i, blocks in enumerate(self.up):
            skip = skips[-i - 2]
            image = F.interpolate(image, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            image = torch.cat((image, skip), dim=1)
            for block in blocks:
                image = block(image, condition)
            if len(self.up) - i - 1 >= 2:
                image = self.up_context[i](image, modules, valid)
        tokens = torch.cat((self.background[None, None].expand(b, -1, -1), modules), dim=1)
        keys = self.mask_key(tokens)
        pixels = self.pixel_descriptor(image)
        logits = torch.einsum("bkd,bdhw->bkhw", keys, pixels) / math.sqrt(keys.shape[-1])
        logits = logits + self.mask_bias(tokens)[:, :, None]
        classes_valid = torch.cat((torch.ones(b, 1, device=valid.device, dtype=torch.bool), valid), dim=1)
        logits = logits.masked_fill(~classes_valid[:, :, None, None], -1e4)
        return {"logits": logits}
