"""Conditional RGB image generator. No coordinates or rectangles are output."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import FEATURES, MAX_MODULES


class RelationAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.relation_weight = nn.Parameter(torch.ones(heads, 3) * .1)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, relations, valid):
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        bias = torch.einsum("hr,brij->bhij", self.relation_weight, relations)
        bias = bias.masked_fill(~valid[:, None, None, :], -torch.inf)
        attention = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(attention.transpose(1, 2).reshape(b, n, d))
        return x + self.ff(x)


class ImageCrossAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, image_tokens, modules, valid):
        query = self.norm(image_tokens)
        image_tokens = image_tokens + self.attention(query, modules, modules,
            key_padding_mask=~valid, need_weights=False)[0]
        return image_tokens + self.ff(image_tokens)


class ConvStage(nn.Module):
    def __init__(self, incoming, outgoing, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(incoming, outgoing, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, outgoing)
        self.affine = nn.Linear(dim, outgoing * 2)
        self.conv2 = nn.Conv2d(outgoing, outgoing, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, outgoing)

    def forward(self, image, condition):
        image = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False)
        image = self.norm1(self.conv1(image))
        scale, shift = self.affine(condition).chunk(2, dim=1)
        image = F.silu(image * (1 + scale[:, :, None, None]) + shift[:, :, None, None])
        return F.silu(self.norm2(self.conv2(image)))


class ReferenceImageGenerator(nn.Module):
    def __init__(self, resolution=512, dim=128, layers=4, heads=4, grid=32):
        super().__init__()
        ratio = resolution // grid
        if resolution % grid or ratio < 2 or ratio & (ratio - 1):
            raise ValueError("Resolution must be grid times a positive power of two")
        self.config = dict(resolution=resolution, dim=dim, layers=layers, heads=heads, grid=grid)
        self.grid = grid
        self.features = nn.Linear(FEATURES, dim)
        self.identity = nn.Embedding(MAX_MODULES, dim)
        self.encoder = nn.ModuleList([RelationAttention(dim, heads) for _ in range(layers)])
        yy, xx = torch.meshgrid(torch.linspace(1, -1, grid), torch.linspace(-1, 1, grid), indexing="ij")
        coords = torch.stack((xx, yy), dim=-1).flatten(0, 1)
        fourier = torch.cat([coords] + [fn(coords * math.pi * 2**i)
                            for i in range(4) for fn in (torch.sin, torch.cos)], dim=1)
        self.register_buffer("coordinates", fourier)
        self.spatial = nn.Linear(fourier.shape[1], dim)
        self.cross = nn.ModuleList([ImageCrossAttention(dim, heads) for _ in range(2)])
        self.global_condition = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        channels = [max(16, (128 // 2 ** max(0, i - 1)) // 8 * 8)
                    for i in range(int(math.log2(ratio)))]
        self.stages = nn.ModuleList()
        incoming = dim
        for outgoing in channels:
            self.stages.append(ConvStage(incoming, outgoing, dim))
            incoming = outgoing
        self.output = nn.Conv2d(incoming, 3, 1)

    def forward(self, batch):
        valid = batch["valid"]
        modules = self.features(batch["features"]) + self.identity.weight[None]
        for layer in self.encoder:
            modules = layer(modules, batch["relations"], valid)
        pooled = (modules * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        condition = self.global_condition(pooled)
        spatial = self.spatial(self.coordinates)[None].expand(len(modules), -1, -1)
        for layer in self.cross:
            spatial = layer(spatial, modules, valid)
        image = spatial.transpose(1, 2).reshape(len(modules), -1, self.grid, self.grid)
        for stage in self.stages:
            image = stage(image, condition)
        return self.output(image).tanh()


def image_loss(prediction, target):
    """Same per-image RGB MSE for gradients, monitoring, best and scheduler."""
    return (prediction - target).square().flatten(1).mean(1)
