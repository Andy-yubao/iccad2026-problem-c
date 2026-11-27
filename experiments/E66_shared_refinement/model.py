"""Final-only fixed G, bounded additive state updates, with optional shared blocks."""
import torch
from torch import nn
from .base_model import GeometryModel as BaseModel, decode


class GeometryModel(BaseModel):
    def __init__(self, width=96, layers=12, rectangle=True, ffn_dropout=.05,
                 shared_blocks=12, correction_scale_xy=.5, correction_scale_shape=.5,
                 minimum_xy_scale=.1):
        # Construct the full old trunk first in both arms. Kept block initializations,
        # initial head and post-init RNG are thus paired despite different unique counts.
        super().__init__(width=width, layers=layers, rectangle=rectangle, ffn_dropout=ffn_dropout)
        if layers != 12 or shared_blocks not in (3, 12):
            raise ValueError('E66 fixes twelve effective applications and3 or12 unique blocks')
        self.effective_steps = layers
        self.shared_blocks = shared_blocks
        self.layers = nn.ModuleList(list(self.layers[:shared_blocks]))
        self.correction_scale_xy = correction_scale_xy
        self.correction_scale_shape = correction_scale_shape
        self.minimum_xy_scale = minimum_xy_scale
        with torch.random.fork_rng(devices=[]):
            self.correction_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                                 nn.SiLU(), nn.Linear(width, 3))
            nn.init.zeros_(self.correction_head[-1].weight)
            nn.init.zeros_(self.correction_head[-1].bias)

    def apply_correction(self, state, delta, g):
        # The public global-scale floor permits small blocks to move between regions;
        # pure sqrt(area) bounds would restrict their total displacement too severely.
        local = g['area'].clamp_min(1e-12).sqrt().clamp_min(self.minimum_xy_scale)
        xy = self.correction_scale_xy*local[..., None]*delta[..., :2].tanh()
        shape = self.correction_scale_shape*delta[..., 2:].tanh()
        return state+torch.cat((xy, shape), -1)*g['mask']*g['valid'][..., None]

    def forward(self, g):
        allowed = {'features', 'adjacency', 'mib', 'group', 'mask', 'known', 'area',
                   'specs', 'constraints', 'valid', 'scale'}
        if set(g)-allowed:
            raise ValueError('Only public graph inputs may enter the network')
        h = self.input(g['features'])*g['valid'][..., None]
        state = self.head(h)
        for step in range(self.effective_steps):
            h = self.layers[step % self.shared_blocks](h+self.state_embed(state*g['mask']), g, state)
            state = self.apply_correction(state, self.correction_head(h), g)
        return state
