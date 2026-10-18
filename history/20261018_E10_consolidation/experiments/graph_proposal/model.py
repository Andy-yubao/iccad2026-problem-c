"""A shared graph encoder with regression or conditional flow matching head."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .data import decode_layout, encode_target, graph_features


class GraphEncoder(nn.Module):
    def __init__(self, width=64, layers=3):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(17, width), nn.SiLU(), nn.LayerNorm(width))
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(width * 5, width), nn.SiLU(),
                                                  nn.Linear(width, width)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])

    def forward(self, graph):
        h = self.input(graph["features"])
        for layer, norm in zip(self.layers, self.norms):
            summary = h.mean(0, keepdim=True).expand_as(h)
            h = norm(h + layer(torch.cat((h, graph["adjacency"] @ h,
                                          graph["relations"][0] @ h, graph["relations"][1] @ h, summary), 1)))
        return h


class GraphProposal(nn.Module):
    def __init__(self, kind="deterministic", width=64, layers=3):
        super().__init__()
        if kind not in ("deterministic", "flow"):
            raise ValueError("kind must be deterministic or flow")
        self.kind, self.width, self.layers_count = kind, width, layers
        self.encoder = GraphEncoder(width, layers)
        extra = 4 if kind == "flow" else 0
        self.head = nn.Sequential(nn.Linear(width + extra, width), nn.SiLU(), nn.Linear(width, 3))

    @property
    def config(self):
        return dict(kind=self.kind, width=self.width, layers=self.layers_count)

    def forward(self, graph, state=None, time=None, encoded=None):
        h = self.encoder(graph) if encoded is None else encoded
        if self.kind == "flow":
            if state is None or time is None:
                raise ValueError("Flow requires state and time")
            t = torch.as_tensor(time, device=h.device).reshape(1, 1).expand(len(h), 1)
            h = torch.cat((h, state, t), dim=1)
        return self.head(h)

    def training_loss(self, inputs, reference):
        graph = graph_features(inputs, next(self.parameters()).device)
        target = encode_target(reference, graph)
        mask = graph["mask"]
        if self.kind == "flow":
            noise = torch.randn_like(target)
            # Locked channels always expose only public specifications.
            noise = noise * mask + target * (1 - mask)
            time = torch.rand((), device=target.device)
            state = (1 - time) * noise + time * target
            prediction = self(graph, state, time)
            truth = target - noise
        else:
            prediction, truth = self(graph), target
        return ((prediction - truth).square() * mask).sum() / mask.sum().clamp_min(1)

    @torch.no_grad()
    def propose(self, solver_inputs, seed=0, candidates=1, steps=8):
        if not 1 <= candidates <= 4 or not 1 <= steps <= 128:
            raise ValueError("candidates must be 1..4 and steps 1..128")
        self.eval()
        graph = graph_features(solver_inputs, next(self.parameters()).device)
        encoded = self.encoder(graph)
        if self.kind == "deterministic":
            output = decode_layout(self(graph, encoded=encoded), graph).cpu()
            return [output.clone() for _ in range(candidates)]
        generator = torch.Generator(device=encoded.device).manual_seed(seed)
        known = torch.cat(((graph["specs"][:, :2] + graph["specs"][:, 2:] / 2) / graph["scale"],
                           torch.log(graph["specs"][:, 2:].clamp_min(1e-8)[:, 0] /
                                     graph["specs"][:, 2:].clamp_min(1e-8)[:, 1])[:, None]), 1)
        mask = graph["mask"]
        outputs = []
        for _ in range(candidates):
            state = torch.randn((len(encoded), 3), generator=generator, device=encoded.device)
            state = state * mask + known * (1 - mask)
            for step in range(steps):
                state = state + self(graph, state, step / steps, encoded=encoded) * mask / steps
            outputs.append(decode_layout(state, graph).cpu())
        return outputs

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        model = cls(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model"])
        return model.eval()
