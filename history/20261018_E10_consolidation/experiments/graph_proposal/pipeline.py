"""E1 three-stage input-only pipeline. Geometry copied from baseline on 2026-10-16."""
import os
import time
import numpy as np
from experiments.graph_proposal.packing import prepare_problem, pack, default_order
from experiments.graph_proposal.feasible import Objective, is_legal, row_control


def correct(raw, p, steps=12):
    """Bounded continuous overlap relaxation; immutable dimensions/anchors stay locked."""
    pos = raw.copy()
    for _ in range(steps):
        delta = np.zeros((p.n, 2))
        for i in range(p.n):
            for j in range(i):
                ov = np.minimum(pos[i, :2] + pos[i, 2:], pos[j, :2] + pos[j, 2:]) - np.maximum(pos[i, :2], pos[j, :2])
                if np.all(ov > 0):
                    axis = int(np.argmin(ov))
                    sign = 1 if pos[i, axis] + pos[i, axis+2]/2 >= pos[j, axis] + pos[j, axis+2]/2 else -1
                    mobile = int(not p.preplaced[i]) + int(not p.preplaced[j])
                    if mobile:
                        shift = sign * (ov[axis] + p.scale * 1e-9) / mobile
                        if not p.preplaced[i]: delta[i, axis] += shift
                        if not p.preplaced[j]: delta[j, axis] -= shift
        pos[:, :2] = np.maximum(0, pos[:, :2] + delta)
        pos[p.preplaced] = p.specs[p.preplaced]
    return pos


def heuristic(p):
    wh = np.sqrt(p.areas)[:, None] * np.ones((1, 2))
    locked = p.fixed | p.preplaced
    wh[locked] = p.specs[locked, 2:]
    pos = np.c_[np.maximum(0, p.anchors - wh / 2), wh]
    pos[p.preplaced] = p.specs[p.preplaced]
    return pos


class Optimizer:
    def __init__(self, strategy='heuristic', seconds=15, seed=0):
        if strategy not in ('heuristic', 'deterministic', 'flow1', 'flow4'):
            raise ValueError(strategy)
        self.strategy, self.seconds, self.seed = strategy, seconds, seed
        self.last_stats = {}

    def solve(self, *inputs):
        start = time.monotonic()
        deadline = start + self.seconds
        p = prepare_problem(*inputs)
        model = None
        if self.strategy != 'heuristic':
            from experiments.graph_proposal.model import GraphProposal
            checkpoint = os.environ.get('GRAPH_PROPOSAL_CHECKPOINT')
            if not checkpoint:
                raise ValueError('A trained --checkpoint is required; random weights are not a learned control')
            model = GraphProposal.from_checkpoint(checkpoint)
            expected = 'deterministic' if self.strategy == 'deterministic' else 'flow'
            if model.kind != expected: raise ValueError('Checkpoint kind does not match strategy')
        count = 4 if self.strategy == 'flow4' else 1
        raws = [heuristic(p)] if model is None else [x.detach().cpu().numpy().astype(float) for x in model.propose(inputs, seed=self.seed, candidates=count, steps=8)]
        objective = Objective(p)
        selected = None
        reports = []
        for raw in raws:
            if time.monotonic() >= deadline - .25: break
            t = time.monotonic()
            revised = correct(raw, p)
            correction_seconds = time.monotonic() - t
            centers = revised[:, :2] + revised[:, 2:]/2
            ratios = revised[:, 2] / revised[:, 3]
            # Common fixed algorithm; proposal changes ordering, shape and attraction.
            order = sorted(default_order(p).tolist(), key=lambda i: (round(centers[i, 1]/max(p.scale*.15, 1e-9)), centers[i, 0], i))
            t = time.monotonic()
            final = pack(p, order=order, ratios=ratios, target_width=p.scale,
                         shape_trials=1, proposal_centers=centers, deadline=deadline-.1)
            fallback = not is_legal(final, p)
            if fallback: final = row_control(p)
            score = objective(final)
            reports.append({'raw':raw.tolist(), 'corrected':revised.tolist(),
                            'continuous_seconds':correction_seconds, 'geometry_seconds':time.monotonic()-t,
                            'continuous_displacement_mean':float(np.linalg.norm(revised[:,:2]-raw[:,:2],axis=1).mean()),
                            'final_displacement_mean':float(np.linalg.norm(final[:,:2]-revised[:,:2],axis=1).mean()),
                            'fallback':fallback, 'selection_surrogate':score})
            if selected is None or score < selected[0]: selected = (score, final.copy(), len(reports)-1)
        if selected is None:
            final = row_control(p)
            self.last_stats = {'fallback':True,'reason':'budget_before_candidate','candidate_count':0}
        else:
            _, final, chosen = selected
            self.last_stats = {'candidate_count':len(reports), 'chosen_candidate':chosen,
                               'candidate_diagnostics':reports,'fallback':reports[chosen]['fallback'],
                               'geometry_solve_count':len(reports)}
        self.last_stats['solve_seconds'] = time.monotonic()-start
        self.last_stats['budget_overrun'] = time.monotonic() > deadline
        return final.tolist()
