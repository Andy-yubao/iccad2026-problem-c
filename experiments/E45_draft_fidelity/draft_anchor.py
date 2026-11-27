"""Immutable original-draft geometry with public-only local confidence/release."""
from dataclasses import dataclass
import hashlib
import numpy as np
from scipy.sparse import coo_matrix

CENTER_STRENGTH = .08
SHAPE_STRENGTH = .01


def conflict_load(pos, p):
    lower, upper = pos[:, :2], pos[:, :2] + pos[:, 2:]
    intersection = np.minimum(upper[:, None], upper[None, :]) - np.maximum(lower[:, None], lower[None, :])
    np.fill_diagonal(intersection[:, :, 0], 0.)
    hit = (intersection > 1e-6).all(2)
    unit = np.sqrt(p.areas)
    penetration = np.maximum(0., intersection.min(2)) / np.maximum(np.minimum(unit[:, None], unit[None, :]), 1e-12)
    load = np.where(hit, penetration, 0.).sum(1)
    obstacle = (hit & p.preplaced[None, :]).any(1)
    return hit.any(1), load, obstacle


@dataclass
class DraftAnchor:
    original: np.ndarray
    target_center: np.ndarray
    target_wh: np.ndarray
    confidence: np.ndarray
    weights: np.ndarray
    streak: np.ndarray
    units: np.ndarray
    dynamic: bool
    original_sha256: str
    releases: int = 0
    rescue_scale: float = 1.

    @classmethod
    def build(cls, draft, initial_wh, p, dynamic):
        original = np.asarray(draft, dtype=float).copy()
        _, load, obstacle = conflict_load(original, p)
        confidence = np.clip(1. / (1. + 4. * load + 2. * obstacle), .1, 1.)
        confidence[p.preplaced] = 0.
        units = np.sqrt(p.areas.copy())
        units[p.fixed | p.preplaced] = np.sqrt(p.specs[p.fixed | p.preplaced, 2:].prod(1))
        center = original[:, :2] + original[:, 2:] / 2
        wh = np.asarray(initial_wh, float).copy()
        for value in (original, center, wh, confidence, units):
            value.flags.writeable = False
        return cls(original, center, wh, confidence, confidence.copy(), np.zeros(p.n, int),
                   units, dynamic, hashlib.sha256(original.tobytes()).hexdigest())

    def child(self):
        # Preserve original targets; weaken rather than replace them on repacking.
        return DraftAnchor(self.original, self.target_center, self.target_wh, self.confidence,
                           self.weights.copy(), np.zeros_like(self.streak), self.units,
                           self.dynamic, self.original_sha256, rescue_scale=self.rescue_scale * .1)

    def observe(self, proposal, p):
        if not self.dynamic or proposal is None:
            return 0
        hit, _, _ = conflict_load(proposal, p)
        self.streak = np.where(hit, self.streak + 1, 0)
        release = (self.streak >= 2) & ~p.preplaced
        previous = self.weights.copy()
        self.weights[release] = np.maximum(.05 * self.confidence[release], .25 * self.weights[release])
        changed = self.weights < previous
        self.streak[changed] = 0
        count = int(changed.sum()); self.releases += count
        return count

    def coefficients(self, p):
        q = self.rescue_scale * self.weights / (p.n * (self.units / p.scale) ** 2)
        center = CENTER_STRENGTH * q
        shape = SHAPE_STRENGTH * q
        shape[p.fixed | p.preplaced] = 0.
        return center, shape

    def diagnostics(self):
        if hashlib.sha256(self.original.tobytes()).hexdigest() != self.original_sha256:
            raise AssertionError('Original draft changed inside optimizer')
        active = self.confidence > 0
        return {'original_draft_sha256': self.original_sha256, 'local_release_events': self.releases,
                'center_strength': CENTER_STRENGTH, 'shape_strength': SHAPE_STRENGTH,
                'initial_confidence_mean': float(self.confidence[active].mean()) if active.any() else 0.,
                'final_weight_mean': float(self.weights[active].mean()) if active.any() else 0.,
                'rescue_scale': self.rescue_scale, 'target_is_original_model_draft': True}


def quadratic(p, context, xy, nv, dims=None, fixed_wh=None):
    """P,q for sum w||XY+WH/2-Cdraft||² + v||WH-WHcompatible||².

    Clarabel minimizes .5*x'P*x + q'x. All coordinates below are normalized.
    No new variables or inequalities are introduced.
    """
    center, shape = context.coefficients(p)
    target = context.target_center / p.scale
    linear = np.zeros(nv)
    ii, jj, vv = [], [], []
    for a in (0, 1):
        ids = xy[:, a]
        ii.extend(ids); jj.extend(ids); vv.extend(2 * center)
        if dims is None:
            shifted = target[:, a] - fixed_wh[:, a] / 2
            linear[ids] -= 2 * center * shifted
        else:
            di = dims[:, a]
            ii.extend(di); jj.extend(di); vv.extend(.5 * center + 2 * shape)
            ii.extend(ids); jj.extend(di); vv.extend(center)
            ii.extend(di); jj.extend(ids); vv.extend(center)
            linear[ids] -= 2 * center * target[:, a]
            linear[di] -= center * target[:, a] + 2 * shape * context.target_wh[:, a] / p.scale
    return coo_matrix((vv, (ii, jj)), shape=(nv, nv)).tocsc(), linear
