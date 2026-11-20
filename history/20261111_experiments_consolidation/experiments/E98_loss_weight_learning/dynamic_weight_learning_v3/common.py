"""Shared identity, archived-cache migration, scoring, and phase scheduling."""
import json
import math
from pathlib import Path
import numpy as np
from ..io import ROOT, HERE, sha, save
from .. import cache

OWNER = Path(__file__).resolve().parent
RUN = HERE / 'runs/20261104_dynamic_train500_v3_seed0'
PLAN_PATH = OWNER / 'plan.json'
MODEL = dict(width=96, layers=12, rectangle=True, ffn_dropout=.10, residual_dropout=.05)
KEYS = ('position', 'shape', 'penetration', 'module_load_mean',
        'module_max_excess10', 'wirelength', 'compactness',
        'boundary_contribution', 'grouping_contribution', 'mib_contribution')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def resolve_part(name):
    name = name.replace('\\', '/')
    if name.startswith(cache.OLD):
        path = cache.OWNER / name[len(cache.OLD):]
    elif name.startswith('experiments/E42_joint_regularization/runs/prepared_v5_e10_cache_v1/'):
        path = ROOT / 'history/20261103_final_delivery_consolidation' / name
    else:
        raise ValueError('Unregistered cache path: ' + name)
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Cache path escapes workspace')
    return path

# This is an explicit in-process path migration for this new run only.
# The historical cache implementation, manifests, and payloads stay unchanged.
cache.resolve_part = resolve_part

def sources():
    files=list(OWNER.glob('*.py'))+[OWNER/'README.md',OWNER/'launch.ps1',PLAN_PATH]
    for directory in ('depth_evaluation',):
        files+=list((HERE/directory).rglob('*.py'))+[HERE/directory/'budget.json']
    files += [HERE/p for p in ('__init__.py','model.py','features.py','batches.py','ports.py',
        'loss.py','simple_loss.py','loss_config.py','group_tree.py','captured_projection.py',
        'step_control.py','resident.py','data.py','dataset_view.py','cache.py','io.py','train.py',
        'qualification/evaluate.py','qualification/plan.json','qualification/metrics.py','generalization_baseline/model.py',
        'generalization_baseline/augmentation.py')]
    files += list((ROOT/'datasets').glob('*.py'))+[ROOT/'datasets/expanded_reference/__init__.py']
    files += list((ROOT/'scoring').glob('*.py'))
    files += [ROOT/p for p in ('protocol/paths.json','protocol/reference_workset_v5.json',
        'protocol/reference_workset_v6.json','protocol/split_v1.json',
        'third_party/FloorSet/iccad2026contest/iccad2026_evaluate.py',
        'third_party/FloorSet/cost.py','third_party/FloorSet/utils.py')]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))}

def coefficients(center, direction, sign, epsilon):
    values = np.array([center[k] for k in KEYS], dtype=np.float64)
    moved = values * np.exp(sign * epsilon * np.array(direction, dtype=np.float64))
    moved *= values.sum() / moved.sum()
    assert np.isfinite(moved).all() and (moved > 0).all()
    return dict(zip(KEYS, moved.tolist()))

def q_summary(records):
    from scoring.official import official_evaluator
    costs = [r['metrics']['cost'] for r in records]
    weighted = official_evaluator().compute_total_score(costs, [r['n'] for r in records])
    mean = float(np.mean(costs))
    seconds = np.array([r['complete_seconds'] for r in records])
    return dict(count=len(records), weighted_Q=weighted, mean_Q=mean,
                mixed_Q=.5*(weighted+mean),
                hard_legal=sum(r['metrics']['is_feasible'] for r in records),
                mean_complete_seconds=float(seconds.mean()),
                p95_complete_seconds=float(np.percentile(seconds,95)),
                max_complete_seconds=float(seconds.max()),
                over250ms=int((seconds > .25).sum()),
                p95_Q=float(np.percentile(costs,95)), max_Q=float(max(costs)),
                fallbacks=sum(r['fallback'] for r in records))

class PhaseSchedule:
    """Epoch2 baseline; bad epochs3,4,5 can first reduce LR after epoch5."""
    def __init__(self):
        self.lr = 1e-4
        self.best = None
        self.bad = 0
        self.drop_epoch = None

    def step(self, value, epoch):
        if epoch < 2:
            return False
        if epoch == 2:
            self.best = value
            return False
        delta = max(1e-5, abs(self.best)*.001)
        if value < self.best-delta:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        if self.drop_epoch is None and self.bad >= 3:
            self.lr, self.drop_epoch, self.bad = 5e-5, epoch, 0
            self.best = value
            return False
        return self.drop_epoch is not None and self.bad >= 3 and epoch-self.drop_epoch >= 3

    def state_dict(self):
        return dict(self.__dict__)

    def load_state_dict(self, state):
        assert set(state) == set(self.__dict__)
        self.__dict__.update(state)
