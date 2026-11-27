from functools import lru_cache
import json
from .io import HERE
from .losses import WEIGHTS
@lru_cache(maxsize=None)
def configuration(arm):
    from .arms import ARMS
    if arm not in ARMS:raise ValueError(arm)
    centerline='centerline' in arm
    kappa=json.loads((HERE/'loss_scale.json').read_text(encoding='utf-8'))['kappa'] if centerline else 1.
    return {'count_factor':False,'revision':'E59.fixed_geometry.v1','weights':dict(WEIGHTS),'overlap_kind':'centerline_v1' if centerline else 'original','overlap_kappa':kappa,'selection':'same fixed total Loss, unchanged dev_monitor500 mean','schedule':'full fixed weights; no ramp or alternate proxy'}
