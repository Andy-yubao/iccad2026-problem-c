from .arms import ARMS
from .losses import WEIGHTS

def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {'revision':'E70.reference_position_shape_public_task.v1','weights':dict(WEIGHTS),
            'reference_allowlist':['position','shape'],'count_factor':False,
            'aggregation':'Per-layout terms then equal-case arithmetic mean',
            'schedule':'Fixed from first batch; no ramps',
            'selection':'Same total on dev_monitor500; never Q or old-G',
            'grouping':'Deterministic GPU Prim forest; gradients through selected contact costs',
            'old_G':'Monitoring-only diagnostic, excluded from new total and selection'}
