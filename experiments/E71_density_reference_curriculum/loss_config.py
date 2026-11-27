from .arms import ARMS
from .losses import WEIGHTS
from .curriculum import configuration as curriculum_configuration

def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {'revision':'E71.density_mild_position_curriculum.v1','weights':dict(WEIGHTS),
            'reference_allowlist':['position','shape'],'count_factor':False,
            'aggregation':'Per-layout terms then equal-case arithmetic mean',
            'schedule':curriculum_configuration(),
            'selection':'Same final total on dev_monitor500, eligible only from epoch16; never Q or old-G',
            'density':'Exact rectangle cosine moments, bands4/8; public area mass normalization, no reference; compactness .5',
            'grouping':'Deterministic GPU Prim forest; gradients through selected contact costs',
            'old_G':'Monitoring-only diagnostic, excluded from new total and selection'}
