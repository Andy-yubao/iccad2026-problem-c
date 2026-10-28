"""Immutable objective, identical across all optimization/monitoring paths."""
from .losses import WEIGHTS
def configuration(arm):
    if arm != 'topology': raise ValueError(arm)
    return {'revision':'E72.fixed_near_separation.v1','weights':{**WEIGHTS,'topology_separation':.5},
        'count_factor':False,'schedule':'Fixed from epoch0; no ramp',
        'selection':'dev_monitor500 arithmetic mean of this same total',
        'aggregation':'Original G plus .5 times ordered near-pair oriented penetration squared / valid block count; equal layout mean',
        'reference_role':'Supervision only; public-only model inputs unchanged',
        'control':'Reuse completed E63 original-G control, same E42 weights, fresh Adam and scheduler'}
