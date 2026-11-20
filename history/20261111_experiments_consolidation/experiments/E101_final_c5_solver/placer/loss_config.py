"""Fixed G plus continuous collision-gated reference-direction guidance."""
from .losses import WEIGHTS
def configuration(arm):
    if arm!='topology':raise ValueError(arm)
    return {'revision':'E74.collision_gated_direction.v1','weights':{**WEIGHTS,'collision_topology':.5},'count_factor':False,
        'schedule':'Fixed from epoch0; no ramp','selection':'dev_monitor500 mean of this same total',
        'aggregation':'Original G plus .5 times ordered eligible-pair squared directed deficit multiplied by intersection/min public area, sum/valid count; equal layout mean',
        'reference_role':'Supervision only; model public inputs unchanged',
        'auxiliary_unconditional_topology':'diagnostic only, not part of total or selection',
        'control':'Frozen E42, completed E73 and saved E63. Data/Loss combination, not isolated causal estimate.'}
