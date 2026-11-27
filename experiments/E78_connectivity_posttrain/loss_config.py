"""Fixed G plus continuous collision-gated reference-direction guidance."""
from .losses import WEIGHTS
def configuration(arm):
    if arm!='topology':raise ValueError(arm)
    return {'revision':'E78.public_MIB_anchored_corridor.v1','weights':{**WEIGHTS,'true_intersection':2.,'box_expansion':5.,'anchored_corridor':.25},'count_factor':False,
        'schedule':'Fixed from epoch0; no ramp','selection':'dev_monitor500 mean of this same total',
        'aggregation':'Original G+2I+5E+.25*(anchored corridor mean-square+.25max-square) pressure after16 mode0 DAG relaxations; equal layout mean',
        'reference_role':'Supervision only; model public inputs unchanged',
        'auxiliary_unconditional_topology':'diagnostic only, not part of total or selection',
        'control':'Frozen E42, completed E73 and saved E63. Data/Loss combination, not isolated causal estimate.'}
