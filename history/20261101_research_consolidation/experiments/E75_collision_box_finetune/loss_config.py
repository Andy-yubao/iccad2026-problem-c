"""Fixed G plus continuous collision-gated reference-direction guidance."""
from .losses import WEIGHTS
def configuration(arm):
    if arm!='topology':raise ValueError(arm)
    return {'revision':'E75.true_intersection_quadratic_box.v2','weights':{**WEIGHTS,'true_intersection':2.,'box_expansion':5.},'count_factor':False,
        'schedule':'Fixed from epoch0; no ramp','selection':'dev_monitor500 mean of this same total',
        'aggregation':'Original G plus2*true pair intersection/public total area +5*relu(log(B/B_reference))^2; equal layout mean',
        'reference_role':'Supervision only; model public inputs unchanged',
        'auxiliary_unconditional_topology':'diagnostic only, not part of total or selection',
        'control':'Frozen E42, completed E73 and saved E63. Data/Loss combination, not isolated causal estimate.'}
