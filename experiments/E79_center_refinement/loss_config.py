"""One fixed center-refinement objective; source-regression guard only."""
from .losses import WEIGHTS
def configuration(arm):
    if arm!='topology':raise ValueError(arm)
    return {'revision':'E79.center_source_guard.v1','weights':{**WEIGHTS,'true_intersection':4.,'box_expansion':5.,'topology_regression':.1},
        'count_factor':False,'schedule':'Fixed from epoch0; no ramp','selection':'same dev_monitor500 arithmetic total',
        'aggregation':'G+4I+5E+.1R, R=mean+.25max one-sided squared source native order/axis shortfall; required source advantage capped.1',
        'output_contract':'prediction3 plus detached public source3; only first3 decoded/scored',
        'reference_role':'Supervision only in G/E; R uses only frozen public source prediction',
        'A_role':'E78 anchored-path term excluded: with fixed dimensions it has no ordinary center gradient except discrete order jumps',
        'source_shapes':'Entire E78 frozen/eval, source ratios and widths/heights unchanged'}
