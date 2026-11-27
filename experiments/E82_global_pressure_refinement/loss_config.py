"""One same fixed per-source and per-case objective, no source curriculum."""
from .losses import WEIGHTS
def configuration(arm):
    if arm!='topology':raise ValueError(arm)
    return {'revision':'E82.multisource_warm_wide.v1','weights':{**WEIGHTS,'true_intersection':4.,'box_expansion':5.,'anchored_corridor':.25},
      'count_factor':False,'schedule':'Fixed from epoch0; both sources on every train/dev example',
      'selection':'same dev_monitor500 arithmetic mean of per-case total',
      'aggregation':'source L=G+4I+5E+.25A; case total=mean_source(L)+.25*(max_source(L)-mean_source(L))',
      'output_contract':'P42 predicted3 then P74 predicted3 for fixed Loss; deployment emits only P42 predicted3',
      'A_role':'16-step selected mode0 fixed-shape anchor path pressure; ordinary gradients through selected dimensions, no axis/order STE; not a feasibility proof',
      'R_role':'E79 moving-source center-only guard removed; all F now learns shapes and positions; G retains all original eight terms',
      'reference_role':'Labels only in supervision G/E, never placer or correction inputs'}
