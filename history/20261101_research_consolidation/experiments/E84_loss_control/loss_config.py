"""One same fixed per-source and per-case objective, no source curriculum."""
from .losses import WEIGHTS
def configuration(arm):
    if arm!='topology':raise ValueError(arm)
    return {'revision':'E84.same_initialization_G_only.v1','weights':{**WEIGHTS,'true_intersection':0.,'box_expansion':0.,'anchored_corridor':0.},
      'count_factor':False,'schedule':'Fixed from epoch0; both sources on every train/dev example',
      'selection':'same dev_monitor500 arithmetic mean of per-case total',
      'aggregation':'source L=G; case total=mean_source(G)+.25*(max_source(G)-mean_source(G))',
      'output_contract':'P42 predicted3 then P74 predicted3 for fixed Loss; deployment emits only P42 predicted3',
      'A_role':'Diagnostic only; no contribution to backward, selection or scheduling. I/E likewise diagnostic only.',
      'R_role':'E79 moving-source center-only guard removed; all F now learns shapes and positions; G retains all original eight terms',
      'reference_role':'Labels only in supervision G/E, never placer or correction inputs'}
