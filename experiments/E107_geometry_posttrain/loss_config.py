"""Fixed new objective, normalization frozen from train500 only."""
import json
from functools import lru_cache
from .io import HERE
A=.08975162861857286
B=.3336832108335229
COEFFICIENTS={'position':.1,'shape':.025,'area_excess_scaled':1.,'tail_scaled':1.,'whitespace_scaled':1.,
    'soft_original':2*B,'soft_barrier':2.,'wirelength':.5*B,'wire_barrier':1.,'trust':1.,'penetration':.05}
SOFT_TOLERANCES=(.01,.01,.0025)

@lru_cache(maxsize=1)
def normalization():
    data=json.loads((HERE/'normalization.json').read_text(encoding='utf-8'))
    if data['status']!='frozen_train500':raise ValueError('Normalization not frozen')
    return data

def configuration():
    return dict(revision='E107.union_geometry.ref10.v1',coefficients=dict(COEFFICIENTS),
        normalization=normalization(),soft_tolerances=list(SOFT_TOLERANCES),
        trust_center_radius=.1,trust_log_aspect_radius=.1,tail_threshold=.1,tail_smooth_width=.05,
        aggregation='layout terms then arithmetic mean',reference_position=.1,reference_shape=.025,
        monitoring='same fixed loss; eval/no augmentation; trust always deterministic student eval')
