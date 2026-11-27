"""Identical E81 per-source fixed objective in both single-E74 arms."""
from .losses import WEIGHTS
def configuration(arm):
    if arm not in ('control','cooperative'):raise ValueError(arm)
    return dict(revision='E85.singleP74.fixed_E81_objective.v1',weights={**WEIGHTS,'true_intersection':4.,'box_expansion':5.,'anchored_corridor':.25},count_factor=False,schedule='Fixed epoch0 onward, one E74 view/member',selection='Same L=G+4I+5E+.25A arithmetic dev_monitor500 mean',aggregation='Unchanged E81 per-source formula; single source',reference_role='Loss only, never cooperative axes/order or inputs')
