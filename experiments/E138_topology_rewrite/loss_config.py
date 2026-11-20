"""Exactly three fixed objectives, own-Loss selection and common final diagnostics."""
from .loss import DEFAULT,configuration as original_configuration
ARMS=('e119_A','e119_B','e119_C')

def layer_weights(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {3:.2/3,6:.2/3,9:.2/3,12:.8}

def configuration(arm):
    value=original_configuration(DEFAULT)
    value.update(version='E138.topology_rewrite.v1',arm=arm,
        layer_weights={str(k):v for k,v in layer_weights(arm).items()},
        intermediate='coarse_F' if arm=='e119_C' else 'complete public-priority-v1',
        clean_weight=1. if arm=='e119_A' else .8,
        corrupted_terminal_weight=0. if arm=='e119_A' else .2,
        coarse_F=dict(center_tolerance=.25,free_shape_coefficient=.05,strong_order_coefficient=.25,
            reference_separation_margin=.25,desired_center_order_margin=.25,
            aggregation='hardest reference-separated relation per module-axis, then eligible module-axis mean; exclude P-P'),
        corruption='swap25pct non-P center slots at one of S3/S6/S9; dimensions unchanged, corrupted state detached, planner public-context reset',
        selection='Own fixed dev objective for backward/monitor/best/LR/earlystop; Q never selects',
        common_final_new='final_total = CLEAN full Loss, irrespective of own mixed objective')
    return value
