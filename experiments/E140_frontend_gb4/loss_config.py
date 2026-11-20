"""One frozen E139-matched arm; grouping/boundary x4 at S2/S6 only."""
from .loss import DEFAULT,configuration as objective_configuration

ARMS=('e119_front_gb4',)
SUPERVISED_LAYERS=(2,6,8,10,12)
NO_OVERLAP_LAYERS=(2,6)


def layer_weights(arm):
    if arm not in ARMS:raise ValueError(arm)
    return {2:.05,6:.05,8:.05,10:.05,12:.8}


def configuration(arm):
    value=objective_configuration(DEFAULT)
    value.update(version='E140.frontend_gb4.v1',arm=arm,
        layer_weights={str(k):v for k,v in layer_weights(arm).items()},
        no_overlap_layers=[2,6],full_loss_layers=[8,10,12],
        grouping_multipliers={'2':4,'6':4,'8':1,'10':1,'12':1},
        boundary_multipliers={'2':4,'6':4,'8':1,'10':1,'12':1},
        removed_group='All direct collision penalties at S2/S6, exactly as E139',
        change='Multiply original weighted G/B contributions by4 only at S2/S6; no renormalization or outer-weight change',
        selection='Unique own fixed aggregate Loss best; never Q',
        common_final_new='final_total diagnostic only, unchanged E139 full Loss',
        common_final_original='original_final_total diagnostic only',topology_similarity_in_training=False)
    return value
