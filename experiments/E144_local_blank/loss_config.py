"""Frozen E139 objective, shared by backward/monitor/best/LR/early stop."""
from .loss import DEFAULT, configuration as objective_configuration

ARMS = ('e144_local_blank',)
SUPERVISED_LAYERS = (2, 6, 8, 10, 12)
NO_OVERLAP_LAYERS = (2, 6)


def layer_weights(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return {2: .05, 6: .05, 8: .05, 10: .05, 12: .8}


def configuration(arm):
    weights = layer_weights(arm)
    value = objective_configuration(DEFAULT)
    value.update(version='E139.delayed_overlap.v1', arm=arm,
        layer_weights={str(k): v for k, v in weights.items()},
        no_overlap_layers=list(NO_OVERLAP_LAYERS), full_loss_layers=[8, 10, 12],
        removed_group='A*(penetration + module_load_mean + module_max_excess10) at S2/S6 only',
        retained_terms='reference, wirelength, compactness, grouping, boundary, public F/P shape and P position; exact parent coefficients and normalization',
        selection='Unique own fixed aggregate Loss best; never Q',
        common_final_new='final_total diagnostic only',
        common_final_original='original_final_total diagnostic only')
    return value
