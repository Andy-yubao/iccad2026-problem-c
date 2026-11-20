"""User-specified A/B/C; auxiliary total .2 and terminal .8, fixed throughout."""
from .loss import DEFAULT, configuration as objective_configuration
ARMS = ('e119_A', 'e119_B', 'e119_C')

def layer_weights(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    if arm == 'e119_B':
        return {0: 1/30, 3: 2/45, 6: 1/18, 9: 1/15, 12: .8}
    return {0: .05, 3: .05, 6: .05, 9: .05, 12: .8}

def configuration(arm):
    value = objective_configuration(DEFAULT)
    value.update(version='E135.initial_supervision.v1', arm=arm,
        layer_weights={str(k): v for k, v in layer_weights(arm).items()},
        s0_objective='frontend_F' if arm == 'e119_C' else 'full public-priority-v1',
        frontend_F=dict(center_deadband_sqrt_area=.25, center_huber_beta=.1,
            center_subset='valid non-P', free_shape_subset='valid non-F/P',
            free_shape_coefficient=.05, public_dimension_coefficient=1., public_position_coefficient=1.),
        selection='Unique own fixed aggregate Loss best; never Q',
        common_final_new='final_total diagnostic only',
        common_final_original='original_final_total diagnostic only')
    return value
