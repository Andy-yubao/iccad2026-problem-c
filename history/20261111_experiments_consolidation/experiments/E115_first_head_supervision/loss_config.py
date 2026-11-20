"""Fixed objectives; no coefficient search or stage-dependent curriculum."""
A = .08975162861857286
B = .3336832108335229
COEFFICIENTS = {
    'position': 1., 'shape': .25,
    'penetration': A, 'module_load_mean': A, 'module_max_excess10': A,
    'wirelength': .5*B, 'compactness': .5*B,
    'boundary_contribution': 2*B, 'grouping_contribution': 2*B,
}
ARMS = ('first9',)

def layer_weights(arm):
    if arm != 'first9': raise ValueError(arm)
    return {1: .2, 9: .8}

def configuration(arm):
    return dict(coefficients=dict(COEFFICIENTS),layer_weights=layer_weights(arm),
        soft_normalization='boundary+group only; MIB excluded from numerator and denominator',
        initial_readout_supervised=False,all_modules_position_and_shape_supervised=True,
        same_objective_backward_monitor_best_lr_earlystop=True,
        selection='own fixed per-case objective arithmetic mean; never Q',
        common_final_total='diagnostic only; not an alternate selection metric')
