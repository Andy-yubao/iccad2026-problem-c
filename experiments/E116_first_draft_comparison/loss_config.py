"""One identical complete single-draft objective for both standalone arms."""
A = .08975162861857286
B = .3336832108335229
COEFFICIENTS = {
    'position': 1., 'shape': .25,
    'penetration': A, 'module_load_mean': A, 'module_max_excess10': A,
    'wirelength': .5*B, 'compactness': .5*B,
    'boundary_contribution': 2*B, 'grouping_contribution': 2*B,
}
ARMS = ('original4', 'new_first')


def configuration(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(coefficients=dict(COEFFICIENTS), final_output_weight=1.,
                intermediate_supervision=False,
                soft_normalization='boundary+group only; MIB excluded',
                all_modules_position_and_shape_supervised=True,
                same_objective_backward_monitor_best_lr_earlystop=True,
                selection='fixed dev500 final Loss arithmetic mean; never Q')
