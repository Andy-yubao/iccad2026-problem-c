"""Single two-stage candidate; S0 reference coefficient is user-selected .75."""
ARMS=('e156_s0_backend',)
WEIGHTS={2:.2,4:.8}
REFERENCE={2:.75,4:.75}
def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E156.two_shared_S0.v1',arm=arm,layer_weights=WEIGHTS,
        reference_coefficients=REFERENCE,no_absolute_overlap_layers=[2,4],
        grouping_bottleneck_weight_by_layer={2:.2,4:.2},mib_weight=2*.3336832108335229,
        relative_overlap_weight=.08975162861857286,relative_overlap_baseline='unperturbed frozen L0',
        total='.2*L2_base+.8*L4_base+A*relu(overlap4-stopgrad(overlap0))',
        predict_all=True,perturbation='train-only:75%clean/12.5%jitter(.1sqrt(area))/12.5%one nearest-center swap',
        validation_perturbation=False,S2_drift_coefficient=0.)
