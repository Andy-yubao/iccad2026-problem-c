"""Single two-stage candidate; S0 reference coefficient is user-selected .75."""
ARMS=('e153_initial','e153_backend')
WEIGHTS={4:.1,8:.1,12:.8}
REFERENCE={4:.75,8:0.,12:0.}
def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E153.mib_grouping.v1',arm=arm,layer_weights=WEIGHTS if arm.endswith('backend') else {0:1.},
        reference_coefficients=REFERENCE if arm.endswith('backend') else {0:1.},
        no_overlap_layers=[4] if arm.endswith('backend') else [0],mib=True,predict_all=True,
        grouping_bottleneck_coefficient=1.,mib_coefficient=1.,S2_drift_coefficient=.2)
