"""Fixed phase-specific public objectives; no reference or bottleneck supervision."""
ARMS=('e157_s1','e157_s2')
WEIGHTS={'e157_s1':{6:.2,8:.8},'e157_s2':{10:.2,12:.8}}
def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E157.sequential_public.v1',arm=arm,layer_weights=WEIGHTS[arm],reference_weight=0.,grouping_bottleneck_weight=0.,absolute_overlap_weight=.08975162861857286,mib_weight=.6673664216670459,predict_all=True,layout_perturbation=False,S2_drift_weight=.2 if arm=='e157_s2' else 0.,drift_scope='L12 only, inside0.8 terminal layer weight; baseline frozen L8',drift_center_tolerance=.05,drift_log_ratio_tolerance=.10,drift_ratio_weight=.25,drift_huber_beta=.10,overlap_relief_threshold=.01,overlap_relief_weight=.25)
