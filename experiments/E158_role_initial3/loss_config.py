"""One L0 objective shared by backpropagation, monitoring, best and scheduler."""
ARMS=('e158_initial',)
OVERLAP_SCALE=.25

def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E158.L0.reference1_overlap025.v1',arm=arm,layer_weights={0:1.},
        reference_coefficient=1.,overlap_scale=OVERLAP_SCALE,
        original_overlap_weight=.08975162861857286,
        effective_overlap_weight=OVERLAP_SCALE*.08975162861857286,
        mib_weight=2*.3336832108335229,grouping_bottleneck_weight=0.,
        grouping_loss='unchanged E155/E157 original grouping; no proposed E158 redesign',
        predict_all=True,geometry_feedback=False,intermediate_layouts=False,
        selection='same total as training, including reference and quarter overlap')
