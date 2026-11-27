ARMS = ('e152_s2_ref', 'e152_s2_no_ref')
REF_WEIGHT = dict(zip(ARMS, (1., 0.)))
DRIFT_WEIGHT = .2
CENTER_TOLERANCE = .05
RATIO_TOLERANCE = .10
RATIO_WEIGHT = .25
HUBER_BETA = .10
OVERLAP_THRESHOLD = .01
OVERLAP_DRIFT_WEIGHT = .25


def configuration(arm):
    return dict(reference_weight=REF_WEIGHT[arm], drift_weight=DRIFT_WEIGHT,
                center_tolerance=CENTER_TOLERANCE, ratio_tolerance=RATIO_TOLERANCE,
                ratio_weight=RATIO_WEIGHT, huber_beta=HUBER_BETA,
                overlap_threshold=OVERLAP_THRESHOLD, overlap_drift_weight=OVERLAP_DRIFT_WEIGHT,
                input_layer=8, supervised_layer=12, public_terms='E151 coefficients')
