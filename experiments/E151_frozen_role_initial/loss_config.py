"""E148 supervision at L4, L8 and L12 for the trainable backend."""
from dataclasses import asdict
from .base_loss import DEFAULT,A,B
ARMS=('e151_frozen_role_initial',)
WEIGHTS={4:.1,8:.1,12:.8}
NO_OVERLAP=(4,)
def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E151.backend.4_8_12.v1',arm=arm,layer_weights={str(k):v for k,v in WEIGHTS.items()},
        no_overlap_layers=list(NO_OVERLAP),full_loss_layers=[8,12],coefficients=asdict(DEFAULT),A=A,B=B,
        selection='Same weighted per-case aggregate for backward/monitor/best/LR/earlystop',
        all_modules_predicted=True,MIB=False,specification_overwrite=False,postprocessing=False,Q=False,test_read=False)
