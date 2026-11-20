"""Initial plus4/8/12: auxiliary.2 evenly split, final.8, no overlap at0/4."""
from dataclasses import asdict
from .base_loss import DEFAULT,A,B
ARMS=('e150_role_joint',)
WEIGHTS={0:1/15,4:1/15,8:1/15,12:.8}
NO_OVERLAP=(0,4)
def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E150.joint.initial_4_8_12.v1',arm=arm,layer_weights={str(k):v for k,v in WEIGHTS.items()},
        no_overlap_layers=list(NO_OVERLAP),full_loss_layers=[8,12],coefficients=asdict(DEFAULT),A=A,B=B,
        selection='Same weighted per-case aggregate for backward/monitor/best/LR/earlystop',
        all_modules_predicted=True,MIB=False,specification_overwrite=False,postprocessing=False,Q=False,test_read=False)
