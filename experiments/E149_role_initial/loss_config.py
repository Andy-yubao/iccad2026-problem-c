"""One directly supervised initial readout, no direct collision objective."""
from dataclasses import asdict
from .loss import DEFAULT,A,B
ARMS=('e149_role_initial',)
def configuration(arm):
    if arm not in ARMS:raise ValueError(arm)
    return dict(version='E149.initial_no_overlap.v1',arm=arm,readout='initial only',weight=1.,
        coefficients=asdict(DEFAULT),A=A,B=B,overlap_weight=0.,
        retained='reference, wirelength, compactness, grouping, boundary, public F/P dimensions and P position, unchanged original coefficients',
        excluded='Entire direct overlap group: penetration, mean load, maximum excess',
        all_modules_predicted=True,specification_overwrite=False,MIB=False,
        selection='Same initial no-overlap per-case Loss arithmetic mean for backward/monitor/best/LR/earlystop',
        postprocessing=False,Q=False,test_read=False)
