"""One initial layout with the original E149 terms and quarter overlap."""
from dataclasses import asdict

from .loss import A, B, DEFAULT

ARMS = ('e164_l0',)
OVERLAP_SCALE = 0.25


def configuration(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(
        version='E164.initial_quarter_overlap.v1', arm=arm, readout='L0 only',
        coefficients=asdict(DEFAULT), A=A, B=B, overlap_weight=OVERLAP_SCALE,
        original_mst_contact=True, new_mst_blockage=False, all_modules_predicted=True,
        specification_overwrite=False, MIB=False,
        selection='Same mean fixed dev500 objective for training, monitoring, best, LR, early stop',
        postprocessing=False, Q=False, test_read=False,
    )
