"""Exact E90 fixed baseline; no calibration at random initialization."""
import json
from pathlib import Path
FIXED={'overlap':0.08975162861857286,'task':0.3336832108335229}
def set_coefficients(values):
    if values!=FIXED:raise ValueError('Only exact fixed E90 coefficients supported')
def load_coefficients(path):
    from .io import ROOT,sources,sha
    p=Path(path);value=json.loads(p.read_text(encoding='utf-8'))
    if value['status']!='registered' or value['sources']!=sources():raise ValueError('Loss/source registration mismatch')
    for name,digest in value['baseline_artifacts'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Frozen E90 baseline changed')
    set_coefficients(value['coefficients'])
    return dict(path=str(p.resolve()),sha256=sha(p),coefficients=dict(FIXED))
def configuration(arm):
    if arm!='simple':raise ValueError('Only E74-architecture candidate supported')
    return dict(revision='E90.fixed_baseline',weights=dict(reference=1.,**FIXED),count_factor=False,threshold=.1,position_shape=[1.,.25],overlap='penetration + mean block load + max excess10',task='.5*(wire+compactness)+2*public soft cost',selection='same total for backward, monitors, best, LR, early stop, evaluation',calibration='exact adopted E90 constants; no recalibration, no ramp')
