"""E87 coefficients multiplied by three; fixed throughout all uses."""
import json
from pathlib import Path
FIXED = {'overlap': 0.08975162861857286, 'task': 0.3336832108335229}

def set_coefficients(values):
    if values != FIXED:raise ValueError('Only registered auxiliary x3 weights supported')

def load_coefficients(path):
    from .io import sources,sha,ROOT
    p=Path(path);value=json.loads(p.read_text(encoding='utf-8'))
    if value['status']!='registered' or value['sources']!=sources():raise ValueError('Fixed weight/source mismatch')
    if sha(ROOT/value['base_artifact'])!=value['base_artifact_sha256']:raise ValueError('Original E87 calibration changed')
    set_coefficients(value['coefficients'])
    return dict(path=str(p.resolve()),sha256=sha(p),coefficients=dict(FIXED))

def configuration(arm):
    if arm not in ('spatial','pooled'):raise ValueError('Only auxiliary x3 candidate supported')
    return dict(revision='E90.E87_auxiliary_x3.v1',weights=dict(reference=1.,**FIXED),
        count_factor=False,threshold=.10,selection='same fixed total, dev500 arithmetic mean',
        position_shape=[1.,.25],overlap='penetration + mean block load + max excess10',
        task='.5*(wire+compactness)+2*public soft cost',
        calibration='original E87 train500 coefficients multiplied by3; no recalibration',
        original_gradient_targets=[.3,.2],new_gradient_targets=[.9,.6])
