"""One train-only calibration; immutable coefficients throughout formal runs."""
import json,math
from pathlib import Path
_COEFFICIENTS={'overlap':1.,'task':1.}
def set_coefficients(values):
    global _COEFFICIENTS
    if set(values)!={'overlap','task'} or any(not math.isfinite(v) or v<=0 or v>100 for v in values.values()):raise ValueError('Invalid fixed coefficients')
    _COEFFICIENTS={k:float(v) for k,v in values.items()}
def load_coefficients(path):
    from .io import sources,sha
    p=Path(path);v=json.loads(p.read_text(encoding='utf-8'))
    if v['status']!='registered' or v['sources']!=sources():raise ValueError('Calibration source identity mismatch')
    set_coefficients(v['coefficients']);return dict(path=str(p.resolve()),sha256=sha(p),coefficients=_COEFFICIENTS)
def configuration(arm):
    if arm=='simple':return dict(revision='E87.three_groups.v1',weights=dict(reference=1.,**_COEFFICIENTS),count_factor=False,threshold=.10,
        selection='same fixed total, dev500 arithmetic mean',position_shape=[1.,.25],
        overlap='penetration + mean block load + max excess10',task='.5*(wire+compactness)+2*public soft cost',
        calibration='train500 parameter gradient fractions .3/.2, fixed once')
    if arm=='legacy':
        from .losses import WEIGHTS
        return dict(revision='E74.original_fixed_G_plus_collision.v1',weights={**WEIGHTS,'collision_topology':.5},count_factor=False)
    raise ValueError(arm)
