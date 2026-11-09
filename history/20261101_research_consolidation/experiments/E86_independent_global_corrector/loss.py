"""Legacy E85 objective for neural-contract checks and common diagnostics ONLY.

This is not a registered E86 formal training objective. The train entry point
requires an explicit fixed task-objective registration before formal training.
"""
from .losses import components,WEIGHTS
from .intersection import fraction
from .box import expansion
from .anchored_paths import corridor

def per_case(state,g):
    v=components(state,g)
    v['true_intersection']=fraction(state,g);v['box_expansion']=expansion(state,g)
    v['anchored_corridor']=corridor(state,g)[0]
    v['total']=v['geometry']+4*v['true_intersection']+5*v['box_expansion']+.25*v['anchored_corridor']
    return v

def losses(state,g):return {k:v.mean() for k,v in per_case(state,g).items()}

def configuration():return dict(formula='G+4I+5E+.25A',G_weights=WEIGHTS,
    selection='same fixed total dev_monitor500 arithmetic mean',
    N5_role='Primary draft acceptance and monitoring metric; no unregistered tail coefficient in first architecture study')
