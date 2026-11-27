"""Fixed original G components, accepting the copied monitor's explicit identity."""
from .losses import components as original_components,WEIGHTS

def components(state,g,*,overlap_kind='original',overlap_kappa=1.):
    if overlap_kind!='original' or overlap_kappa!=1.:raise ValueError('E61 fixes original overlap and weights')
    return original_components(state,g)
