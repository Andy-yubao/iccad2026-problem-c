"""Saved-layout geometric measures, preserved from the old display helper."""
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union

def measures(positions):
    p=np.asarray(positions,dtype=np.float64)
    total=p[:,2:].prod(1).sum()
    union=unary_union([box(x,y,x+w,y+h) for x,y,w,h in p]).area
    bbox=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)))
    return dict(repeated_coverage_fraction=float(max(0,total-union)/total),bbox_over_area=bbox/float(total))
