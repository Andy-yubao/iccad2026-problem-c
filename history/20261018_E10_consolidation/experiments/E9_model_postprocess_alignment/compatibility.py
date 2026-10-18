"""Public-only strict MIB compatibility witness for a future separate panel.

This tests local MIB/area/locked-dimension compatibility, not joint geometric
feasibility of every hard and soft constraint. It never reads datasets itself.
"""
import math
import numpy as np


def inspect_mib(inputs):
    n,areas,_,_,_,constraints,specs=inputs
    areas=np.asarray(areas,dtype=float);c=np.asarray(constraints);specs=np.asarray(specs,dtype=float)
    if areas.shape!=(int(n),) or c.shape!=(int(n),5) or specs.shape!=(int(n),4):
        raise ValueError('Invalid public input dimensions')
    if not np.isfinite(areas).all() or (areas<=0).any():raise ValueError('Invalid public areas')
    locked=(c[:,0]!=0)|(c[:,1]!=0);groups=[]
    # Verify input masking, so future selection cannot accidentally use free
    # reference shapes or locations even when given an unmasked data object.
    if (specs[~locked]!=-1).any() or (specs[(c[:,0]!=0)&(c[:,1]==0),:2]!=-1).any():
        raise ValueError('Only masked public specifications are accepted')
    for label in sorted(set(c[:,2].astype(int))-{0}):
        ids=np.flatnonzero(c[:,2]==label);rigid=ids[locked[ids]];free=ids[~locked[ids]]
        lower=float((.99000001*areas[free]).max()) if len(free) else 0.
        upper=float((1.00999999*areas[free]).min()) if len(free) else math.inf
        witness=None;reason='compatible'
        if len(rigid):
            wh=specs[rigid[0],2:]
            if not np.isfinite(wh).all() or (wh<=0).any():raise ValueError('Missing locked dimensions')
            if not np.allclose(specs[rigid,2:],wh,rtol=0,atol=1e-10):
                reason='locked_dimensions_disagree'
            elif not lower<=float(wh.prod())<=upper:
                reason='locked_shape_incompatible_with_free_area'
            else:witness=wh.tolist()
        elif lower>upper:reason='free_area_intersection_empty'
        else:
            area=(lower+upper)/2;side=math.sqrt(area);witness=[side,side]
        groups.append({'group':int(label),'blocks':ids.tolist(),'compatible':witness is not None,
                       'reason':reason,'common_shape_witness':witness,
                       'shared_area_interval':[lower,upper if math.isfinite(upper) else None]})
    return {'eligible_for_local_mib_compatibility':all(g['compatible'] for g in groups),
            'groups':groups,'scope':'Strict exact common dimensions; conservative filtering with margins. Does not prove all constraints jointly feasible.'}
