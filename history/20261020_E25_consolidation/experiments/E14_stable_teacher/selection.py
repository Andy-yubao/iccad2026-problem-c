"""Reference-free, certified non-regression for the neutral-time Q formula.

For F=1+.5*((A/Aref-1)+ + (W/Wref-1)+), F(new)<=r*F(old),
where r=max(1,Anew/Aold,Wnew/Wold). Multiplying by the exact soft factor
gives an upper bound valid for every positive unknown reference denominator.
The official cap preserves this ordering. Actual runtime is a separate metric.
"""
import math
import numpy as np
from scoring.official import official_evaluator


def public_metrics(x, inputs):
    _, areas,b2b,p2b,pins,c,specs=inputs
    official=official_evaluator()
    assert official.ALPHA == .5 and official.BETA == 2.
    # All fields come from allowed inputs. These arbitrary public denominators
    # are irrelevant: only raw wire/bbox/violation counts below are consumed.
    m=official.evaluate_solution({'positions':np.asarray(x).tolist(),'runtime':1.},
        {'hpwl_baseline':1.,'area_baseline':1.},c,b2b,p2b,pins,areas,specs.tolist(),median_runtime=1.)
    return {'area':float(m.bbox_area),'wire':float(m.hpwl_total),
            'soft':int(m.boundary_violations+m.grouping_violations+m.mib_violations),
            'boundary':int(m.boundary_violations),'grouping':int(m.grouping_violations),
            'mib':int(m.mib_violations),'feasible':bool(m.is_feasible)}


def bound(old,new,nsoft):
    if not new['feasible']:
        return float('inf')
    ratios=[1.]
    for key in ('area','wire'):
        if old[key] <= 0:
            if new[key] > 0:
                return float('inf')
        else:
            ratios.append(new[key]/old[key])
    return max(ratios)*math.exp(2.*(new['soft']-old['soft'])/max(1,nsoft))
