"""Retain the E10 normal solve and add certified, input-triggered rescue."""
import time
from .frozen.packing import prepare_problem
from .frozen.feasible import Objective
from .frozen.solver import hard_valid
from .rescue import refine,seed_pack
from .selection import public_metrics,bound

POLICY='E14.certified_rescue.v2'


def polish(draft, baseline, inputs, baseline_stats, *, seconds=16.):
    begin=time.perf_counter();p=prepare_problem(*inputs);obj=Objective(p)
    best=baseline.copy();original=public_metrics(best,inputs);current=original
    risk=bool(baseline_stats.get('rescue_used') or baseline_stats.get('initial_row_retained')
              or original['soft']/obj.nsoft >= .06 or original['area']/obj.area_scale>1.5
              or obj(best)>.6)
    stats={'policy':POLICY,'triggered':risk,'accepted':[],'candidates':[],
           'original_public':original,'additional_budget_seconds':seconds,
           'certificate':'max(1,Anew/Aold,Wnew/Wold)*exp(2*(Vnew-Vold)/Nsoft)<=1'}
    if risk:
        for phase in ['incumbent_boundary','square_repack','draft_repack']:
            left=seconds-(time.perf_counter()-begin)
            if left<.15:
                break
            seed=best if phase=='incumbent_boundary' else seed_pack(draft,inputs,2 if phase=='square_repack' else 0,seconds=min(1.,left*.2))
            if seed is None:
                stats['candidates'].append({'phase':phase,'error':'invalid_seed'})
                continue
            x,search=refine(seed,inputs,seconds=min(5.,max(.05,seconds-(time.perf_counter()-begin))),
                            mode=1,boundary=40.,contact=20.)
            record={'phase':phase,'seconds':search['seconds'],'legal':x is not None,
                    'trace':search['trace']}
            if x is not None:
                proposed=public_metrics(x,inputs)
                factor=bound(current,proposed,obj.nsoft)
                record.update(public_metrics=proposed,certified_ratio_bound=factor)
                strictly_better=(proposed['soft']<current['soft'] or
                    proposed['area']<current['area']*(1.-1e-8) or
                    proposed['wire']<current['wire']*(1.-1e-8))
                if factor<=1. and strictly_better:
                    best=x;current=proposed
                    stats['accepted'].append({'phase':phase,'ratio_bound':factor})
            stats['candidates'].append(record)
            # Healthy compact geometry plus small residual soft penalty is no
            # longer a tail rescue. Avoid further expensive seed exploration.
            if current['soft']/obj.nsoft<=.04 and current['area']/obj.area_scale<=1.25:
                break
    assert hard_valid(best,p)
    stats.update(seconds=time.perf_counter()-begin,final_public=current,
                 certified_ratio_bound=bound(original,current,obj.nsoft),
                 baseline_retained=not bool(stats['accepted']))
    return best,stats
