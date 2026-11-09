"""Editable numerical budgets, using the unchanged V3 deformation algorithm.

Only validation/defaults change; V3's original Config and evidence stay frozen.
"""
from dataclasses import dataclass,fields
from numbers import Integral
from pathlib import Path
import inspect,json
import numpy as np
from .deformation import Config as V3Config


@dataclass(frozen=True)
class Parameters(V3Config):
    iterations: int = 20
    step_fraction: float = .05
    total_fraction: float = .4
    shape_step_fraction: float = .03
    shape_total_fraction: float = .5

    def __post_init__(self):
        if isinstance(self.iterations,bool) or not isinstance(self.iterations,Integral) or self.iterations<0:
            raise ValueError('iterations must be a nonnegative integer')
        if self.mass_weighted is not True or not isinstance(self.shape_enabled,bool):
            raise ValueError('area mass must remain enabled; shape_enabled must be boolean')
        for item in fields(self):
            if item.name in ('iterations','mass_weighted','shape_enabled'):continue
            v=getattr(self,item.name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<=0:
                raise ValueError(f'{item.name} must be a positive finite number')


def load(path):
    value=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    allowed={'schema_version','label','relaxation','planner','comparison','render_views'}
    if set(value)-allowed:raise ValueError('Unknown top-level settings: '+str(sorted(set(value)-allowed)))
    if value.get('schema_version')!=1:raise ValueError('schema_version must be1')
    if not isinstance(value.get('label'),str) or not value['label'].strip():raise ValueError('label is required')
    config=Parameters(**value['relaxation'])
    comparison=value.get('comparison',{})
    if set(comparison)-{'baseline_mode','previous_best_run'}:raise ValueError('Unknown comparison settings')
    mode=comparison.get('baseline_mode','auto')
    if mode not in ('auto','fresh'):raise ValueError('baseline_mode must be auto or fresh')
    previous=comparison.get('previous_best_run','auto')
    if not isinstance(previous,str) or not previous:raise ValueError('previous_best_run must be auto or a saved run path')
    views=value.get('render_views',['four_stage'])
    if views==['pipeline','comparison']:views=['four_stage']
    if views!=['four_stage']:raise ValueError('The four requested stages are required')
    from .planner.solver import optimize
    signature=inspect.signature(optimize)
    keys=set(signature.parameters)-{'draft','inputs'}
    if not isinstance(value.get('planner'),dict) or set(value['planner'])-keys:
        raise ValueError('Unknown planner settings')
    value['comparison']={'baseline_mode':mode,'previous_best_run':previous};value['render_views']=views
    return value,config
