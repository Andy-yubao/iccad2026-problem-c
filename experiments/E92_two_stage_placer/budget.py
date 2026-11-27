"""All historical charges retained plus E92 checks; no double charging."""
import json,time
from .io import ROOT,HERE,save
PRIOR_SECONDS=38831.07569789997
NAMES=('E87_simple_supervised_placer','E90_auxiliary_weight_placer','E88_sparse_training',
       'E89_spatial_neural_refinement','E91_e74_32k_scratch','E92_two_stage_placer')

def read_seconds(path,key):
    return json.loads(path.read_text(encoding='utf-8'))[key] if path.exists() else 0.

def global_charged_seconds(exclude=None):
    exclude=None if exclude is None else exclude.resolve()
    total=PRIOR_SECONDS
    for name in NAMES:
        folder=ROOT/'experiments'/name/'runs'
        total+=read_seconds(folder/'preparation_budget.json','charged_seconds')
        # E91 diagnostic_budget.json describes a check already in its preparation
        # ledger (59.0486811 seconds). Do not charge that same wall twice.
        for path in folder.glob('20*_seed0/budget.json'):
            if path.parent.resolve()!=exclude:
                total+=read_seconds(path,'elapsed_seconds')
    return total

def charge_preflight(seconds):
    path=HERE/'runs/preparation_budget.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'charged_seconds':0.,'checks':[]}
    value['charged_seconds']+=seconds
    value['checks'].append({'seconds':seconds,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    value.update(policy='Full updating-preflight wall, including failure; no reset',test_read=False)
    save(path,value)
