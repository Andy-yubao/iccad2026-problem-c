"""Independent authorized paired budget; all older ledgers remain read-only."""
import json,time
from .io import HERE,ROOT,save

def read(path,key):return json.loads(path.read_text(encoding='utf-8'))[key] if path.exists() else 0.

def historical_charged_seconds():
    total=38831.07569789997
    for name in ('E87_simple_supervised_placer','E90_auxiliary_weight_placer','E88_sparse_training',
                 'E89_spatial_neural_refinement','E91_e74_32k_scratch','E92_two_stage_placer','E93_matched_posttrain'):
        folder=ROOT/'experiments'/name/'runs'
        total+=read(folder/'preparation_budget.json','charged_seconds')
        for path in folder.glob('20*_seed0/budget.json'):total+=read(path,'elapsed_seconds')
        if name=='E92_two_stage_placer':total+=read(folder/'task_auxiliary_budget.json','charged_seconds')
        if name=='E93_matched_posttrain':
            total+=read(folder/'evaluation_budget.json','charged_seconds')
            total+=read(folder/'evaluation/preparation.json','preparation_wall_seconds_conservative')
    return total

def branch_charged_seconds(exclude=None):
    exclude=None if exclude is None else exclude.resolve();total=0.
    for path in (HERE/'runs').glob('20*_seed0/budget.json'):
        if path.parent.resolve()!=exclude:total+=read(path,'elapsed_seconds')
    return total

def charge_preflight(seconds):
    path=HERE/'runs/preparation_budget.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'charged_seconds':0.,'checks':[]}
    value['charged_seconds']+=seconds
    value['checks'].append({'seconds':seconds,'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    value.update(policy='Full wall of actual updating preflight including failures; outside new2h training window',test_read=False)
    save(path,value)
