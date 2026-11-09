"""Retain prior work and E87/E90/E88 charges under the unchanged18h cap."""
import json,time
from .io import ROOT,HERE,save
PRIOR_SECONDS=38831.07569789997

def read_seconds(path,key):return json.loads(path.read_text(encoding='utf-8'))[key] if path.exists() else 0.

def global_charged_seconds(exclude=None):
    exclude=None if exclude is None else exclude.resolve();total=PRIOR_SECONDS
    for name in ('E87_simple_supervised_placer','E90_auxiliary_weight_placer','E88_sparse_training','E89_spatial_neural_refinement'):
        folder=ROOT/'experiments'/name/'runs';total+=read_seconds(folder/'preparation_budget.json','charged_seconds')
        for path in folder.glob('20*_seed0/budget.json'):
            if path.parent.resolve()!=exclude:total+=read_seconds(path,'elapsed_seconds')
    return total

def charge_preflight(seconds):
    path=HERE/'runs/preparation_budget.json'
    old=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(charged_seconds=0.,checks=[])
    old['charged_seconds']+=seconds;old['checks'].append(dict(seconds=seconds,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    old.update(policy='Charge complete wall of every updating short check, including failures',test_read=False);save(path,old)
