"""Explicit budget-only migration, retaining every numerical training state."""
import copy,hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
import numpy as np
import torch
from experiments.E92_two_stage_placer.io import HERE,ROOT,sha,save,sources,snapshot,checkpoint
from experiments.E92_two_stage_placer.preflight import verify
from experiments.E92_two_stage_placer.train import RUN

def digest(value):
    h=hashlib.sha256()
    def walk(v):
        if isinstance(v,torch.Tensor):
            h.update(b'tensor'+str(v.dtype).encode()+repr(tuple(v.shape)).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):
            h.update(b'array'+str(v.dtype).encode()+repr(v.shape).encode()+v.tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for key in sorted(v,key=repr):walk(key);walk(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for item in v:walk(item)
        else:h.update(repr(v).encode())
    walk(value);return h.hexdigest()

def main():
    verify()
    folder=Path(__file__).resolve().parent
    if (folder/'migration.json').exists():raise RuntimeError('Migration already recorded; do not reapply')
    latest=RUN/'checkpoints/latest.pt'
    if sha(latest)!=sha(folder/'latest.pt'):raise ValueError('Run advanced after safe stop')
    state=torch.load(latest,map_location='cpu',weights_only=False)
    if state['status']!='user_stop' or state['progress']['step']!=711:raise ValueError('Wrong safe boundary')
    old_identity=copy.deepcopy(state['identity'])
    with zipfile.ZipFile(folder/'source_snapshot.zip') as archive:
        for name,expected in old_identity['sources'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=expected:
                raise ValueError('Old source snapshot corrupt: '+name)
    current_sources=sources()
    changed={name for name in set(old_identity['sources'])|set(current_sources)
             if old_identity['sources'].get(name)!=current_sources.get(name)}
    allowed={f'experiments/E92_two_stage_placer/{name}' for name in ('train.py','pipeline.py','preflight.py','io.py','plan.json','README.md')}
    if not changed<=allowed:raise ValueError('Unexpected algorithm/data changes: '+repr(changed-allowed))
    new_plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
    plan_changes={key for key in set(new_plan)|set(old_identity['plan']) if new_plan.get(key)!=old_identity['plan'].get(key)}
    if not plan_changes<={'global_cap_seconds','budget_policy','budget_extension_authorization'}:
        raise ValueError('Non-budget plan change')
    keys=('model','optimizer','scheduler','rng','progress','graph_step_control')
    before={key:digest(state[key]) for key in keys}
    state['identity']={**old_identity,'plan':new_plan,'sources':current_sources}
    old_elapsed=state['elapsed_seconds']
    state['elapsed_seconds']=max(old_elapsed,json.loads((RUN/'budget.json').read_text())['elapsed_seconds'])
    checkpoint(latest,state)
    reloaded=torch.load(latest,map_location='cpu',weights_only=False)
    after={key:digest(reloaded[key]) for key in keys}
    assert before==after
    snapshot(folder/'source_snapshot_after.zip',state['identity'])
    config=json.loads((RUN/'config.json').read_text(encoding='utf-8'))
    config['identity']=state['identity'];config['budget_only_migration']='runs/budget2h_extension/migration.json'
    save(RUN/'config.json',config)
    report={'status':'passed','authorization':'用户要求完整2h并每小时跟进，E90达标后条件研究修正模型',
            'old_checkpoint_sha256':sha(folder/'latest.pt'),'new_checkpoint_sha256':sha(latest),
            'state_fingerprints_before':before,'state_fingerprints_after':after,
            'model_Adam_scheduler_RNG_progress_projection_unchanged':True,
            'accepted_steps':state['progress']['step'],'cursor':state['progress']['cursor'],
            'epoch':state['progress']['epoch'],'phase':state['progress']['phase'],
            'lr':state['optimizer']['param_groups'][0]['lr'],
            'elapsed_before_checkpoint':old_elapsed,'elapsed_preserved':state['elapsed_seconds'],
            'layout_cumulative_cap_seconds':7200,'scoped_global_cap_seconds':new_plan['global_cap_seconds'],
            'changed_sources':sorted(changed),'plan_changed_keys':sorted(plan_changes),
            'test_read':False}
    save(folder/'migration.json',report)
    print(json.dumps({k:v for k,v in report.items() if not k.startswith('state_fingerprints')},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
