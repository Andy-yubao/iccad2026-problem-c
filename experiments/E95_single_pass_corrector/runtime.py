"""FP32 deterministic state, bounded updates, and cumulative wall accounting."""
import json,random,time
import numpy as np
import torch
from .io import HERE,ROOT,save

CONFIG=dict(width=192,layers=12,heads=4,dropout=.05,normalize_local=True,attention_backend='efficient')
OUTPUT=HERE/'runs/20261101_bbox_engine_resume'

def seed(value=0,threads=4):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)
    torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False

def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
def restore_rng(x):
    random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])

def checked_update(model,opt,g,draft):
    from .loss import per_case
    if not bool(torch.isfinite(draft).all()):
        raise RuntimeError('Nonfinite source draft; Adam not executed')
    model.train();opt.zero_grad(set_to_none=True)
    values=per_case(model.corrector(g,draft),g,draft)
    stack=torch.stack(list(values.values()))
    if not bool(torch.isfinite(stack).all()) or bool((values['total']>=1000).any()):
        raise RuntimeError('Nonfinite or excessive per-case Loss; Adam not executed')
    if not bool((model.corrector.last_hidden_rms<1000).all()):
        raise RuntimeError('Hidden-scale amplification; Adam not executed')
    values['total'].mean().backward()
    norm=torch.nn.utils.clip_grad_norm_(model.corrector.parameters(),1.,error_if_nonfinite=True)
    if float(norm)>=10000:raise RuntimeError('Excessive preclip gradient; Adam not executed')
    opt.step()
    from .step_engine import finite_parameters
    if not bool(finite_parameters(list(model.corrector.parameters()))):
        raise RuntimeError('Nonfinite parameters after Adam; stop, do not retry')
    keys=list(values);means=torch.stack([values[k].mean().detach() for k in keys]).cpu().tolist()
    return dict(zip(keys,means)),float(norm)

def charge(seconds,kind='preparation',output=None):
    path=(output if output is not None else HERE/'runs')/f'{kind}_budget.json'
    value=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(charged_seconds=0.,checks=[])
    value['charged_seconds']+=seconds
    value['checks'].append(dict(seconds=seconds,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    save(path,value)

def historical_ledger():
    # E94 records prior authorized windows. Preserve those identities; active
    # E94 charges are a point-in-time snapshot and cannot be treated as final.
    path=ROOT/'experiments/E94_posttrain_lr_reset/runs/authorization.json'
    if not path.exists():path=ROOT/'history/20261101_research_consolidation/experiments/E94_posttrain_lr_reset/runs/authorization.json'
    prior=json.loads(path.read_text(encoding='utf-8'))['historical_charged_seconds_readonly']
    folder=path.parent
    add=0.;artifacts={}
    for p in [folder/'preparation_budget.json',*folder.glob('20*_seed0/budget.json'),folder/'evaluation_budget.json']:
        if p.exists():
            v=json.loads(p.read_text(encoding='utf-8'));amount=v.get('elapsed_seconds',v.get('charged_seconds',0.));add+=amount
            artifacts[str(p.relative_to(ROOT))]=amount
    e95={}
    for name in ('preparation','evaluation'):
        p=HERE/f'runs/{name}_budget.json'
        if p.exists():e95[name]=json.loads(p.read_text(encoding='utf-8'))['charged_seconds']
    e95['family_training']=prior_family_seconds()
    return dict(prior_E94_snapshot_seconds=prior,E94_additional_snapshot_seconds=add,
        E95_prior_charges=e95,total_snapshot_seconds=prior+add+sum(e95.values()),
        artifacts=artifacts,policy='read-only historical charges; user authorized a new bbox window, no historical reset')

def prior_family_seconds():
    amounts=[]
    # V2 already includes V1 wall cost, so take the maximum rather than sum.
    for old in ('20261101_single_call_seed0','20261101_normalized_local_seed0'):
        for name in ('budget.json','progress.json','results.json'):
            path=HERE/'runs'/old/name
            if path.exists():amounts.append(json.loads(path.read_text(encoding='utf-8'))['elapsed_seconds'])
    base=max(amounts,default=0.)
    current=HERE/'runs/20261101_direct_bbox846_seed0'
    extra=[]
    for name in ('budget.json','progress.json','results.json'):
        p=current/name
        if p.exists():extra.append(json.loads(p.read_text(encoding='utf-8'))['elapsed_seconds'])
    return base+max(extra,default=0.)
