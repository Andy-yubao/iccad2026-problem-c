"""One real update on five train cases; frozen-parent safety and prepared entrypoint."""
import ast
import time
import numpy as np
import torch
from ..io import ROOT,HERE,save,sha
from ..generalization_baseline.train import seed_all
from ..generalization_baseline.data import load_rows,register_resident
from ..resident import ResidentRows
from ..train import rng_state
from .common import OWNER,PLAN_PATH,MODEL,read,sources,coefficients
from .run import make_model
from .evaluation import model_hash,choose
from ..generalization_baseline.model import GeometryModel

def main():
    started=time.perf_counter()
    for path in OWNER.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    plan=read(PLAN_PATH)
    assert plan['feedback_repeats']==1 and plan['maximum_epochs_per_arm']==15
    assert len(set(plan['feedback_train_indices']))==500
    assert all(0<=i<16000 for i in plan['feedback_train_indices'])
    assert len(plan['feedback_members'])==500 and len(plan['rounds'])==10
    assert [r['kind'] for r in plan['rounds']]==['designed']*5+['seeded_random']*5
    assert (HERE/'runs/20261104_dynamic_mixedq_seed0/STOP').exists()
    assert abs(plan['log_perturbation']-np.log(1.2))<1e-15
    seed_all(0)
    rows=load_rows('train',[1683,10027,5474,10140,15001])
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
    register_resident(pack)
    parent=torch.load(ROOT/plan['initial_checkpoint'],map_location='cpu',weights_only=False)
    assert sha(ROOT/plan['initial_checkpoint'])==plan['initial_sha256']
    assert parent['progress']['epoch']==71
    before_hash=model_hash(parent['model'])
    co=coefficients(plan['initial_coefficients'],plan['rounds'][0]['direction'],1,plan['log_perturbation'])
    model,opt,cap,schedule=make_model(parent,pack,co)
    assert not opt.state
    cap.inputs['axis_swap'].copy_(torch.as_tensor(np.arange(32)%2==0,device='cuda'))
    cap.replay()
    assert all(int(v['step'])==1 for v in opt.state.values())
    assert before_hash!=model_hash(model.state_dict())
    assert before_hash==model_hash(parent['model'])
    same={a:[dict(hard_legal=500,mixed_Q=1.1)] for a in ('control','plus','minus')}
    assert choose(same,plan)[0]=='control'
    lower={a:[dict(hard_legal=500,mixed_Q=q)] for a,q in [('control',1.1),('plus',1.05),('minus',1.08)]}
    assert choose(lower,plan)[0]=='plus'
    lower['plus'][0]['hard_legal']=499
    assert choose(lower,plan)[0]=='minus'
    from .common import PhaseSchedule
    schedule=PhaseSchedule()
    assert not any(schedule.step(1.,epoch) for epoch in range(6))
    assert schedule.drop_epoch==5 and schedule.lr==5e-5
    for spec in plan['rounds']:
        for sign in (-1,1):
            moved=coefficients(plan['initial_coefficients'],spec['direction'],sign,plan['log_perturbation'])
            assert abs(sum(moved.values())-sum(plan['initial_coefficients'].values()))<1e-12
    assert all(int(v['step'])==35500 for v in parent['optimizer']['state'].values())
    assert sha(ROOT/plan['initial_checkpoint'])==plan['initial_sha256']
    save(OWNER/'preflight.json',dict(status='passed',sources=sources(),unique_train_cases=5,
        actual_updates=1,train_cases=16000,feedback_source_population=16000,each_feedback_panel=500,
        parent_unchanged=True,model_tensor_identity_verified=True,same_Q_retains_parent=True,
        mixed_batch_swapped=16,test_read=False,dev_payloads=0,formal_search_started=False,
        baseline_frozen_best_epoch=71,control_zero_updates=True,control_summary_reused=True,elapsed_seconds=time.perf_counter()-started,
        note='One update from actual frozen baseline; five train cases only; no dev/test or search feedback'))
    print('Prepared v3 passed: actual best71 parent unchanged, one fresh Adam update, fixed500/16000, selection and schedule checks; no search launched.',flush=True)

if __name__=='__main__':main()
