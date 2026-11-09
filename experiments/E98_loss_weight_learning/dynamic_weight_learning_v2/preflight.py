"""One real update on five train cases; frozen-parent safety and prepared entrypoint."""
import ast
import time
import numpy as np
import torch
from ..io import save
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
    assert len(set(plan['feedback_pool_train_indices']))==1000
    assert all(len(set(r['feedback_train_indices']))==100 and
        set(r['feedback_train_indices'])<=set(plan['feedback_pool_train_indices']) for r in plan['rounds'])
    assert abs(plan['log_perturbation']-np.log(1.2))<1e-15
    seed_all(0)
    rows=load_rows('train',[1683,10027,5474,10140,15001])
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
    register_resident(pack)
    initial=GeometryModel(**MODEL).cuda()
    parent=dict(model={k:v.cpu().clone() for k,v in initial.state_dict().items()},rng=rng_state(),projection=None)
    before_hash=model_hash(parent['model'])
    assert before_hash==model_hash({k:v.clone() for k,v in parent['model'].items()})
    del initial
    co=coefficients(plan['initial_coefficients'],plan['rounds'][0]['direction'],1,plan['log_perturbation'])
    model,opt,cap,schedule=make_model(parent,pack,co)
    assert not opt.state
    cap.inputs['axis_swap'].copy_(torch.as_tensor(np.arange(32)%2==0,device='cuda'))
    cap.replay()
    assert all(int(v['step'])==1 for v in opt.state.values())
    assert before_hash!=model_hash(model.state_dict())
    assert before_hash==model_hash(parent['model'])
    same={a:[dict(hard_legal=100,mixed_Q=1.1)] for a in ('control','plus','minus')}
    assert choose(same,plan)[0]=='control'
    save(OWNER/'preflight.json',dict(status='passed',sources=sources(),unique_train_cases=5,
        actual_updates=1,train_cases=16000,feedback_pool=1000,each_feedback_panel=100,
        parent_unchanged=True,model_tensor_identity_verified=True,same_Q_retains_parent=True,
        mixed_batch_swapped=16,test_read=False,dev_payloads=0,formal_search_started=False,
        baseline_freeze_pending=True,elapsed_seconds=time.perf_counter()-started,
        note='Implementation check while independent baseline trains; not a throughput benchmark'))
    print('Prepared v2 passed: one update, frozen parent unchanged, strict100/1000, single feedback, 15epoch cap, factor1.2; no search launched.',flush=True)

if __name__=='__main__':main()
