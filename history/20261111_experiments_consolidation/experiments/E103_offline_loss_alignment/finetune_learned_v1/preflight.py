"""Five train examples and one real high-LR update; no resume or Q tests."""
import ast
import time
import numpy as np
import torch
from .common import ROOT,OWNER,PLAN_PATH,read,save,sha,sources
from .runtime import setup
from .data import load_rows,register_resident
from .resident import ResidentRows
from .training import make_model,monitor
from .schedule import PhaseSchedule

def main():
    started=time.perf_counter();setup()
    for path in OWNER.glob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    plan=read(PLAN_PATH);co=plan['coefficients']
    assert co==read(ROOT/plan['coefficient_source'])['physical_coefficients']
    assert plan['maximum_epochs'] is None and plan['maximum_seconds'] is None and plan['protection_epochs']==0
    schedule=PhaseSchedule()
    assert not any(schedule.step(1.,epoch) for epoch in range(4))
    assert schedule.last_drop_epoch==3 and schedule.lr==2.5e-4
    assert not any(schedule.step(1.,epoch) for epoch in range(4,12))
    assert schedule.lr==5e-5 and schedule.step(1.,12)
    parent_path=ROOT/plan['source_checkpoint'];assert sha(parent_path)==plan['source_sha256']
    parent=torch.load(parent_path,map_location='cpu',weights_only=False)
    assert parent['progress']['epoch']==71
    rows=load_rows('train',plan['implementation_indices'])
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
    model,opt,cap,schedule=make_model(parent,pack,co)
    assert not opt.state and opt.param_groups[0]['lr']==1e-3
    measured=monitor(model,rows,co,lambda:None)
    model.train();cap.inputs['axis_swap'].copy_(torch.as_tensor(np.arange(32)%2==0,device='cuda'))
    before=model.layers[0].local.weight.detach().clone();cap.replay()
    assert not torch.equal(before,model.layers[0].local.weight)
    assert all(int(v['step'])==1 for v in opt.state.values())
    assert all(int(v['step'])==35500 for v in parent['optimizer']['state'].values())
    assert sha(parent_path)==plan['source_sha256']
    for entry in read(OWNER/'copy_provenance.json'):
        assert sha(ROOT/entry['copy'])==entry['sha256']
    save(OWNER/'preflight.json',dict(status='passed',sources=sources(),unique_train_cases=5,
        actual_updates=1,initial_lr=1e-3,fresh_Adam_age=1,parent_optimizer_age_unchanged=35500,
        parent_sha256=plan['source_sha256'],parent_best_epoch=71,parent_unchanged=True,
        coefficients_exact=True,no_epoch_or_time_cap=True,no_protection=True,
        constant_plateau_first_drop_epoch=3,constant_plateau_early_stop_epoch=12,
        axis_swap_implementation_unchanged=True,mixed_batch_swapped=16,monitor_loss=measured['total'],
        dev_payloads=0,postprocessing_calls=0,test_read=False,elapsed_seconds=time.perf_counter()-started))
    print('Passed: actual1e-3 Adam update on5 train examples, exact learned-v1 Loss, parent71 preserved, no protection or cap.',flush=True)

if __name__=='__main__':main()
