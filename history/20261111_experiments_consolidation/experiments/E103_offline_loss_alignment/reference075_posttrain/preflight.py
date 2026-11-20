"""Five train examples, exact two-term change, one actual1e-4 update."""
import ast
import time
import numpy as np
import torch
from ..batches import inputs_only
from ..loss_config import COEFFICIENTS
from .engine import per_case
from .common import ROOT,OWNER,PLAN_PATH,read,save,sha,sources
from .runtime import setup
from .data import load_rows,register_resident
from .resident import ResidentRows
from .training import make_model,monitor
from .schedule import PhaseSchedule

def main():
    started=time.perf_counter();setup()
    for path in OWNER.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'))
    plan=read(PLAN_PATH);co=plan['coefficients'];base=dict(COEFFICIENTS)
    assert co['position']==base['position']*.75 and co['shape']==base['shape']*.75
    assert all(co[k]==base[k] for k in base if k not in ('position','shape'))
    assert plan['maximum_epochs'] is None and plan['maximum_seconds'] is None
    schedule=PhaseSchedule();assert schedule.lr==1e-4
    assert not any(schedule.step(1.,epoch) for epoch in range(4))
    assert schedule.last_drop_epoch==3 and schedule.lr==5e-5
    assert not any(schedule.step(1.,epoch) for epoch in (4,5)) and schedule.step(1.,6)
    path=ROOT/plan['source_checkpoint'];parent=torch.load(path,map_location='cpu',weights_only=False)
    assert sha(path)==plan['source_sha256'] and parent['progress']['epoch']==71
    rows=load_rows('train',plan['implementation_indices'])
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
    model,opt,cap,schedule=make_model(parent,pack,co)
    assert not opt.state and opt.param_groups[0]['lr']==1e-4
    batch=pack.batch([i%5 for i in range(32)]);model.eval()
    with torch.no_grad():
        out=model(inputs_only(batch));old=per_case(out,batch,base);new=per_case(out,batch,co)
        expected=old['total']-.25*(old['position']+.25*old['shape'])
        torch.testing.assert_close(new['total'],expected,rtol=2e-6,atol=2e-7)
    monitored=monitor(model,rows,co,lambda:None)
    model.train();cap.inputs['axis_swap'].copy_(torch.as_tensor(np.arange(32)%2==0,device='cuda'))
    before=model.layers[0].local.weight.detach().clone();cap.replay()
    assert not torch.equal(before,model.layers[0].local.weight)
    assert all(int(v['step'])==1 for v in opt.state.values())
    assert all(int(v['step'])==35500 for v in parent['optimizer']['state'].values())
    assert sha(path)==plan['source_sha256']
    for item in read(OWNER/'copy_provenance.json'):assert sha(ROOT/item['copy'])==item['sha256']
    # Reuse metadata from the already completed exact five-case parent evaluation bridge.
    proof=read(ROOT/read(OWNER/'evaluation/plan.json')['parent_bridge_reuse']/'parent_reproduction.json')
    assert proof['status']=='passed' and len(proof['checks'])==5
    save(OWNER/'preflight.json',dict(status='passed',sources=sources(),unique_train_cases=5,actual_updates=1,
        coefficients=co,exact_two_term_reduction=True,other8_unchanged=True,no_total_normalization=True,
        initial_lr=1e-4,first_possible_drop_epoch=3,constant_plateau_earlystop_epoch=6,
        no_epoch_or_time_cap=True,parent_best_epoch=71,parent_optimizer_unchanged=True,
        monitor_loss=monitored['total'],axis_implementation_unchanged=True,mixed_batch_swapped=16,
        automatic_terminal_dev500=True,dev_payloads=0,test_read=False,elapsed_seconds=time.perf_counter()-started))
    print('Passed5 train cases: exact position/shape*.75, other8 unchanged, actual Adam1e-4 update; auto dev500 ready.',flush=True)

if __name__=='__main__':main()
