"""Five train members; exact two-term coefficient change and one real update."""
import ast
import time
import numpy as np
import torch
from ..io import ROOT,HERE
from ..generalization_baseline.train import seed_all
from ..generalization_baseline.data import load_rows,register_resident
from ..resident import ResidentRows
from ..dynamic_weight_learning_v2.engine import per_case
from .common import OWNER,PLAN_PATH,read,sha,save,sources
from .training import make_model,monitor

def main():
    started=time.perf_counter()
    for path in OWNER.glob('*.py'):ast.parse(path.read_text(encoding='utf-8'))
    plan=read(PLAN_PATH);base=plan['initial_coefficients'];co=plan['candidate_coefficients']
    assert co['position']==base['position']/1.2 and co['shape']==base['shape']/1.2
    assert all(co[k]==v for k,v in base.items() if k not in ('position','shape'))
    assert len(set(plan['feedback_pool_train_indices']))==1000
    assert len(set(plan['feedback_train_indices']))==100 and set(plan['feedback_train_indices'])<=set(plan['feedback_pool_train_indices'])
    assert plan['outer_rounds']==plan['candidate_count']==1 and not plan['other_directions']
    seed_all(0)
    parent=torch.load(ROOT/plan['parent_checkpoint'],map_location='cpu',weights_only=False)
    assert parent['progress']['epoch']==71 and sha(ROOT/plan['parent_checkpoint'])==plan['parent_sha256']
    rows=load_rows('train',plan['implementation_train_indices'])
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146);register_resident(pack)
    model,opt,cap,schedule=make_model(parent,pack,co)
    assert not opt.state and opt.param_groups[0]['lr']==1e-4
    batch=pack.batch([i%5 for i in range(32)])
    model.eval()
    with torch.no_grad():
        from ..batches import inputs_only
        out=model(inputs_only(batch));a=per_case(out,batch,base);b=per_case(out,batch,co)
        expected=a['total']-(1-1/1.2)*(a['position']+.25*a['shape'])
        torch.testing.assert_close(b['total'],expected,rtol=2e-6,atol=2e-7)
    monitored=monitor(model,rows,co,lambda:None)
    model.train();cap.inputs['axis_swap'].copy_(torch.as_tensor(np.arange(32)%2==0,device='cuda'))
    before=model.layers[0].local.weight.detach().clone();cap.replay(batch)
    assert not torch.equal(before,model.layers[0].local.weight)
    assert all(int(v['step'])==1 for v in opt.state.values())
    assert all(int(v['step'])==35500 for v in parent['optimizer']['state'].values())
    assert (HERE/'runs/20261104_dynamic_mixedq_seed0/STOP').exists()
    save(OWNER/'preflight.json',dict(status='passed',sources=sources(),unique_train_cases=5,actual_updates=1,
        parent_best_epoch=71,parent_sha256=plan['parent_sha256'],parent_unchanged=True,
        coefficients=co,other8_coefficients_unchanged=True,no_total_sum_normalization=True,
        fixed_reference_mask_and_axis_transform='unchanged validated generalization implementation',
        mixed_axis_flags=16,monitor_loss=monitored['total'],test_read=False,dev_payloads=0,
        elapsed_seconds=time.perf_counter()-started))
    print('Passed: single two-term change only, fresh Adam update1, original best71 preserved, train100/1000 registered.',flush=True)

if __name__=='__main__':main()
