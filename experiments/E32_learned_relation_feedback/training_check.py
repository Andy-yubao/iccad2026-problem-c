"""Reuse the original five train members for one batch and exact optimizer restore."""
from .resources import configure
RESOURCE=configure()
import copy
import json
import random
import numpy as np
import torch
from datasets.reference_layouts import ReferenceLayouts
from .data import load_rows
from .train import batch,monitor,rng_state,restore_rng
from .prepare import load_states
from .relations import RelationPolicy,objective
from .batches import inputs_only
from .schedule import PlateauController
from .io import HERE,save,checkpoint

def main():
    for role in ('test','tail_test'):
        try:ReferenceLayouts(role)
        except PermissionError:pass
        else:raise ValueError('Sealed role unexpectedly allowed')
    random.seed(7);np.random.seed(7);torch.manual_seed(7);torch.cuda.manual_seed_all(7);torch.use_deterministic_algorithms(True)
    members=json.loads((HERE/'runs/20261022_interface_check5/config.json').read_text())['members']
    rows=load_rows('train');mapping={tuple(r['member']):r for r in rows};rows=[mapping[tuple(m)] for m in members];states,_=load_states('train')
    model=RelationPolicy().cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    def update(m,o):
        selected=[rows[i%len(rows)] for i in range(32)]
        g,d,c=batch(selected,states,'cuda',np.random.randint(0,3,32));o.zero_grad(set_to_none=True)
        loss=objective(m(inputs_only(g),d,c),g)['per_case'].mean();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);o.step()
    update(model,optimizer)
    saved={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':rng_state()}
    update(model,optimizer);expected=copy.deepcopy(model.state_dict())
    restored=RelationPolicy().cuda();restored.load_state_dict(saved['model']);other=torch.optim.AdamW(restored.parameters(),lr=1e-3);other.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);update(restored,other)
    assert all(torch.equal(expected[k],restored.state_dict()[k]) for k in expected)
    parameter=torch.nn.Parameter(torch.tensor(1.));test_optimizer=torch.optim.SGD([parameter],lr=1e-3);scheduler=PlateauController(test_optimizer)
    for epoch in range(27):
        scheduler.step(1.,epoch)
        if epoch<26:assert not scheduler.should_stop()
    assert scheduler.should_stop() and scheduler.drops==3 and scheduler.lowest_completed_epochs==2
    values=monitor(restored,rows,states,'cuda');assert np.isfinite(values['selection_proxy'])
    folder=HERE/'runs/20261022_training_check5';folder.mkdir(parents=True,exist_ok=False)
    checkpoint(folder/'checkpoints/best.pt',{'model':restored.cpu().state_dict(),'model_config':{'width':64},'progress':{'best_epoch':0},'implementation_check':True})
    save(folder/'results.json',{'same_five_members':members,'batch_size':32,'repeated_members_for_batch_check':True,'finite_backward':True,'exact_optimizer_resume':True,'three_lr_drops_lowest_two_epochs':True,'sealed_roles_rejected_before_payload_read':True,'monitor':values,'test_read':False,'resource':RESOURCE})
    print('Same-five training and exact restore passed',flush=True)

if __name__=='__main__':main()
