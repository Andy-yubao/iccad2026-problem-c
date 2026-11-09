"""Five actual train members; E74-architecture updates, capture and exact recovery."""
import gc,json,time
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,sources,snapshot
from .initialization import configure_trainability,verify_protected
from .arms import model_config
from .model import GeometryModel
from .data import load_rows,collate
from .batches import inputs_only,decode
from .loss import per_case,losses
from .loss_config import FIXED,load_coefficients
from .resident import ResidentRows
from .cuda_step import CapturedStep
from .train import parser,run,seed_all,PlateauController

IDS=[1683,10027,5474,18140,25001]

def verify():
    v=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if v['status']!='passed' or v['sources']!=sources():raise ValueError('Current-source five-case preflight required')
    verify_protected();load_coefficients(HERE/'runs/registered_loss.json')
    return v

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

def clear():
    from . import data
    data._resident.clear();gc.collect();torch.cuda.empty_cache()

def schedule_check():
    parameter=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([parameter],lr=1e-3)
    s=PlateauController(opt,start_epoch=15)
    for epoch in range(15):s.step(1.,epoch)
    assert s.best is None and s.drops==0 and s.lr_bad==0
    for epoch in range(15,24):s.step(1.,epoch)
    assert s.drops==2 and not s.should_stop()
    s.step(1.,24);assert s.drops==3 and not s.should_stop()
    s.step(1.,25);assert not s.should_stop()
    s.step(1.,26);assert s.should_stop() and s.lowest_completed_epochs==2
    return dict(protected_epochs='1-14',baseline_epoch=15,plateau_drops=[18,21,24],stop_epoch=26)

def main():
    begin=time.perf_counter();seed_all(0);protected=verify_protected()
    load_coefficients(HERE/'runs/registered_loss.json')
    session=HERE/('runs/preflight/'+time.strftime('%Y%m%d_%H%M%S'))
    rows=load_rows('train',IDS);raw=load_rows('train',IDS,use_cache=False)
    for a,b in zip(rows,raw):
        assert a['member']==b['member']
        for key,value in a['graph'].items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,b['graph'][key],rtol=0,atol=0)
    from .dataset_view import ReferenceLayouts
    assert len(ReferenceLayouts('train'))==32000
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role)
        except PermissionError:pass
        else:raise AssertionError('Sealed role accepted')
    batch=[rows[i%5] for i in range(32)]
    m=configure_trainability(GeometryModel(**model_config('simple')).cuda()).eval()
    assert all(p.requires_grad for p in m.parameters())
    pack=ResidentRows(batch,'cuda',max_nodes=120);g=pack.batch(list(range(32)))
    try:m(g)
    except ValueError:pass
    else:raise AssertionError('Network accepted supervision fields')
    with torch.no_grad():pred=m(inputs_only(g))
    v=per_case(pred,g,'simple')
    torch.testing.assert_close(v['total'],v['reference']+FIXED['overlap']*v['overlap_total']+FIXED['task']*v['task'])
    assert sha(HERE/'model.py')==sha(ROOT/'experiments/E74_collision_32k_posttrain/model.py')
    for name in ('simple_loss.py','features.py','batches.py'):
        assert sha(HERE/name)==sha(ROOT/'experiments/E90_auxiliary_weight_placer'/name)
    # The exact E74 decoder preserves the public hard specifications.
    pos=decode(pred,g)
    pre=(g['constraints'][...,1]!=0)&g['valid'];locked=(g['constraints'][...,:2]!=0).any(-1)&g['valid']
    torch.testing.assert_close(pos[pre],g['specs'][pre],rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(pos[...,2:][locked],g['specs'][...,2:][locked],rtol=1e-6,atol=1e-6)
    for i,row in enumerate(rows):
        one,_=collate([row],'cuda')
        with torch.no_grad():solo=per_case(m(inputs_only(one)),one,'simple')
        torch.testing.assert_close(v['total'][i],solo['total'][0],rtol=5e-5,atol=5e-5)
    changed={**g,'target':g['target']+1};changed_v=per_case(pred,changed,'simple')
    for key in ('overlap_total','task'):assert torch.equal(v[key],changed_v[key])
    assert not torch.equal(v['reference'],changed_v['reference'])
    assert torch.equal(losses(pred,g,epoch=0)['total'],losses(pred,g,epoch=200)['total'])
    initial_total=v['total'][:5].cpu().tolist()
    # One deterministic eager/captured step comparison, then separate real-Dropout recovery.
    m.train()
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    before={k:x.detach().clone() for k,x in m.state_dict().items()}
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-4)
    y=losses(m(inputs_only(g)),g);y['total'].backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
    grads={k:p.grad.detach().clone() for k,p in m.named_parameters() if p.grad is not None}
    assert all(float(grads[f'layers.{i}.local.weight'].abs().max())>0 for i in range(12))
    from .step_control import GraphStepControl
    bound=GraphStepControl(m);bound.before();opt.step();bound.after()
    for group,old in zip(bound.groups,bound.previous):
        norm=sum(x.square().sum() for x in old).sqrt()
        distance=sum((x-y).square().sum() for x,y in zip(group,old)).sqrt()
        assert float(distance)<=.005001*float(norm)+1e-7
    eager={k:x.detach().clone() for k,x in m.state_dict().items()}
    # Separate leaf parameters prevent eager-autograd stream history from entering
    # the capture comparison. Formal training also captures a fresh model.
    m=GeometryModel(**model_config('simple')).cuda().train()
    m.load_state_dict(before)
    for module in m.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-4)
    step=CapturedStep(m,opt,g,'simple')
    assert all(torch.equal(before[k],x) for k,x in m.state_dict().items()) and not opt.state
    step.replay()
    for key,x in m.state_dict().items():torch.testing.assert_close(x,eager[key],rtol=2e-5,atol=2e-5)
    # Nonfinite inputs must leave all parameters and Adam state untouched.
    guarded={k:x.detach().clone() for k,x in m.state_dict().items()}
    step.inputs['target'].fill_(float('nan'))
    try:step.replay()
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    else:raise AssertionError('Nonfinite update accepted')
    assert all(torch.equal(guarded[k],x) for k,x in m.state_dict().items())
    del m,opt,step,pack,g,pred,pos,v,changed,changed_v,before,grads,eager,guarded;clear()
    # Actual architecture with Dropout .05: interrupted and uninterrupted two-step Adam.
    args=parser().parse_args(['--output',str(session/'resume'),'--max-hours','.25','--max-epochs','2'])
    run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
    first=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    start=torch.load(args.output/'checkpoints/best.pt',map_location='cpu',weights_only=False)
    assert not start['optimizer']['state'] and start['progress']['step']==0
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    resumed=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    args.resume=False;args.output=session/'whole'
    run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    whole=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','graph_step_control'):same(resumed[key],whole[key])
    for key in ('torch','cuda'):same(resumed['rng'][key],whole['rng'][key])
    for i in range(12):
        key=f'layers.{i}.local.weight'
        assert not torch.equal(start['model'][key],resumed['model'][key])
    seed_all(0);fresh=GeometryModel(**model_config('simple'))
    for key,value in fresh.state_dict().items():assert torch.equal(value,start['model'][key])
    assert first['progress']['step']==1 and resumed['progress']['step']==2
    assert verify_protected()==protected
    schedules=schedule_check()
    save(HERE/'runs/preflight/results.json',dict(status='passed',sources=sources(),ids=IDS,unique_real_train_cases=5,
        raw_cache_bit_exact=True,train_count=32000,public_only=True,fixed_E90_loss=True,all_placer_parameters_trainable=True,
        all12_graph_local_gradients_nonzero=True,all12_graph_local_parameters_updated=True,random_constructor_matches_epoch0=True,fresh_empty_Adam=True,actual_relative_module_step_bound=.005,
        padded120_consistent=True,public_specs_preserved=True,initial_case_losses=initial_total,
        deterministic_eager_capture_step_equivalent=True,dropout_Adam_recovery_bit_exact=True,
        capture_does_not_update=True,nonfinite_rejected=True,scheduler=schedules,protected_unchanged=True,
        seconds=time.perf_counter()-begin,formal_training_started=False,test_read=False))
    snapshot(HERE/'runs/prepared_source_snapshot.zip',dict(sources=sources()))
    print('E91 E74-architecture five-case preflight passed; formal training NOT started',flush=True)

if __name__=='__main__':
    started=time.perf_counter()
    try:main()
    finally:
        from .budget import charge_preflight
        charge_preflight(time.perf_counter()-started)
