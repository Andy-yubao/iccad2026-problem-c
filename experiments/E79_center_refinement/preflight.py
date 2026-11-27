"""Five real cases: exact source shapes, center limits, guard and true resume."""
import gc,time
import torch
from .io import HERE,save,sources
from .initialization import load,verify_protected
from .model import GeometryModel,decode
from .warm_model import GeometryModel as OldModel
from .source_guard import regression
from .arms import model_config
from .batches import inputs_only
from .data import load_rows,collate
from .loss import per_case,losses
from .train import seed_all,parser,run
from .cuda_step import CapturedStep

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
def check_geometry(state,g,frame):
    p=decode(state,g);source=decode(state[...,3:],g);valid=g['valid'];lo,hi=frame
    assert torch.equal(p[...,2:],source[...,2:])
    assert bool((p[...,:2]>=lo[:,None]-2e-6).masked_fill(~valid[...,None],True).all())
    assert bool((p[...,:2]+p[...,2:]<=hi[:,None]+2e-6).masked_fill(~valid[...,None],True).all())
    torch.testing.assert_close(p[...,2:].prod(-1)[valid],g['area'][valid],rtol=3e-6,atol=2e-8)
    placed=(~g['mask'][...,0].bool())&valid;locked=(~g['mask'][...,2].bool())&valid
    assert torch.equal(p[placed],g['specs'][placed]);assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
    local=g['area'].clamp_min(1e-12).sqrt()
    shift=(state[...,:2]-state[...,3:5]).abs()/local[...,None]
    assert float(shift[valid].max())<=.15001
    return float(shift[valid].max())
def synthetic():
    d='cuda';g={'area':torch.ones(1,2,device=d),'valid':torch.ones(1,2,dtype=torch.bool,device=d),
        'mask':torch.ones(1,2,3,device=d),'specs':torch.zeros(1,2,4,device=d)}
    s=torch.tensor([[[.5,.5,0.],[2.5,.5,0.]]],device=d)
    zero,flips=regression(s,s,g);assert float(zero)==0 and float(flips)==0
    farther=s.clone();farther[0,1,0]+=1
    healthy,_=regression(farther,s,g);assert float(healthy)==0
    different=s.clone();different[0,1,1]+=4;different.requires_grad_()
    penalty,flips=regression(different,s,g);assert float(penalty)>0 and float(flips)>0
    gradient=torch.autograd.grad(penalty.sum(),different)[0];assert torch.isfinite(gradient).all() and bool((gradient!=0).any())
def main():
    begin=time.perf_counter();seed_all(0);protected=verify_protected();synthetic();ids=[1683,10027,5474,18140,25001]
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for raw,prepared in zip(rows,cached):
        assert raw['member']==prepared['member']
        for k,v in raw['graph'].items():
            if isinstance(v,torch.Tensor):same(v,prepared['graph'][k])
            else:assert v==prepared['graph'][k]
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda');pub=inputs_only(g)
    model=GeometryModel(**model_config('topology')).cuda().eval();source=load(model,model_config('topology'))
    old=OldModel(**source['model_config']).cuda().eval();old.load_state_dict(source['model'])
    assert all(torch.equal(x.cpu(),source['model'][k]) for k,x in model.source.state_dict().items())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==21763
    with torch.no_grad():
        previous=old(pub);s=model(pub);frame=model.anchor(pub)[1]
        assert s.shape[-1]==6 and torch.equal(s[...,3:],previous)
        torch.testing.assert_close(s[...,:3],previous,rtol=0,atol=2e-7)
        initial_delta=float((s[...,:3]-previous).abs().max());check_geometry(s,g,frame)
    v=per_case(s,g,'topology');assert float(v['topology_regression'].max())<1e-10
    a=losses(s,g,None,'topology',0);b=losses(s,g,None,'topology',200,include_quality=False)
    assert all(torch.equal(a[k],b[k]) for k in a)
    torch.testing.assert_close(v['total'],v['geometry']+4*v['true_intersection']+5*v['box_expansion']+.1*v['topology_regression'])
    changed=s.detach().requires_grad_();grad=torch.autograd.grad(per_case(changed,g,'topology')['total'].sum(),changed)[0]
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad[...,3:])==0
    records=[{'train_index':index,**{k:float(v[k][i]) for k in ('total','true_intersection','mib','topology_regression')}} for i,index in enumerate(ids)]
    for i,row in enumerate(rows):
        solo,_=collate([row],'cuda')
        with torch.no_grad():value=per_case(model(inputs_only(solo)),solo,'topology')['total'][0]
        torch.testing.assert_close(value,v['total'][i],rtol=5e-5,atol=3e-6)
    initial={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    model.train();assert not model.source.training and model.corrector.training
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,g,'topology');captured.replay()
    with torch.no_grad():direct=losses(captured.prediction,captured.inputs,None,'topology',0)
    for k in direct:torch.testing.assert_close(captured.values[k],direct[k],rtol=1e-5,atol=1e-6)
    assert all(p.grad is None for p in model.source.parameters())
    assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay();raise AssertionError('Nonfinite accepted')
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
    del captured,optimizer,model,source,old,s,g,pub,previous,v,a,b,changed,grad,direct;clear()
    resumed=HERE/'runs/preflight/resumed';whole=HERE/'runs/preflight/uninterrupted'
    args=parser().parse_args(['--arm','topology','--output',str(resumed),'--max-hours','.1','--max-epochs','2'])
    run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
    one=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    assert any(not torch.equal(one['model'][k],initial[k]) for k in one['model'] if k.startswith('corrector.'))
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    second=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    full=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','graph_step_control'):same(second[key],full[key])
    for key in ('torch','cuda'):same(second['rng'][key],full['rng'][key])
    for key in full['model']:
        if key.startswith('source.'):assert torch.equal(full['model'][key],initial[key])
    assert full['progress']['step']==second['progress']['step']==2 and one['progress']['step']==1
    assert second['elapsed_seconds']>one['elapsed_seconds']
    check=GeometryModel(**model_config('topology')).cuda();check.load_state_dict(full['model']);check.train()
    graph,_=collate(batch,'cuda')
    with torch.no_grad():prediction=check(inputs_only(graph));frame=check.anchor(inputs_only(graph))[1];bound=check_geometry(prediction,graph,frame)
    assert not check.source.training and check.corrector.training
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',{'status':'passed','indices':ids,'unique_train_cases':5,'expanded_train_cases':2,
        'source_E78_weights_frozen_bit_exact':True,'source_forward_reconstructed_bit_exact':True,
        'initial_physical_state_max_abs_delta':initial_delta,'source_shape_dimensions_bit_exact_before_and_after_updates':True,
        'initial_guard_zero_and_healthy_separation_no_reward':True,'synthetic_order_regression_penalty_and_gradient':True,
        'output_channels':'predicted3 plus detached public source3, not labels','auxiliary_source_channels_gradient_zero':True,
        'fixed_loss_epoch_invariant':True,'cuda_matches_direct_same_output':True,'five_initial_metrics':records,
        'area_fixed_preplaced_and_frame_after_real_updates':True,'max_normalized_center_shift_after_updates':bound,
        'source_eval_during_training':True,'raw_cached_inputs_bit_exact':True,'test_roles_rejected_before_payload':True,
        'batch_padding_consistent':True,'cuda_warmup_no_update':True,'nonfinite_rejected_before_update':True,
        'real_dropout_Adam_resume_bit_exact':True,'sources':sources(),'test_read':False,'formal_training_started':False,
        'seconds':time.perf_counter()-begin})
    print('E79 preflight passed',flush=True)
if __name__=='__main__':main()
