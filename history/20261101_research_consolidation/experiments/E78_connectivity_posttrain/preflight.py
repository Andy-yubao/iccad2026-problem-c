"""Five train cases: public coherent shapes, corridor, CUDA and exact resume."""
import gc,time
import torch
from .io import HERE,save,sources
from .initialization import load,verify_protected
from .model import GeometryModel,decode
from .pretrained_model import GeometryModel as OldModel
from .mib_projection import coherent_shapes
from .anchored_paths import corridor
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

def check_geometry(s,g,frame):
    p=decode(s,g);valid=g['valid'];lo,hi=frame
    assert bool((p[...,:2]>=lo[:,None]-2e-6).masked_fill(~valid[...,None],True).all())
    assert bool((p[...,:2]+p[...,2:]<=hi[:,None]+2e-6).masked_fill(~valid[...,None],True).all())
    torch.testing.assert_close(p[...,2:].prod(-1)[valid],g['area'][valid],rtol=3e-6,atol=2e-8)
    placed=(~g['mask'][...,0].bool())&valid;locked=(~g['mask'][...,2].bool())&valid
    assert torch.equal(p[placed],g['specs'][placed]);assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
    return p

def synthetic():
    device='cuda';valid=torch.ones(1,3,device=device,dtype=torch.bool)
    mask=torch.tensor([[[0.,0.,0.],[1.,1.,1.],[0.,0.,0.]]],device=device)
    specs=torch.tensor([[[0.,0.,1.,1.],[0.,0.,0.,0.],[1.5,0.,1.,1.]]],device=device)
    state=torch.tensor([[[.5,.5,0.],[1.25,.5,0.],[2.,.5,0.]]],device=device,requires_grad=True)
    g={'area':torch.ones(1,3,device=device),'valid':valid,'mask':mask,'specs':specs}
    value,deficit=corridor(state,g);torch.testing.assert_close(value,torch.tensor([.125],device=device),rtol=1e-5,atol=1e-6)
    gradient=torch.autograd.grad(value.sum(),state)[0];assert torch.isfinite(gradient).all() and gradient[0,1,2]>0
    g['specs']=specs.clone();g['specs'][0,2,0]=2
    feasible,_=corridor(state,g);assert float(feasible)==0
    g['mask']=torch.ones_like(mask);empty,_=corridor(state,g);assert float(empty)==0
    g={'area':torch.ones(1,2,device=device),'valid':torch.ones(1,2,device=device,dtype=torch.bool),
       'mask':torch.ones(1,2,3,device=device),'specs':torch.zeros(1,2,4,device=device),
       'known':torch.zeros(1,2,3,device=device),'mib':torch.tensor([[[0.,1.],[1.,0.]]],device=device),
       'constraints':torch.tensor([[[0,0,1,0,0],[0,0,1,0,0]]],device=device)}
    s=torch.tensor([[[.6,.7,.2],[1.8,.7,-.2]]],device=device,requires_grad=True)
    frame=(torch.zeros(1,2,device=device),torch.tensor([[3.,2.]],device=device))
    tied=coherent_shapes(s,g,frame);torch.testing.assert_close(tied[...,2],torch.zeros(1,2,device=device))
    check_geometry(tied,g,frame)
    assert torch.isfinite(torch.autograd.grad(tied.square().sum(),s)[0]).all()
    g['mask'][0,0,2]=0;g['specs'][0,0,2:]=1
    tied=coherent_shapes(s,g,frame);check_geometry(tied,g,frame);assert float(tied[0,1,2])==0
    g['mask'][0,0,2]=1;g['area'][0,1]=1.5
    untied=coherent_shapes(s,g,frame);torch.testing.assert_close(untied[...,2],s[...,2])

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
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda');public=inputs_only(g)
    model=GeometryModel(**model_config('topology')).cuda().eval();source=load(model,model_config('topology'))
    old=OldModel(**source['model_config']).cuda().eval();old.load_state_dict(source['model'])
    assert all(torch.equal(x.cpu(),source['model'][k]) for k,x in model.state_dict().items())
    assert torch.count_nonzero(model.corrector.output[-1].weight)>0
    with torch.no_grad():
        previous=old(public);expected_frame=old.anchor(public)[1];s=model(public)
        check_geometry(s,g,expected_frame)
        expected=coherent_shapes(previous,g,expected_frame)
        torch.testing.assert_close(decode(s,g),decode(expected,g),rtol=0,atol=2e-7)
    v=per_case(s,g,'topology');a=losses(s,g,None,'topology',0);b=losses(s,g,None,'topology',200,include_quality=False)
    assert all(torch.equal(a[k],b[k]) for k in a)
    torch.testing.assert_close(v['total'],v['geometry']+2*v['true_intersection']+5*v['box_expansion']+.25*v['anchored_corridor'])
    changed=s.detach().requires_grad_();grad=torch.autograd.grad(per_case(changed,g,'topology')['total'].sum(),changed)[0]
    assert torch.isfinite(grad).all()
    records=[{'train_index':index,**{k:float(v[k][i]) for k in ('total','true_intersection','mib','anchored_corridor')}} for i,index in enumerate(ids)]
    for i,row in enumerate(rows):
        solo,_=collate([row],'cuda')
        with torch.no_grad():value=per_case(model(inputs_only(solo)),solo,'topology')['total'][0]
        torch.testing.assert_close(value,v['total'][i],rtol=5e-5,atol=3e-6)
    initial={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    model.train();assert not model.source.training and model.corrector.training
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,g,'topology');captured.replay()
    with torch.no_grad():
        direct=losses(captured.prediction,captured.inputs,None,'topology',0)
    for k in direct:torch.testing.assert_close(captured.values[k],direct[k],rtol=1e-5,atol=1e-6)
    assert all(p.grad is None for p in model.source.parameters())
    assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay();raise AssertionError('Nonfinite accepted')
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
    del captured,optimizer,model,source,old,s,g,public,previous,expected,v,a,b,changed,grad,direct;clear()
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
    graph,_=collate(batch,'cuda');pub=inputs_only(graph)
    with torch.no_grad():
        prediction=check(pub);frame=check.anchor(pub)[1];check_geometry(prediction,graph,frame)
        vv=per_case(prediction,graph,'topology');assert torch.isfinite(vv['total']).all()
    assert not check.source.training and check.corrector.training
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',{'status':'passed','indices':ids,'unique_train_cases':5,'expanded_train_cases':2,
        'full_E77_initial_weights_bit_exact':True,'initial_function':'Public MIB-coherent projection, not unprojected E77',
        'fixed_loss_epoch_invariant':True,'new_loss_cuda_capture_matches_direct_same_output':True,
        'train_dropout_differs_from_eval_expected':True,'five_initial_metrics':records,
        'synthetic_anchor_overfill_value_and_gradient':True,'synthetic_feasible_and_no_anchor_zero':True,
        'synthetic_MIB_tie_public_locked_and_unequal_area_skip':True,
        'area_fixed_preplaced_and_frame_after_real_updates':True,'whole_source_frozen_bit_exact':True,
        'source_eval_training':True,'raw_cached_inputs_bit_exact':True,'test_roles_rejected_before_payload':True,
        'batch_padding_consistent':True,'cuda_warmup_no_update':True,'nonfinite_rejected_before_update':True,
        'real_trained_head_dropout_Adam_resume_bit_exact':True,'sources':sources(),'test_read':False,
        'formal_training_started':False,'seconds':time.perf_counter()-begin})
    print('E78 preflight passed',flush=True)

if __name__=='__main__':main()
