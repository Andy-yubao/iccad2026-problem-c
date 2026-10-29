"""Five cases: warm widening, both-source fixed Loss, specs, CUDA and exact resume."""
import gc,time
import torch
from .io import HERE,ROOT,save,sources
from .initialization import load,verify_protected,provenance
from .model import GeometryModel,DeploymentModel,deployment_weights,decode
from .old_model import GeometryModel as OldModel,extract
from .arms import model_config
from .batches import inputs_only
from .data import load_rows,collate
from .loss import per_case,source_terms,losses
from .anchored_paths import corridor
from .frame import predicted_frame
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
def geometry(state,g,base):
    p=decode(state,g);lo,hi=predicted_frame(base,g);valid=g['valid']
    assert bool((p[...,:2]>=lo[:,None]-3e-6).masked_fill(~valid[...,None],True).all())
    assert bool((p[...,:2]+p[...,2:]<=hi[:,None]+3e-6).masked_fill(~valid[...,None],True).all())
    torch.testing.assert_close(p[...,2:].prod(-1)[valid],g['area'][valid],rtol=4e-6,atol=3e-8)
    placed=(~g['mask'][...,0].bool())&valid;locked=(~g['mask'][...,2].bool())&valid
    assert torch.equal(p[placed],g['specs'][placed]);assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
def synthetic():
    g={'area':torch.ones(1,3,device='cuda'),'valid':torch.ones(1,3,device='cuda',dtype=torch.bool),
       'mask':torch.tensor([[[1.,1.,1.],[1.,1.,1.],[0.,0.,0.]]],device='cuda'),
       'specs':torch.tensor([[[0.,0.,0.,0.],[0.,0.,0.,0.],[1.5,0.,1.,1.]]],device='cuda')}
    s=torch.tensor([[[.5,.5,0.],[1.5,.5,0.],[2.,.5,0.]]],device='cuda',requires_grad=True)
    a,deficit=corridor(s,g);assert float(a)>0
    grad=torch.autograd.grad(a.sum(),s)[0]
    assert torch.isfinite(grad).all() and bool((grad[...,:2]==0).all()) and bool((grad[...,2]!=0).any())
    return {'penalty':float(a),'ordinary_shape_gradient':True,'ordinary_center_gradient_zero':True}

def main():
    begin=time.perf_counter();seed_all(0);protected=verify_protected();syn=synthetic();ids=[1683,10027,5474,18140,25001]
    # Validate all qualified pipeline module names before any formal background run.
    import importlib.util,json,subprocess,sys
    commands=json.loads(subprocess.check_output([sys.executable,'-B','-m','experiments.E81_multisource_refinement.pipeline','--dry-run'],cwd=ROOT,text=True))
    for _,command in commands:
        module=command[command.index('-m')+1]
        assert module.startswith('experiments.E81_multisource_refinement.') and importlib.util.find_spec(module) is not None
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for raw,packed in zip(rows,cached):
        assert raw['member']==packed['member']
        for k,v in raw['graph'].items():
            if isinstance(v,torch.Tensor):same(v,packed['graph'][k])
            else:assert v==packed['graph'][k]
    from .dataset_view import ReferenceLayouts
    for role in ('test','local_test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda');pub=inputs_only(g);config=model_config('topology')
    model=GeometryModel(**config).cuda().eval();source=load(model,config)
    p=provenance()['parents'];placer=torch.load(ROOT/p['placer']['checkpoint'],map_location='cpu',weights_only=False)
    old_small=extract(OldModel(**{**source['model_config'],'width':96}),source['model'],placer['model']).cuda().eval()
    old_large=extract(OldModel(**source['model_config']),source['model']).cuda().eval()
    assert sum(p.numel() for p in model.parameters())==7543025
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==241547
    with torch.no_grad():
        state=model(pub);first=old_small(pub)[...,:3];second=old_large(pub)[...,:3]
        torch.testing.assert_close(state[...,:3],first,rtol=1e-5,atol=2e-5)
        torch.testing.assert_close(state[...,3:],second,rtol=1e-5,atol=2e-5)
        geometry(state[...,:3],g,model.placer(pub));geometry(state[...,3:],g,model.alternative(pub))
        initial_error=[float((state[...,:3]-first).abs().max()),float((state[...,3:]-second).abs().max())]
    try:model(g);raise AssertionError('Labels accepted')
    except ValueError:pass
    a=losses(state,g,None,'topology',0);b=losses(state,g,None,'topology',200)
    assert all(torch.equal(a[k],b[k]) for k in a)
    v=per_case(state,g,'topology');mean=(v['P42_total']+v['P74_total'])/2
    torch.testing.assert_close(v['total'],mean+.25*(torch.maximum(v['P42_total'],v['P74_total'])-mean))
    duplicated=torch.cat((state[...,:3],state[...,:3]),-1)
    torch.testing.assert_close(per_case(duplicated,g,'topology')['total'],source_terms(state[...,:3],g)['total'])
    records=[{'train_index':index,**{k:float(v[k][i]) for k in ('total','P42_total','P74_total','true_intersection','anchored_corridor','P74_anchored_corridor')}} for i,index in enumerate(ids)]
    for i,row in enumerate(rows):
        solo,_=collate([row],'cuda')
        with torch.no_grad():value=per_case(model(inputs_only(solo)),solo,'topology')['total'][0]
        torch.testing.assert_close(value,v['total'][i],rtol=1e-4,atol=1e-5)
    initial={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    deploy=DeploymentModel(**config).cuda().eval();deploy.load_state_dict(deployment_weights(initial),strict=True)
    with torch.no_grad():torch.testing.assert_close(deploy(pub),model(pub)[...,:3],rtol=0,atol=0)
    assert sum(p.numel() for p in deploy.parameters())==1713086
    model.train();assert not model.placer.training and not model.alternative.training and model.refiner.training
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,g,'topology');captured.replay()
    direct=losses(captured.prediction,captured.inputs,None,'topology',0)
    for k in direct:torch.testing.assert_close(captured.values[k],direct[k],rtol=1e-5,atol=1e-6)
    assert all(p.grad is None for p in model.placer.parameters()) and all(p.grad is None for p in model.alternative.parameters())
    assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay();raise AssertionError('Nonfinite accepted')
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    assert all(torch.equal(x.cpu(),initial[k]) for k,x in model.state_dict().items())
    del captured,optimizer,model,deploy,source,old_small,old_large,placer,state,first,second,g,pub,v,a,b,mean,duplicated,direct;clear()
    resumed=HERE/'runs/preflight/resumed_v2';whole=HERE/'runs/preflight/uninterrupted_v2'
    args=parser().parse_args(['--arm','topology','--output',str(resumed),'--max-hours','.15','--max-epochs','2'])
    run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
    one=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    for stage in ('first','second','third'):
        assert any(not torch.equal(one['model'][k],initial[k]) for k in one['model'] if k.startswith('refiner.'+stage+'.'))
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    second=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    full=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','graph_step_control'):same(second[key],full[key])
    for key in ('torch','cuda'):same(second['rng'][key],full['rng'][key])
    for key in full['model']:
        if key.startswith(('placer.','alternative.')):assert torch.equal(full['model'][key],initial[key])
    assert full['progress']['step']==second['progress']['step']==2
    assert second['elapsed_seconds']>one['elapsed_seconds']
    check=GeometryModel(**config).cuda();check.load_state_dict(full['model']);check.train();graph,_=collate(batch,'cuda');public=inputs_only(graph)
    with torch.no_grad():
        check.eval();prediction=check(public);geometry(prediction[...,:3],graph,check.placer(public));geometry(prediction[...,3:],graph,check.alternative(public))
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',{'status':'passed','indices':ids,'unique_train_cases':5,'expanded_train_cases':2,
        'warm_widened_initial_max_state_error_P42_P74':initial_error,'synthetic_corridor':syn,
        'initial_F_all_stages_finite_and_trainable':True,'frozen_placers_bit_exact_after_updates':True,
        'public_inputs_only':True,'both_sources_every_case_fixed_Loss_epoch_invariant':True,
        'deployment_matches_first_training_source_exact':True,'deployment_parameters':1713086,'trainable_F_parameters':241547,
        'five_initial_metrics':records,'raw_cache_parity_and_test_role_rejection':True,'batch_padding_consistent':True,
        'specs_area_frames_after_real_updates':True,'CUDA_Loss_consistency':True,'nonfinite_before_update':True,
        'real_Dropout_Adam_RNG_cursor_resume_bit_exact':True,'all_pipeline_entrypoints_resolve':True,'sources':sources(),'test_read':False,'formal_training_started':False,
        'seconds':time.perf_counter()-begin})
    print('E81 five-case warm-wide multisource preflight PASSED',flush=True)

if __name__=='__main__':main()
