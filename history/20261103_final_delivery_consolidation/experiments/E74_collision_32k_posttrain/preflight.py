"""Same five train cases, fixed Loss and real Adam/CUDA exact-resume checks."""
import gc,time
import torch
from .io import HERE,ROOT,save,sha,sources
from .initialization import load,verify_protected,configure_trainability
from .arms import model_config
from .model import GeometryModel
from .batches import inputs_only
from .data import load_rows,collate
from .loss import losses,per_case
from .losses import WEIGHTS
from .topology import details
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

def main():
    begin=time.perf_counter();seed_all(0);protected=verify_protected();ids=[1683,10027,5474,18140,25001]
    synthetic={'area':torch.ones(1,2,device='cuda'),'valid':torch.ones(1,2,device='cuda',dtype=torch.bool),
        'mask':torch.ones(1,2,3,device='cuda'),'specs':torch.zeros(1,2,4,device='cuda'),
        'reference':torch.tensor([[[0.,0.,1.,1.],[1.,0.,1.,1.]]],device='cuda')}
    separated=torch.tensor([[[2.5,.5,0.],[.5,.5,0.]]],device='cuda',requires_grad=True)
    penalty,diagnostic=details(separated,synthetic)
    assert float(penalty)==0 and float(diagnostic['unconditional'])>0
    assert torch.equal(torch.autograd.grad(penalty.sum(),separated)[0],torch.zeros_like(separated))
    touching=[]
    for epsilon in (1e-3,1e-4,0.):
        point=torch.tensor([[[1.5-epsilon,.5,0.],[.5,.5,0.]]],device='cuda',requires_grad=True)
        cost,_=details(point,synthetic);assert torch.isfinite(torch.autograd.grad(cost.sum(),point)[0]).all();touching.append(float(cost))
    assert touching[0]>touching[1]>touching[2] and touching[2]==0
    rows=load_rows('train',ids,use_cache=False)
    cached=load_rows('train',ids)
    for raw,prepared in zip(rows,cached):
        assert raw['member']==prepared['member']
        for key,value in raw['graph'].items():
            if isinstance(value,torch.Tensor):torch.testing.assert_close(value,prepared['graph'][key],rtol=0,atol=0)
            else:assert value==prepared['graph'][key]
    from .dataset_view import ReferenceLayouts
    for role in ('local_test','test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda')
    model=GeometryModel(**model_config('topology')).cuda().eval();source=load(model,model_config('topology'));configure_trainability(model)
    original_config={**model_config('topology'),'width':96}
    original=GeometryModel(**original_config).cuda().eval();original.load_state_dict(source['model'])
    with torch.no_grad():
        s=model(inputs_only(g));old=original(inputs_only(g))
    # FP32 reduction order changes with doubled Linear/attention dimensions.
    # Absolute tolerance is in normalized center/log-aspect coordinates.
    torch.testing.assert_close(s,old,rtol=5e-5,atol=2e-5)
    widening_max_abs_delta=float((s-old).abs().max())
    initial_widened={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    s=s.detach().requires_grad_();v=per_case(s,g,'topology');a=losses(s,g,None,'topology',0);b=losses(s,g,None,'topology',200,include_quality=False)
    assert all(torch.equal(a[k],b[k]) for k in a)
    torch.testing.assert_close(v['total'],v['geometry']+.5*v['collision_topology'])
    assert torch.isfinite(torch.autograd.grad(v['total'].mean(),s)[0]).all()
    # Per-case aggregation is invariant to batching/padding.
    for i,row in enumerate(rows):
        one,_=collate([row],'cuda')
        with torch.no_grad():output=model(inputs_only(one));solo=per_case(output,one,'topology')
        torch.testing.assert_close(solo['total'][0],v['total'][i],rtol=3e-5,atol=2e-6)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,g,'topology');captured.replay()
    assert all(p.grad is None for layer in model.layers for p in layer.local.parameters())
    torch.testing.assert_close(captured.values['total'],a['total'],rtol=1e-5,atol=1e-6)
    assert all(torch.equal(x.cpu(),initial_widened[k]) for k,x in model.state_dict().items())
    # NaNs must be rejected before optimizer updates.
    before={k:x.detach().clone() for k,x in model.state_dict().items()}
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay();raise AssertionError('Nonfinite step accepted')
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    for k,x in model.state_dict().items():assert torch.equal(x,before[k])
    del captured,optimizer,model,original,old,s,v,a,b,g,source,before;clear()
    resumed=HERE/'runs/preflight/resumed';whole=HERE/'runs/preflight/uninterrupted'
    args=parser().parse_args(['--arm','topology','--output',str(resumed),'--max-hours','.1','--max-epochs','2'])
    run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
    one=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    relative_steps=[]
    for name in one['graph_step_control']['group_names']:
        keys=[k for k in one['model'] if k.rsplit('.',1)[0]==name]
        old_norm=sum(initial_widened[k].square().sum() for k in keys).sqrt()
        displacement=sum((one['model'][k]-initial_widened[k]).square().sum() for k in keys).sqrt()
        relative=float(displacement/old_norm.clamp_min(1e-8));relative_steps.append(relative)
        assert relative<=.005001, (name,relative)
    assert max(relative_steps)>0
    for key in one['model']:
        if '.local.' in key:assert torch.equal(one['model'][key],initial_widened[key])
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    second=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    full=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','graph_step_control'):same(second[key],full[key])
    for key in ('torch','cuda'):same(second['rng'][key],full['rng'][key])
    assert second['progress']['step']==full['progress']['step']==2 and one['progress']['step']==1
    assert second['elapsed_seconds']>one['elapsed_seconds']
    for key in full['model']:
        if '.local.' in key:assert torch.equal(full['model'][key],one['model'][key])
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',{'status':'passed','unique_train_cases':5,'indices':ids,'fixed_loss':True,'raw_cached_inputs_bit_exact':True,'expanded_train_cases':2,'test_roles_rejected_before_payload':True,
        'batch_padding_consistent':True,'cuda_no_warmup_update':True,'nonfinite_rejected_before_update':True,
        'widening_eval_max_abs_delta':widening_max_abs_delta,'source_width':96,'new_width':192,'graph_local_fixed':True,'max_relative_trainable_group_step':.005,
        'verified_first_update_max_relative_step':max(relative_steps),'frozen_graph_initial_and_after_two_steps_identical':True,
        'synthetic_noncollision_penalty_and_gradient_zero':True,'continuous_contact_limit_values':touching,
        'real_dropout_adam_resume_bit_exact':True,'sources':sources(),'test_read':False,'formal_training_started':False,'seconds':time.perf_counter()-begin})
    print('E74 preflight passed',flush=True)
if __name__=='__main__':main()
