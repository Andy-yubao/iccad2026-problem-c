"""Same five train cases, fixed Loss and real Adam/CUDA exact-resume checks."""
import gc,time
import torch
from .io import HERE,ROOT,save,sha,sources
from .initialization import load,verify_protected,configure_trainability
from .arms import model_config
from .model import GeometryModel
from .base_model import GeometryModel as SourceModel
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
    from .box import expansion
    box_cost=expansion(separated,synthetic)
    torch.testing.assert_close(box_cost,torch.tensor([1.5],device='cuda').log().square())
    assert torch.isfinite(torch.autograd.grad(box_cost.sum(),separated,retain_graph=True)[0]).all()
    inside=torch.tensor([[[.5,.5,0.],[1.4,.5,0.]]],device='cuda',requires_grad=True)
    inside_cost=expansion(inside,synthetic);assert float(inside_cost)==0
    assert torch.equal(torch.autograd.grad(inside_cost.sum(),inside)[0],torch.zeros_like(inside))
    from .intersection import fraction
    for value in (inside,separated):
        predicted=fraction(value,synthetic)
        assert float(predicted)>=0
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
    original_config=dict(source['model_config'])
    original=SourceModel(**original_config).cuda().eval();original.load_state_dict(source['model'])
    with torch.no_grad():
        s=model(inputs_only(g));old=original(inputs_only(g))
    # FP32 reduction order changes with doubled Linear/attention dimensions.
    # Absolute tolerance is in normalized center/log-aspect coordinates.
    torch.testing.assert_close(s,old,rtol=5e-5,atol=2e-5)
    widening_max_abs_delta=float((s-old).abs().max())
    initial_widened={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    s=s.detach().requires_grad_();v=per_case(s,g,'topology');a=losses(s,g,None,'topology',0);b=losses(s,g,None,'topology',200,include_quality=False)
    assert all(torch.equal(a[k],b[k]) for k in a)
    torch.testing.assert_close(v['total'],v['geometry']+2*v['true_intersection']+5*v['box_expansion'])
    new_gradient=torch.autograd.grad(v['total'].sum(),s,retain_graph=True)[0]
    old_gradient=torch.autograd.grad((v['geometry']+.5*v['collision_topology']).sum(),s,retain_graph=True)[0]
    assert torch.isfinite(new_gradient).all()
    from .batches import decode
    direction_checks=[]
    for i,index in enumerate(ids):
        record={'train_index':index,'new_loss':float(v['total'][i].detach()),'box_expansion':float(v['box_expansion'][i].detach()),'collision':float(v['collision_topology'][i].detach())}
        for label,gradient in (('current',None),('old_direction',old_gradient),('new_direction',new_gradient)):
            delta=torch.zeros_like(s) if gradient is None else -.005*gradient/gradient.flatten(1).norm(dim=1)[:,None,None].clamp_min(1e-12)
            positions=decode(s.detach()+delta.detach(),g)[i,g['valid'][i]]
            lo=positions[:,:2];hi=lo+positions[:,2:];area=g['area'][i].sum()
            inter=(torch.minimum(hi[:,None],hi[None,:])-torch.maximum(lo[:,None],lo[None,:])).clamp_min(0)
            overlap=(inter[...,0]*inter[...,1]);overlap.fill_diagonal_(0)
            record[label]={'bbox_over_A':float((hi.amax(0)-lo.amin(0)).prod()/area),'intersection_over_A':float(overlap.sum()/2/area)}
        direction_checks.append(record)
    # Per-case aggregation is invariant to batching/padding.
    for i,row in enumerate(rows):
        one,_=collate([row],'cuda')
        with torch.no_grad():output=model(inputs_only(one));solo=per_case(output,one,'topology')
        torch.testing.assert_close(solo['total'][0],v['total'][i],rtol=3e-5,atol=2e-6)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,g,'topology');captured.replay()
    assert all(p.grad is None for p in model.backbone.parameters())
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
    assert any(not torch.equal(one['model'][k],initial_widened[k]) for k in one['model'] if k.startswith('corrector.'))
    for key in one['model']:
        if key.startswith('backbone.'):assert torch.equal(one['model'][key],initial_widened[key])
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    second=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    full=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','graph_step_control'):same(second[key],full[key])
    for key in ('torch','cuda'):same(second['rng'][key],full['rng'][key])
    assert second['progress']['step']==full['progress']['step']==2 and one['progress']['step']==1
    assert second['elapsed_seconds']>one['elapsed_seconds']
    for key in full['model']:
        if key.startswith('backbone.'):assert torch.equal(full['model'][key],one['model'][key])
    check=GeometryModel(**model_config('topology')).cuda();check.load_state_dict(full['model'])
    check.train();assert not check.backbone.training and check.corrector.training
    graph,_=collate(batch,'cuda')
    with torch.no_grad():
        teacher=check.backbone(inputs_only(graph));prediction=check(inputs_only(graph))
    local=graph['area'].clamp_min(1e-12).sqrt();scale=torch.stack((local,local,torch.ones_like(local)),-1)
    normalized=(prediction-teacher).abs()/scale
    bound=float(normalized.max());assert bound<=.150001
    inactive=(graph['mask']==0)|(~graph['valid'][...,None])
    assert torch.equal((prediction-teacher)[inactive],torch.zeros_like((prediction-teacher)[inactive]))
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',{'status':'passed','unique_train_cases':5,'indices':ids,'fixed_loss':True,'raw_cached_inputs_bit_exact':True,'expanded_train_cases':2,'test_roles_rejected_before_payload':True,
        'batch_padding_consistent':True,'cuda_no_warmup_update':True,'nonfinite_rejected_before_update':True,
        'widening_eval_max_abs_delta':widening_max_abs_delta,'source_width':192,'new_width':192,'whole_encoder_fixed':True,'parameter_step_projection':False,'max_output_correction_fraction':.15,
        'verified_two_step_max_output_correction_fraction':bound,'frozen_encoder_initial_and_after_two_steps_identical':True,'encoder_eval_during_training':True,
        'synthetic_noncollision_penalty_and_gradient_zero':True,'continuous_contact_limit_values':touching,'quadratic_box_one_sided_value_and_gradient_checked':True,
        'five_output_direction_checks':direction_checks,'direction_check_limit':'Output-space diagnostic only; not an Adam parameter update or quality guarantee',
        'real_dropout_adam_resume_bit_exact':True,'sources':sources(),'test_read':False,'formal_training_started':False,'seconds':time.perf_counter()-begin})
    print('E76 preflight passed',flush=True)
if __name__=='__main__':main()
