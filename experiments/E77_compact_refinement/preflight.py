"""Five real cases: physical projection, frozen source, CUDA and exact resume."""
import gc,time
import numpy as np
import torch
from .io import HERE,save,sources
from .initialization import load,verify_protected,configure_trainability
from .arms import model_config
from .model import GeometryModel,decode
from .source_model import GeometryModel as SourceModel
from .frame import contain,predicted_frame
from .batches import inputs_only
from .data import load_rows,collate
from .loss import losses,per_case
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


def geometry(p):
    p=p.detach().cpu().double().numpy();lo=p[:,:2];hi=lo+p[:,2:];area=np.prod(p[:,2:],1).sum()
    inter=np.maximum(0,np.minimum(hi[:,None],hi[None,:])-np.maximum(lo[:,None],lo[None,:])).prod(-1)
    return {'box_over_A':float(np.prod(hi.max(0)-lo.min(0))/area),'intersection_over_A':float(np.triu(inter,1).sum()/area)}


def independent_projection(rectangles,areas,mask,specs,frame):
    # CPU float64 physical-coordinate oracle, including dimension changes.
    p=rectangles.detach().cpu().double().numpy().copy();a=areas.cpu().double().numpy()
    m=mask.cpu().numpy();s=specs.cpu().double().numpy();lo,hi=[v.cpu().double().numpy() for v in frame]
    width,height=hi-lo
    for i in range(len(p)):
        if not m[i,0]:p[i]=s[i];continue
        center=p[i,:2]+p[i,2:]/2
        if m[i,2]:
            ratio=np.log(p[i,2]/p[i,3]);ratio=np.clip(ratio,max(-5,np.log(a[i]/height**2)),min(5,np.log(width**2/a[i])))
            p[i,2:]=[np.sqrt(a[i]*np.exp(ratio)),np.sqrt(a[i]/np.exp(ratio))]
        center=np.clip(center,lo+p[i,2:]/2,hi-p[i,2:]/2)
        p[i,:2]=center-p[i,2:]/2
    return p


def verify_geometry(state,g,frame):
    p=decode(state,g);lo,hi=frame;valid=g['valid'];v=valid[...,None]
    assert bool((p[...,:2]>=lo[:,None]-2e-6).masked_fill(~v,True).all())
    assert bool((p[...,:2]+p[...,2:]<=hi[:,None]+2e-6).masked_fill(~v,True).all())
    areas=p[...,2:].prod(-1)
    torch.testing.assert_close(areas[valid],g['area'][valid],rtol=3e-6,atol=2e-8)
    placed=(~g['mask'][...,0].bool())&valid;locked=(~g['mask'][...,2].bool())&valid
    assert torch.equal(p[placed],g['specs'][placed])
    assert torch.equal(p[...,2:][locked],g['specs'][...,2:][locked])
    return p


def main():
    begin=time.perf_counter();seed_all(0);protected=verify_protected();ids=[1683,10027,5474,18140,25001]
    rows=load_rows('train',ids,use_cache=False);cached=load_rows('train',ids)
    for raw,prepared in zip(rows,cached):
        assert raw['member']==prepared['member']
        for k,v in raw['graph'].items():
            if isinstance(v,torch.Tensor):same(v,prepared['graph'][k])
            else:assert v==prepared['graph'][k]
    from .dataset_view import ReferenceLayouts
    for role in ('local_test','test','tail_test'):
        try:ReferenceLayouts(role);raise AssertionError('Sealed role accepted')
        except PermissionError:pass
    batch=[rows[i%5] for i in range(32)];g,_=collate(batch,'cuda');public=inputs_only(g)
    model=GeometryModel(**model_config('topology')).cuda().eval();source=load(model,model_config('topology'))
    teacher=SourceModel(**source['model_config']).cuda().eval();teacher.load_state_dict(source['model'])
    with torch.no_grad():
        old=teacher(public);base=teacher.backbone(public);frame=predicted_frame(base,g)
        result=model(public);initial=verify_geometry(result,g,frame)
        reconstructed=model.source.backbone(public)
        local=g['area'].clamp_min(1e-12).sqrt();bounds=.15*torch.stack((local,local,torch.ones_like(local)),-1)
        reconstructed=reconstructed+model.source.corrector(reconstructed,public).tanh()*bounds*g['mask']*g['valid'][...,None]
        assert torch.equal(reconstructed,old)
        projected=contain(old,g,frame);torch.testing.assert_close(decode(projected,g),initial,rtol=0,atol=2e-7)
    records=[]
    for i,index in enumerate(ids):
        valid=g['valid'][i];p=decode(old,g)[i,valid]
        oracle=independent_projection(p,g['area'][i,valid],g['mask'][i,valid],g['specs'][i,valid],tuple(v[i] for v in frame))
        np.testing.assert_allclose(initial[i,valid].cpu().numpy(),oracle,atol=2e-6,rtol=3e-6)
        records.append({'train_index':index,'E74':geometry(decode(base,g)[i,valid]),'E76':geometry(p),'projected_E76':geometry(initial[i,valid])})
    # Exaggerated free-shape/center proposals must still fit without altering
    # fixed/preplaced public specifications. Includes both aspect extremes.
    for factor in (-1.,1.):
        stress=old.clone();stress[...,:2]+=factor*10;stress[...,2]=factor*100
        stress.requires_grad_();bounded=contain(stress,g,frame);verify_geometry(bounded,g,frame)
        assert bool(torch.isfinite(torch.autograd.grad(bounded.square().sum(),stress)[0]).all())
    s=result.detach().requires_grad_();v=per_case(s,g,'topology')
    a=losses(s,g,None,'topology',0);b=losses(s,g,None,'topology',200,include_quality=False)
    assert all(torch.equal(a[k],b[k]) for k in a)
    torch.testing.assert_close(v['total'],v['geometry']+2*v['true_intersection']+5*v['box_expansion'])
    assert torch.isfinite(torch.autograd.grad(v['total'].sum(),s)[0]).all()
    for i,row in enumerate(rows):
        solo,_=collate([row],'cuda')
        with torch.no_grad():value=per_case(model(inputs_only(solo)),solo,'topology')['total'][0]
        torch.testing.assert_close(value,v['total'][i],rtol=3e-5,atol=2e-6)
    frozen={k:x.detach().cpu().clone() for k,x in model.state_dict().items()}
    model.train();assert not model.source.training and not model.source.corrector.training and model.corrector.training
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0,weight_decay=3e-4)
    captured=CapturedStep(model,optimizer,g,'topology');captured.replay()
    assert all(p.grad is None for p in model.source.parameters())
    assert all(torch.equal(x.cpu(),frozen[k]) for k,x in model.state_dict().items())
    torch.testing.assert_close(captured.values['total'],a['total'],rtol=1e-5,atol=1e-6)
    captured.inputs['target'].fill_(float('nan'))
    try:captured.replay();raise AssertionError('Nonfinite accepted')
    except RuntimeError as error:assert 'Nonfinite' in str(error)
    assert all(torch.equal(x.cpu(),frozen[k]) for k,x in model.state_dict().items())
    del captured,optimizer,model,teacher,source,old,base,result,initial,g,public,s,v,a,b;clear()
    resumed=HERE/'runs/preflight/resumed';whole=HERE/'runs/preflight/uninterrupted'
    args=parser().parse_args(['--arm','topology','--output',str(resumed),'--max-hours','.1','--max-epochs','2'])
    run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
    one=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    assert any(not torch.equal(one['model'][k],frozen[k]) for k in one['model'] if k.startswith('corrector.'))
    args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    second=torch.load(resumed/'checkpoints/latest.pt',map_location='cpu',weights_only=False);clear()
    args.resume=False;args.output=whole;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
    full=torch.load(whole/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','scheduler','graph_step_control'):same(second[key],full[key])
    for key in ('torch','cuda'):same(second['rng'][key],full['rng'][key])
    assert full['progress']['step']==second['progress']['step']==2 and one['progress']['step']==1
    assert second['elapsed_seconds']>one['elapsed_seconds']
    for key in full['model']:
        if key.startswith('source.'):assert torch.equal(full['model'][key],frozen[key])
    check=GeometryModel(**model_config('topology')).cuda();check.load_state_dict(full['model']);check.train()
    graph,_=collate(batch,'cuda');pub=inputs_only(graph)
    with torch.no_grad():
        state=check(pub);anchor,bounds_frame=check.anchor(pub);verify_geometry(state,graph,bounds_frame)
        raw=check.corrector(anchor,pub);local=graph['area'].clamp_min(1e-12).sqrt()
        scale=torch.stack((local,local,torch.ones_like(local)),-1)
        residual=raw.tanh()*.15*scale*graph['mask']*graph['valid'][...,None]
        bound=float((residual.abs()/scale).max());assert bound<=.150001
    assert not check.source.training and not check.source.corrector.training
    assert verify_protected()==protected
    save(HERE/'runs/preflight/results.json',{'status':'passed','unique_train_cases':5,'indices':ids,'expanded_train_cases':2,
        'source_E76_function_reconstructed_bit_exact':True,'initial_projected_physical_output_independent_CPU_oracle':True,
        'projection_is_not_unprojected_source':'Canonical physical state, raw masked channels may differ',
        'five_frame_tradeoff_probes':records,'shape_extremes_and_centers_contained':True,'area_fixed_preplaced_checked':True,
        'whole_source_frozen_bit_exact_after_real_updates':True,'source_and_old_corrector_eval_during_training':True,
        'fixed_loss_epoch_invariant':True,'batch_padding_consistent':True,'raw_cached_inputs_bit_exact':True,
        'test_roles_rejected_before_payload':True,'cuda_no_warmup_update':True,'nonfinite_rejected_before_update':True,
        'real_dropout_adam_resume_bit_exact':True,'two_step_max_preprojection_residual_fraction':bound,
        'sources':sources(),'formal_training_started':False,'test_read':False,'seconds':time.perf_counter()-begin})
    print('E77 preflight passed',flush=True)


if __name__=='__main__':main()
