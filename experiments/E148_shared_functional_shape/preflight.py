"""Independent geometry oracle, interface interventions, objective and GPU gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import copy
import hashlib
import io
import json
import time
import traceback
import numpy as np
import torch
from .io import ROOT,HERE,save,sources,sha
from . import train
from .data import load_rows,cache_identity
from .resident import ResidentRows
from .batches import inputs_only
from .model import GeometryModel
from .geometry_feedback import module_features,pair_features
from .shape_feedback import shape_features,DELTAS
from .loss import per_case,TOTAL_COMPONENTS
from .loss_config import ARMS,layer_weights
from .cuda_step import CapturedStep,EagerStep

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def assert_tree(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree(x,y)
    else:assert a==b

def synthetic():
    p=torch.tensor([[[5.,5.,1.,1.],[6.1,5.,.8,1.25],[5.,7.,1.,1.],[0.,0.,1e-6,1e-6]]])
    c=torch.zeros((1,4,5),dtype=torch.long);c[0,0,4]=5;c[0,1,0]=1;c[0,2,1]=1
    group=torch.zeros((1,4,4));group[0,0,1]=group[0,1,0]=1
    g=dict(area=torch.ones((1,4)),valid=torch.tensor([[True,True,True,False]]),constraints=c,group=group)
    measured=shape_features(p,g)[0].numpy()
    base=p[0,:3].double().numpy();centers=base[:,:2]+base[:,2:]/2
    def metric(rectangles,i):
        q=rectangles[i];ql=q[:2];qh=ql+q[2:];load=0.;contact=[]
        for j,r in enumerate(rectangles):
            if i==j:continue
            rl=r[:2];rh=rl+r[2:];signed=np.minimum(qh,rh)-np.maximum(ql,rl)
            load+=np.maximum(signed,0).prod()
            if group[0,i,j]>0:
                fx=min((qh[0]-rl[0])**2,(rh[0]-ql[0])**2)
                fy=min((qh[1]-rl[1])**2,(rh[1]-ql[1])**2)
                ty=max(.01*min(q[3],r[3])-signed[1],0)**2
                tx=max(.01*min(q[2],r[2])-signed[0],0)**2
                contact.append(min(fx+ty,fy+tx))
        lower=rectangles[:,:2].min(0);upper=(rectangles[:,:2]+rectangles[:,2:]).max(0)
        sides=np.array([ql[0]-lower[0],upper[0]-qh[0],upper[1]-qh[1],ql[1]-lower[1]])
        req=np.array([bool(int(c[0,i,4])&bit) for bit in (1,2,4,8)])
        boundary=sides[req].mean() if req.any() else 0.
        return np.array([np.log1p(load),min(contact) if contact else 0.,boundary,(upper-lower).prod()/3])
    expected=np.zeros((4,16))
    for i in range(3):
        old=metric(base,i)
        for k,d in enumerate(DELTAS):
            r=np.clip(np.log(base[i,2]/base[i,3])+d,-5,5)
            dims=np.array([np.exp(r/2),np.exp(-r/2)])
            trial=base.copy();trial[i]=np.r_[np.maximum(centers[i]-dims/2,0),dims]
            np.testing.assert_allclose(dims.prod(),1,rtol=1e-12)
            expected[i,4*k:4*k+4]=np.arcsinh(metric(trial,i)-old)
    np.testing.assert_allclose(measured,expected,rtol=3e-5,atol=2e-6)
    assert np.count_nonzero(measured[3])==0 and np.count_nonzero(measured[:3])>0
    scaled=dict(g,area=g['area']*9)
    torch.testing.assert_close(shape_features(p*3,scaled),torch.from_numpy(measured)[None],rtol=3e-5,atol=2e-6)
    moved=p.clone();moved[...,:2]+=10
    torch.testing.assert_close(shape_features(moved,g),torch.from_numpy(measured)[None],rtol=3e-5,atol=2e-6)
    lone=dict(g,valid=torch.tensor([[True,False,False,False]]),group=torch.zeros_like(group))
    assert bool(torch.isfinite(shape_features(p,lone)).all())
    boundary=p.clone();boundary[...,:2]=0
    assert bool(torch.isfinite(shape_features(boundary,g)).all())
    # The old capped escape would map both cases to1; original E142 remains unbounded.
    from experiments.E142_dual_geometry.geometry_feedback import module_features as old_features
    rich=dict(g,adjacency=torch.zeros_like(group))
    torch.testing.assert_close(module_features(p,rich),old_features(p,rich),rtol=0,atol=0)
    assert (HERE/'geometry_feedback.py').read_bytes()==(ROOT/'experiments/E142_dual_geometry/geometry_feedback.py').read_bytes()
    return dict(independent_numpy_shape_oracle=True,padding_singleton_origin_finite=True,
        original_49_bitwise=True,scale_translation_invariance=True,F_P_not_overwritten=True,blank_statistics=False)

def cache_gate(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert key==plan['required_cache_key'];base=ROOT/'datasets/static_tensor_cache_v1'/key
    assert read(base/'identity.json')==identity
    counts={}
    for role in ('train','dev_monitor500'):
        ds=RandomReference(role);blocks={}
        for i in ds.indices:blocks.setdefault(i//256,[]).append(i)
        for block,ids in blocks.items():
            path=base/ds.role/f'{block:04d}.pt';meta=read(path.with_suffix('.json'))
            assert path.exists() and meta['identity']==key
            assert all(meta['members'][str(i)]==ds.records[i]['member'] for i in ids)
        counts[role]=len(ds.indices)
    return dict(key=key,coverage=counts,metadata_only=True)

def objective_check(states,batch):
    from experiments.E139_delayed_overlap.loss import base_per_case as parent_base
    x=states.detach().clone().requires_grad_(True);weights=layer_weights(ARMS[0])
    a=per_case(x,batch,ARMS[0]);expected=0.
    for k,w in weights.items():
        raw=parent_base(x[k-1],batch)
        value=sum(raw[name] for name in TOTAL_COMPONENTS if k!=4 or name!='overlap_weighted')
        torch.testing.assert_close(a[f'layer{k:02d}_total'],value,rtol=0,atol=0)
        expected=expected+w*value
    torch.testing.assert_close(a['total'],expected,rtol=0,atol=0)
    ga=torch.autograd.grad(a['total'].sum(),x,retain_graph=True)[0]
    gb=torch.autograd.grad(expected.sum(),x)[0]
    torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    for k in range(1,13):assert bool(ga[k-1].abs().sum()>0)==(k in weights)
    assert torch.count_nonzero(a['layer04_overlap_weighted'])==0
    return dict(base_terms_and_coefficients_exact=True,loss_and_gradient_exact=True,
        layers=[4,8,12],weights=[.1,.1,.8],no_overlap=[4])

def interface_check(model,g,states,batch):
    counts=[0,0,0];sizes=[];hooks=[]
    for i,unit in enumerate(model.units):
        def count(module,args,value,index=i):counts[index]+=1
        hooks.append(unit.register_forward_hook(count))
        for k,encoder in enumerate(unit.demand_encoder):
            hooks.append(encoder.register_forward_pre_hook(lambda module,args,k=k:sizes.append((k,args[0].shape[-1]))))
    with torch.no_grad():again=model(g)
    for h in hooks:h.remove()
    torch.testing.assert_close(again,states,rtol=0,atol=0)
    assert counts==[2,2,2] and all(size==(81 if k==0 else 65) for k,size in sizes)
    with torch.no_grad():
        condition=model.problem_encoder(g)
        for unit in model.units:
            a=unit(g,states[0],condition);b=unit(g,states[0],condition)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Perturb final hidden in every call but force that emitted layout unchanged.
    # All subsequent unit calls must remain identical: no hidden bypass exists.
    for boundary in range(5):
        unit=model.units[model.unit_order[boundary]];occurrence=boundary%2
        calls=[0];head_calls=[0]
        def disturb(module,args,value):
            call=calls[0];calls[0]+=1
            return value+.37*g['valid'][...,None] if call==occurrence else value
        def fixed(module,args,value):
            call=head_calls[0];head_calls[0]+=1
            return states[2*boundary+1].clone() if call==occurrence else value
        h=unit.layers[1].register_forward_hook(disturb);f=unit.heads[1].register_forward_hook(fixed)
        try:
            with torch.no_grad():altered=model(g)
        finally:h.remove();f.remove()
        torch.testing.assert_close(altered[2*boundary+2:],states[2*boundary+2:],rtol=0,atol=0)
    # Inside a unit the hidden path must remain active when its first layout is fixed.
    unit=model.units[0];calls=[0];head_calls=[0]
    def disturb_first(module,args,value):
        k=calls[0];calls[0]+=1
        return value+.37*g['valid'][...,None] if k==0 else value
    def fix_first(module,args,value):
        k=head_calls[0];head_calls[0]+=1
        return states[0].clone() if k==0 else value
    h=unit.layers[0].register_forward_hook(disturb_first);f=unit.heads[0].register_forward_hook(fix_first)
    try:
        with torch.no_grad():altered=model(g)
    finally:h.remove();f.remove()
    assert not torch.equal(altered[1],states[1])
    # Initial hidden cannot leak past a fixed initial layout into pure P conditioning.
    initial=[];hook=model.head.register_forward_hook(lambda m,a,v:initial.append(v.detach().clone()))
    with torch.no_grad():model(g)
    hook.remove()
    h=model.initial_blocks[-1].register_forward_hook(lambda m,a,v:v+.37*g['valid'][...,None])
    f=model.head.register_forward_hook(lambda m,a,v:initial[0].clone())
    try:
        with torch.no_grad():altered=model(g)
    finally:h.remove();f.remove()
    torch.testing.assert_close(altered,states,rtol=0,atol=0)
    # Final objective must still backpropagate through layout across all five cuts.
    emitted=[];hooks=[]
    for unit in model.units:
        for head in unit.heads:hooks.append(head.register_forward_hook(lambda m,a,v:emitted.append(v)))
    try:
        output=model(g);loss=per_case(output,batch,ARMS[0])['final_total'].mean()
        gradients=torch.autograd.grad(loss,[emitted[k] for k in (1,3,5,7,9)])
        assert all(bool(torch.isfinite(v).all()) and bool(v.abs().sum()>0) for v in gradients)
    finally:
        for h in hooks:h.remove()
    return dict(shared_unit_calls=counts,first_static_input=81,second_dynamic_only_input=65,
        hidden_cuts_all_five_verified=True,internal_hidden_retained=True,
        initial_hidden_isolated=True,layout_gradients_across_cuts=True,repeat_operator_bitwise=True)

def main():
    started=time.perf_counter();result=dict(status='preparing',formal_training_started=False,dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0)
    try:
        assert not (HERE/'runs').exists() and not read(HERE/'authorization.json')['training_enabled']
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        result['synthetic']=synthetic();print('Independent shape oracle and unchanged E14249 passed',flush=True)
        plan=read(HERE/'plan.json');result['cache']=cache_gate(plan)
        # Inherit unchanged tested train/signal/schedule/checkpoint implementation.
        parent=ROOT/'experiments/E147_supervision_hidden_reset';checked=[]
        for name in ('train.py','cuda_step.py','captured_projection.py','step_control.py','resident.py','schedule.py'):
            expected=(parent/name).read_text(encoding='utf-8').replace('experiments.E147_supervision_hidden_reset','experiments.E148_shared_functional_shape').replace('E147.supervision_hidden_reset.v1.seed0','E148.shared_functional_shape.v1.seed0').replace('E147','E148')
            assert (HERE/name).read_text(encoding='utf-8')==expected,name
            checked.append(name)
        assert read(parent/'preflight.json')['ctrl_c_resume_verified']
        result['inherited_engine']=dict(parent_sha256=sha(parent/'preflight.json'),checked_files=checked)
        train.seed_all(0);rows=load_rows('train',plan['preflight_train_indices'])
        assert len({tuple(r['member']) for r in rows})==5
        result['members']=[r['member'] for r in rows]
        model=GeometryModel(**train.MODEL).cuda().eval()
        assert sum(p.numel() for p in model.parameters())==plan['parameters']['e148']
        assert all(p.requires_grad for p in model.parameters())
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145)
        batch=pack.batch(range(5));public=inputs_only(batch)
        with torch.no_grad():states=model(public)
        assert states.shape==(12,5,120,3) and bool(torch.isfinite(states).all())
        try:model(dict(public,target=batch['target']))
        except ValueError:pass
        else:raise AssertionError('Teacher accepted')
        assert bool((public['mask'][public['valid']]==1).all())
        result['objective']=objective_check(states,batch)
        result['interfaces']=interface_check(model,public,states,batch)
        from .original_model import RectangleBlock
        rectangle=RectangleBlock(64).cuda().eval();pairs=[]
        hook=rectangle.bias.register_forward_pre_hook(lambda m,a:pairs.append(a[0]))
        pair,_=pair_features(states[0],public)
        with torch.no_grad():rectangle(torch.zeros((5,120,64),device='cuda'),public,states[0])
        hook.remove();torch.testing.assert_close(pair,pairs[0],rtol=0,atol=0)
        perm=torch.arange(119,-1,-1,device='cuda')
        pg={k:v[:,perm][:,:,perm] if k in ('group','adjacency') else v if k=='scale' else v[:,perm] for k,v in public.items()}
        with torch.no_grad():other=model(pg)
        torch.testing.assert_close(other,states[:,:,perm],rtol=1e-4,atol=2e-5)
        result['public_boundary']=dict(labels_rejected=True,pair18_exact=True,predict_all_F_P=True,node_permutation=True)
        print('Sharing, hidden interventions, entry-only conditioning, loss and input gates passed',flush=True)
        del rectangle,pairs,pair,states,other,pg,batch,public
        model.train();optimizer=train.optimizer_for(model)
        initial=copy.deepcopy(model.state_dict());rng=train.rng_state()
        t=time.perf_counter();cap=CapturedStep(model,optimizer,pack.batch([i%5 for i in range(64)]),ARMS[0]);capture_seconds=time.perf_counter()-t
        assert_tree(model.state_dict(),initial);assert not optimizer.state
        train.restore_rng(rng);print('CUDA64 captured without optimizer updates',flush=True)
        times=[]
        for _ in range(2):
            t=time.perf_counter();values,norm=cap.replay();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        for name,p in model.named_parameters():
            assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum()>0),'No joint gradient: '+name
        sample=pack.batch([i%5 for i in range(32)])
        tail=EagerStep(model,optimizer,sample,ARMS[0],cap.projection);tail.replay();cap.replay()
        assert cap.projection.steps==4
        # Serialize real shared-parameter Adam state, then compare identical next eager32 update.
        checkpoint=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=train.rng_state(),projection=cap.projection.summary()),checkpoint)
        checkpoint.seek(0);saved=torch.load(checkpoint,weights_only=False)
        clone=GeometryModel(**train.MODEL).cuda().train();clone.load_state_dict(saved['model'])
        other_opt=train.optimizer_for(clone);other_opt.load_state_dict(saved['optimizer'])
        from .step_control import GraphStepControl
        projection=GraphStepControl(clone);projection.steps=saved['projection']['steps']
        projection.clipped.copy_(torch.tensor(saved['projection']['clipped_steps_per_group'],device='cuda'))
        projection.last_scales.copy_(torch.tensor(saved['projection']['last_scales'],device='cuda'))
        resumed=EagerStep(clone,other_opt,sample,ARMS[0],projection)
        train.restore_rng(saved['rng']);tail.replay()
        train.restore_rng(saved['rng']);resumed.replay()
        assert_tree(model.state_dict(),clone.state_dict());assert_tree(optimizer.state_dict(),other_opt.state_dict())
        assert_tree(cap.projection.summary(),projection.summary())
        result.update(status='passed',ctrl_c_resume_verified=True,parameters=plan['parameters']['e148'],all_joint_gradients=True,
            batch64=True,tail32=True,capture_without_updates=True,serialized_shared_Adam_next_update_bitwise=True,
            disposable_updates_executed=5,formal_model_not_reused=True,capture_seconds=capture_seconds,
            step_seconds=times,last_short_loss=float(values['total']),last_gradient_norm=float(norm),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),sources=sources())
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','cache','members')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)

if __name__=='__main__':main()
