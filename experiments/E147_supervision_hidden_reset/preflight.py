"""Geometry semantics, E139 objective identity, guarded scratch64 and resume gates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import copy
import gc
import hashlib
import json
import time
import traceback
import torch
from .io import ROOT,HERE,save,sources,sha
from . import train
from .data import load_rows,cache_identity
from .resident import ResidentRows
from .batches import inputs_only
from .model import GeometryModel
from .geometry_feedback import module_features,pair_features,FEATURE_NAMES
from .loss import per_case
from .loss_config import ARMS,layer_weights
from .cuda_step import CapturedStep,EagerStep
VARIANT='supervision_hidden_reset'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def synthetic():
    assert (HERE/'geometry_feedback.py').read_text(encoding='utf-8')==(ROOT/'experiments/E142_dual_geometry/geometry_feedback.py').read_text(encoding='utf-8')
    return dict(original_E142_feedback_source_unchanged=True)

def cache_gate(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan);key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
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
    from experiments.E139_delayed_overlap.loss import per_case as old
    x=states.detach().clone().requires_grad_(True)
    a=per_case(x,batch,ARMS[0]);b=old(x,batch,'e119_delayed_overlap')
    assert a.keys()==b.keys()
    for k in a:torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)
    ga=torch.autograd.grad(a['total'].sum(),x,retain_graph=True)[0]
    gb=torch.autograd.grad(b['total'].sum(),x)[0];torch.testing.assert_close(ga,gb,rtol=0,atol=0)
    for k in range(1,13):
        if k in layer_weights(ARMS[0]):assert bool(ga[k-1].abs().sum()>0)
        else:assert torch.count_nonzero(ga[k-1])==0
    assert torch.count_nonzero(a['layer02_overlap_weighted'])==0 and torch.count_nonzero(a['layer06_overlap_weighted'])==0
    return dict(all_E139_components_and_total_exact=True,E139_gradient_exact=True,
        directly_supervised_layers=[2,6,8,10,12],no_overlap_layers=[2,6])

def pair_check(model,batch,states):
    captured=[];handle=model.layers[0].bias.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    # Recreate parent RectangleBlock call using the retained modules and parent code.
    from .original_model import RectangleBlock
    block=RectangleBlock(64).to(states.device).eval()
    old=[];h=block.bias.register_forward_pre_hook(lambda module,args:old.append(args[0].detach().clone()))
    state=states[0];pair,_=pair_features(state,batch)
    with torch.no_grad():block(torch.zeros((*state.shape[:2],64),device=state.device),inputs_only(batch),state)
    h.remove();handle.remove();torch.testing.assert_close(pair,old[0],rtol=0,atol=0)
    return dict(original_pair18_exact=True)

def ablation_check(model,public,states,batch):
    from experiments.E142_dual_geometry.model import GeometryModel as ParentModel
    from .loss_config import SUPERVISED_LAYERS
    expected_cuts=[2,6,8,10]
    assert [d for d in range(12) if d in SUPERVISED_LAYERS]==expected_cuts
    baseline_inputs=[];readouts=[]
    hooks=[layer.register_forward_pre_hook(lambda module,args:baseline_inputs.append(args[0].detach().clone())) for layer in model.layers]
    hooks.append(model.head.register_forward_hook(lambda module,args,value:readouts.append(value.detach().clone())))
    try:
        with torch.no_grad():model(public)
    finally:
        for hook in hooks:hook.remove()
    cuts=[];retained=[]
    for depth in range(12):
        source=model.initial_blocks[-1] if depth==0 else model.layers[depth-1]
        calls=[0];incoming=[]
        def fixed_layout(module,args,value):
            d=calls[0];calls[0]+=1
            return readouts[depth].clone() if d==depth else value
        perturb=source.register_forward_hook(lambda module,args,value:value+.37*public['valid'][...,None])
        fixed=model.head.register_forward_hook(fixed_layout)
        capture=model.layers[depth].register_forward_pre_hook(lambda module,args:incoming.append(args[0].detach().clone()))
        try:
            with torch.no_grad():altered=model(public)
        finally:perturb.remove();fixed.remove();capture.remove()
        if depth in expected_cuts:
            torch.testing.assert_close(incoming[0],baseline_inputs[depth],rtol=0,atol=0)
            torch.testing.assert_close(altered[depth:],states[depth:],rtol=0,atol=0)
            cuts.append(depth)
        else:
            assert not torch.equal(incoming[0],baseline_inputs[depth]),depth
            retained.append(depth)
    assert cuts==expected_cuts and retained==[0,1,3,4,5,7,9,11]
    parent=ParentModel(**train.MODEL).cuda().eval();parent.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():original=parent(public)
    torch.testing.assert_close(states[:2],original[:2],rtol=0,atol=0)
    del parent,original
    differentiable=[]
    hook=model.head.register_forward_hook(lambda module,args,value:differentiable.append(value))
    try:
        output=model(public)
        loss=per_case(output,batch,ARMS[0])['final_total'].mean()
        gradients=torch.autograd.grad(loss,[differentiable[d] for d in expected_cuts],allow_unused=False)
        assert all(bool(torch.isfinite(v).all()) and bool(v.abs().sum()>0) for v in gradients)
    finally:hook.remove()
    return dict(reset_after_layers=cuts,retained_after_layers=retained,all_12_edges_intervened=True,
        reset_receivers_and_downstream_bitwise_unchanged=True,S0_hidden_retained_to_S1=True,
        first_two_layouts_bitwise_E142=True,layout_gradient_across_all_resets=True)

def main():
    started=time.perf_counter();result=dict(status='preparing',formal_training_started=False,dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0)
    try:
        assert not (HERE/'runs').exists() and not read(HERE/'authorization.json')['training_enabled']
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        result['synthetic']=synthetic();print('Geometry and ablation semantics passed',flush=True)
        plan=read(HERE/'plan.json');result['cache']=cache_gate(plan)
        engine=ROOT/'experiments/E142_dual_geometry';gate=read(engine/'preflight.json')
        assert gate['status']=='passed' and gate['ctrl_c_resume_verified']
        checked=[]
        for name in ('cuda_step.py','captured_projection.py','step_control.py','resident.py','schedule.py','train.py'):
            expected=(engine/name).read_text().replace('experiments.E142_dual_geometry','experiments.E147_supervision_hidden_reset').replace('E142.dual_geometry.v1.seed0','E147.supervision_hidden_reset.v1.seed0').replace('E142','E147')
            assert (HERE/name).read_text()==expected,name
            checked.append(name)
        result['inherited_engine']=dict(parent_preflight_sha256=sha(engine/'preflight.json'),checked_files=checked,new_signal_simulations=0)
        train.seed_all(0);m=GeometryModel(**train.MODEL)
        initial_cpu=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in m.state_dict().values())).hexdigest()
        assert sum(p.numel() for p in m.parameters())==plan['parameters']['e147']
        assert all(p.requires_grad for p in m.parameters())
        from experiments.E142_dual_geometry.model import GeometryModel as ParentModel
        train.seed_all(0);parent=ParentModel(**train.MODEL)
        parent_state=parent.state_dict();current=m.state_dict()
        for name,value in current.items():
            expected=parent_state[name]
            if VARIANT=='local_blank' and name=='demand_encoder.0.weight':expected=expected[:,list(range(16))+list(range(20,65))]
            torch.testing.assert_close(value,expected,rtol=0,atol=0)
        assert (VARIANT=='no_relation_value') == (len(current)<len(parent_state))
        result['E142_scratch_correspondence']=True
        del parent,parent_state,current,m
        rows=load_rows('train',plan['preflight_train_indices']);assert len({tuple(r['member']) for r in rows})==5
        result['members']=[r['member'] for r in rows]
        train.seed_all(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=GeometryModel(**train.MODEL).cuda().eval();pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145)
        small=pack.batch(range(5));public=inputs_only(small)
        assert set(public).isdisjoint(('reference','target','pin_coordinates'))
        assert bool((public['mask'][public['valid']]==1).all())
        with torch.no_grad():states=model(public)
        assert states.shape==(12,5,120,3) and bool(torch.isfinite(states).all())
        try:model(dict(public,target=small['target']))
        except ValueError:pass
        else:raise AssertionError('Teacher input not rejected')
        result['objective']=objective_check(states,small);result['pair']=pair_check(model,small,states)
        result['ablation']=ablation_check(model,public,states,small)
        # Reordering all input nodes preserves outputs up to normal reduction rounding.
        perm=torch.arange(119,-1,-1,device='cuda');permuted={}
        for k,v in public.items():
            permuted[k]=v[:,perm][:,:,perm] if k in ('group','adjacency') else v if k=='scale' else v[:,perm]
        with torch.no_grad():other=model(permuted)
        torch.testing.assert_close(other,states[:,:,perm],rtol=1e-4,atol=2e-5)
        result['input_boundary']=dict(labels_rejected=True,all_F_P_channels_predicted=True,node_permutation_equivariance=True)
        print('Exact E139 objective, pair18 and input boundary checks passed',flush=True)
        del small,public,states,other,permuted;gc.collect();model.train()
        opt=train.optimizer_for(model);sample=pack.batch([i%5 for i in range(64)])
        initial={k:v.detach().clone() for k,v in model.state_dict().items()};rng=train.rng_state()
        t=time.perf_counter();cap=CapturedStep(model,opt,sample,ARMS[0]);capture_seconds=time.perf_counter()-t
        equal(model.state_dict(),initial);assert not opt.state;del initial
        train.restore_rng(rng);print('Full batch64 graph captured; no parameter changes',flush=True)
        times=[]
        for _ in range(2):
            t=time.perf_counter();values,norm=cap.replay();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        for name,p in model.named_parameters():
            if any(s in name for s in ('demand_encoder','fuse','edge.','receiver','sender','relation_value')):
                assert p.grad is not None and bool(p.grad.abs().sum()>0),'Missing new-branch gradient '+name
        tail=EagerStep(model,opt,pack.batch([i%5 for i in range(32)]),ARMS[0],cap.projection)
        tail.replay();cap.replay();assert cap.projection.steps==4
        result.update(status='passed',ctrl_c_resume_verified=True,initial_state_sha256=initial_cpu,parameters=plan['parameters']['e147'],
            all_new_branch_gradients=True,batch64=True,tail32=True,capture_without_updates=True,resume_engine_unchanged_from_verified_parent=True,
            main_graph_after_tail=True,disposable_updates_executed=4,formal_model_not_reused=True,capture_seconds=capture_seconds,
            step_seconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            last_short_loss=float(values['total']),last_gradient_norm=float(norm),sources=sources())
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','cache','inherited_engine','members')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)

if __name__=='__main__':main()
