"""Verify the frozen initial and guarded E151 backend updates before launch."""
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
from .io import ROOT,HERE,save,sha,sources
from . import train
from .data import load_rows,cache_identity,records
from .resident import ResidentRows
from .public import inputs_only
from .role_features import build
from .model import GeometryModel
from .initialization import load_parent_initial, verify_parent_initial
from .loss import per_case,TOTAL_COMPONENTS
from .loss_config import ARMS
from .cuda_step import CapturedStep,EagerStep
from .stage_audit import baselines_for,read_pinned,stage_report

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def synthetic():
    wire=torch.zeros((4,4))
    for a,b in ((0,1),(1,2)):wire[a,b]=wire[b,a]=.5
    c=torch.zeros((4,5),dtype=torch.long);c[0,1]=1;c[2,0]=1
    coordinates=torch.zeros((4,2,2));coordinates[2]=torch.tensor([[10.,5.],[10.,5.]])
    valid=torch.zeros((4,2),dtype=torch.bool);valid[2]=True
    weights=torch.zeros((4,2));weights[2]=torch.tensor([1.,2.])
    g=dict(n=4,wire_b2b=wire,constraints=c,pin_coordinates=coordinates,pin_valid=valid,pin_weights=weights)
    role=build(g);assert role['hops'].shape==(4,5)
    # Independent breadth-first search, including a referenced port token.
    adjacency={0:[1],1:[0,2],2:[1,4],3:[],4:[2]}
    expected=np.full((4,5),-1,dtype=np.int16)
    for source in range(4):
        expected[source,source]=0;queue=[source]
        for node in queue:
            for other in adjacency[node]:
                if expected[source,other]<0:expected[source,other]=expected[source,node]+1;queue.append(other)
    np.testing.assert_array_equal(role['hops'],expected)
    np.testing.assert_allclose(role['role_features'][:,9],np.arcsinh([0,1,2,0]),rtol=1e-6)
    np.testing.assert_array_equal(role['role_features'][:,10],[1,1,1,0])
    np.testing.assert_allclose(role['role_features'][:,11],np.arcsinh([3,2,1,0]),rtol=1e-6)
    assert role['port_edges'][2,0]==3 and len(role['port_features'])==1
    noP=dict(g,constraints=c.clone());noP['constraints'][:,1]=0
    assert np.count_nonzero(build(noP)['role_features'][:,10])==0
    altered=dict(g,target=torch.full((4,3),999.),reference=torch.full((4,4),999.))
    for k,v in build(altered).items():np.testing.assert_array_equal(v,role[k])
    empty=dict(g,pin_valid=torch.zeros_like(valid),pin_weights=torch.zeros_like(weights))
    assert build(empty)['port_features'].shape==(0,4) and np.isfinite(build(empty)['role_features']).all()
    perm=np.array([3,2,0,1]);reordered=dict(g,wire_b2b=wire[perm][:,perm],constraints=c[perm],pin_coordinates=coordinates[perm],pin_valid=valid[perm],pin_weights=weights[perm])
    other=build(reordered);keys=np.r_[perm,4]
    np.testing.assert_array_equal(other['hops'],role['hops'][perm][:,keys])
    np.testing.assert_allclose(other['role_features'],role['role_features'][perm],rtol=0,atol=0)
    return dict(independent_BFS_exact=True,unreachable_separate=True,F_not_position_anchor=True,
        duplicate_coordinate_ports_preserve_weight=True,no_port_case=True,labels_ignored=True,node_permutation=True)

def cache_gate(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan);key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert key==plan['required_cache_key'];base=ROOT/'datasets/static_tensor_cache_v1'/key
    assert read(base/'identity.json')==identity;counts={}
    for role in ('train','dev_monitor500'):
        ds=RandomReference(role);blocks={}
        for i in ds.indices:blocks.setdefault(i//256,[]).append(i)
        for block,ids in blocks.items():
            path=base/ds.role/f'{block:04d}.pt';meta=read(path.with_suffix('.json'))
            assert path.exists() and meta['identity']==key and all(meta['members'][str(i)]==ds.records[i]['member'] for i in ids)
        counts[role]=len(ds.indices)
    return dict(key=key,counts=counts,metadata_only=True)

from .joint_checks import objective,interfaces

def main():
    started=time.perf_counter();result=dict(status='preparing',formal_training_started=False,dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0)
    try:
        assert not (HERE/'runs').exists() and not read(HERE/'authorization.json')['training_enabled']
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        result['synthetic']=synthetic();print('Public-role BFS/anchor/label gates passed',flush=True)
        plan=read(HERE/'plan.json');result['cache']=cache_gate(plan)
        comparison = baselines_for(plan, [row['member'] for row in records('dev_monitor500')])
        for name, stages in comparison.items():
            assert all(np.isfinite(v).all() for v in stages.values())
            assert stage_report(stages)['S1_L8_minus_L4']['overlap_weighted']['mean'] < 0, name
        parent_losses = read_pinned(plan['baselines']['E149_loss'])
        assert parent_losses['checkpoint_sha256'] == plan['parent_initial']['checkpoint_sha256']
        assert len(parent_losses['dev_components']['per_case_total']) == 500
        result['paired_baselines_verified'] = True
        parent=ROOT/'experiments/E149_role_initial'
        for name in ('features.py','batches.py','ports.py'):assert (HERE/name).read_bytes()==(parent/name).read_bytes()
        engine_parent=ROOT/'experiments/E150_role_joint'
        for name in ('cuda_step.py','captured_projection.py','step_control.py','schedule.py'):
            assert (HERE/name).read_bytes()==(engine_parent/name).read_bytes(),name
        assert read(parent/'preflight.json')['ctrl_c_resume_verified']
        result['engine']=dict(parent_resume_gate=True,core_engine_preserved=True)
        train.seed_all(0);rows=load_rows('train',plan['preflight_train_indices'])
        assert len({tuple(r['member']) for r in rows})==5;result['members']=[r['member'] for r in rows]
        # Pad beyond all five actual port counts; no token truncation is permitted.
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145,max_ports=max(512,max(len(r['role']['port_features']) for r in rows)))
        batch=pack.batch(range(5));public=inputs_only(batch)
        model=GeometryModel(**train.MODEL).cuda().eval()
        parent_initial=load_parent_initial(plan)
        model.initial_generator.load_state_dict(parent_initial)
        verify_parent_initial(model,parent_initial)
        assert sum(p.numel() for p in model.parameters())==plan['parameters']['e151']
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==plan['trainable_parameters']['e151']
        assert len(model.units)==3
        with torch.no_grad():states=model(public)
        assert states.shape==(13,5,120,3) and bool(torch.isfinite(states).all())
        try:model(dict(public,target=batch['target']))
        except ValueError:pass
        else:raise AssertionError('Teacher accepted')
        assert bool((public['mask'][public['valid']]==1).all())
        result['objective']=objective(states,batch)
        result['interfaces']=interfaces(model,public,states,batch)
        # Module and port reorderings, including relation rows/columns.
        perm=torch.arange(119,-1,-1,device='cuda');mp=public['port_valid'].shape[1];pp=torch.arange(mp-1,-1,-1,device='cuda')
        keys=torch.cat((perm,120+pp));pg={}
        for k,v in public.items():
            if k in ('adjacency','group'):pg[k]=v[:,perm][:,:,perm]
            elif k in ('role_hops','role_edges','role_group'):pg[k]=v[:,perm][:,:,keys]
            elif k in ('port_features','port_valid'):pg[k]=v[:,pp]
            elif k=='scale':pg[k]=v
            else:pg[k]=v[:,perm]
        with torch.no_grad():other=model(pg)
        torch.testing.assert_close(other,states[:,:,perm],rtol=1e-4,atol=2e-5)
        from .augmentation import axis_swap
        augmented=axis_swap(dict(batch,axis_swap=torch.ones(5,device='cuda',dtype=torch.bool)))
        torch.testing.assert_close(augmented['port_features'][...,0],batch['port_features'][...,1],rtol=0,atol=0)
        torch.testing.assert_close(augmented['role_hops'],batch['role_hops'],rtol=0,atol=0)
        swapped=torch.stack((states[...,1],states[...,0],-states[...,2]),-1)
        torch.testing.assert_close(per_case(swapped,augmented,ARMS[0])['total'],per_case(states,batch,ARMS[0])['total'],rtol=1e-5,atol=2e-5)
        result['boundary']=dict(all_F_P_channels_predicted=True,original_refinement_modules=True,module_and_port_permutation=True,axis_swap_including_port_tokens=True,labels_rejected=True)
        print('Joint13-readout objective, hidden interfaces, permutations and augmentation passed',flush=True)
        del batch,public,states,other,pg,augmented,swapped
        import gc
        gc.collect();torch.cuda.empty_cache()
        model.train();opt=train.optimizer_for(model);initial=copy.deepcopy(model.state_dict());rng=train.rng_state()
        verify_parent_initial(model,parent_initial)
        assert all(p.requires_grad for group in opt.param_groups for p in group['params'])
        sample=pack.batch([i%5 for i in range(64)])
        begin=time.perf_counter();cap=CapturedStep(model,opt,sample,ARMS[0]);capture_seconds=time.perf_counter()-begin
        equal(model.state_dict(),initial);assert not opt.state;train.restore_rng(rng)
        print('CUDA64 stress capture passed without optimizer updates',flush=True)
        times=[]
        for _ in range(2):
            begin=time.perf_counter();values,norm=cap.replay();torch.cuda.synchronize();times.append(time.perf_counter()-begin)
        for name,p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and bool(torch.isfinite(p.grad).all()) and bool(p.grad.abs().sum()>0),name
        verify_parent_initial(model,parent_initial)
        sample32=pack.batch([i%5 for i in range(32)]);tail=EagerStep(model,opt,sample32,ARMS[0],cap.projection)
        tail.replay();cap.replay();assert cap.projection.steps==4
        buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=train.rng_state(),projection=cap.projection.summary()),buffer)
        buffer.seek(0);saved=torch.load(buffer,weights_only=False)
        clone=GeometryModel(**train.MODEL).cuda().train();clone.load_state_dict(saved['model']);other_opt=train.optimizer_for(clone);other_opt.load_state_dict(saved['optimizer'])
        from .step_control import GraphStepControl
        control=GraphStepControl(clone);control.steps=saved['projection']['steps']
        control.clipped.copy_(torch.tensor(saved['projection']['clipped_steps_per_group'],device='cuda'))
        control.last_scales.copy_(torch.tensor(saved['projection']['last_scales'],device='cuda'))
        resumed=EagerStep(clone,other_opt,sample32,ARMS[0],control)
        train.restore_rng(saved['rng']);tail.replay();train.restore_rng(saved['rng']);resumed.replay()
        equal(model.state_dict(),clone.state_dict());equal(opt.state_dict(),other_opt.state_dict());equal(cap.projection.summary(),control.summary())
        verify_parent_initial(model,parent_initial)
        verify_parent_initial(clone,parent_initial)
        result.update(status='passed',parameters=plan['parameters']['e151'],trainable_parameters=plan['trainable_parameters']['e151'],
            frozen_l0_exact_after_updates_and_reload=True,all_backend_gradients=True,
            batch64=True,tail32=True,max_stress_ports=pack.max_ports,actual_five_port_counts=[len(r['role']['port_features']) for r in rows],
            capture_without_updates=True,serialized_next_update_bitwise=True,disposable_updates=5,formal_model_not_reused=True,
            capture_seconds=capture_seconds,step_seconds=times,peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            short_loss=float(values['total']),gradient_norm=float(norm),sources=sources())
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','cache','members')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)

if __name__=='__main__':main()
