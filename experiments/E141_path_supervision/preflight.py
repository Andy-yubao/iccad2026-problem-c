"""Synthetic geometry plus five-member disposable batch64 gates; no formal run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import copy
import gc
import hashlib
import json
import time
import traceback
import numpy as np
import torch
from .io import ROOT,HERE,save,sha,sources
from . import train
from .data import load_rows,cache_identity
from .resident import ResidentRows
from .batches import inputs_only
from .model import GeometryModel
from .loss import base_per_case,supervised_per_case,per_case,TOTAL_COMPONENTS
from .loss_config import ARMS,layer_weights,FULL_LAYERS,has_blockage
from .cuda_step import CapturedStep,EagerStep
from .blockage_loss import strip_cost,route_cost,components as block_components

def read(path):return json.loads(path.read_text(encoding='utf-8'))

def synthetic():
    p=torch.tensor([[[0.,0.,2.,2.],[6.,0.,2.,2.],[3.,.3,1.,1.2]]],dtype=torch.double,requires_grad=True)
    scale=torch.ones((1,1,3),dtype=torch.double)*2
    allowed=torch.tensor([[[False,False,True]]])
    cost=lambda q:route_cost(q[:,:1],q[:,1:2],q,scale,allowed).sum()
    v=cost(p);assert v>0
    clear=p.detach().clone();clear[0,2,1]=4;assert cost(clear)==0
    other=clear.clone();other[0,2,0]=10;assert cost(other)==0
    touching=p.detach().clone();touching[0,1,0]=2;assert cost(touching)==0
    overlap=p.detach().clone();overlap[0,2,:]=torch.tensor([.2,.3,1.,1.2]);assert cost(overlap)==0
    assert route_cost(p[:,:1],p[:,1:2],p,scale,torch.zeros_like(allowed)).sum()==0
    torch.testing.assert_close(v,route_cost(p[:,1:2],p[:,:1],p,scale,allowed).sum())
    for factor in (.1,10.):
        torch.testing.assert_close(v,route_cost(p[:,:1]*factor,p[:,1:2]*factor,p*factor,scale*factor,allowed).sum())
    trans=torch.tensor([2.1,-3.7,0.,0.],dtype=torch.double)
    torch.testing.assert_close(v,cost(p+trans))
    swap=torch.tensor([1,0,3,2]);torch.testing.assert_close(v,cost(p[...,swap]))
    # Finite differences away from discrete route/orientation changes.
    grad=torch.autograd.grad(v,p)[0];eps=1e-6
    for index in ((0,0,0),(0,2,2),(0,2,3)):
        plus=p.detach().clone();minus=plus.clone();plus[index]+=eps;minus[index]-=eps
        torch.testing.assert_close(grad[index],(cost(plus)-cost(minus))/(2*eps),rtol=1e-5,atol=1e-7)
    # Boundary obstruction on every side; each target already at its boundary is zero.
    for side in range(4):
        q=torch.tensor([[[4.,4.,2.,2.],[2.,4.3,1.,1.2],[0.,0.,1.,1.],[9.,9.,1.,1.]]],dtype=torch.double)
        q[0,1,:2]=torch.tensor([(2.,4.3),(7.,4.3),(4.3,7.),(4.3,2.)][side])
        valid=torch.ones((1,4),dtype=torch.bool);req=torch.zeros((1,4,4),dtype=torch.bool);req[0,0,side]=True
        rel=torch.zeros((1,4,4),dtype=torch.bool);tree=rel.clone();area=q[...,2:].prod(-1)
        out=block_components(q,area,valid,rel,tree,req,torch.zeros((1,2)),torch.ones((1,2))*10)
        assert out['boundary_blockage']>0
        far=q.clone();far[0,1,:2]=torch.tensor([7.,7.]) if side in (0,3) else torch.tensor([2.,2.])
        assert block_components(far,area,valid,rel,tree,req,torch.zeros((1,2)),torch.ones((1,2))*10)['boundary_blockage']==0
    return dict(clear_zero=True,blocked_positive=True,all_four_boundary_directions=True,
        existing_overlap_not_counted_as_path=True,contact_zero=True,group_relay_excluded=True,
        finite_difference=True,translation_scale_axis_invariance=True,reverse_direction=True)

def objective_check(states,g,arm):
    x=states.detach().clone().requires_grad_(True)
    out=per_case(x,g,arm);expected=0
    from experiments.E140_frontend_gb4.loss import base_per_case as parent
    for k,w in layer_weights(arm).items():
        common=base_per_case(x[k-1],g)
        inherited=parent(x[k-1],g)
        for key in TOTAL_COMPONENTS:torch.testing.assert_close(common[key],inherited[key],rtol=0,atol=0)
        v=supervised_per_case(x[k-1],g,k,arm)
        if k not in FULL_LAYERS:assert torch.count_nonzero(v['overlap_weighted'])==0
        for key in ('grouping_weighted','boundary_weighted'):
            torch.testing.assert_close(v[key],common[key],rtol=0,atol=0)
        retained=sum(v[key] for key in TOTAL_COMPONENTS)+v['grouping_blockage_weighted']+v['boundary_blockage_weighted']
        torch.testing.assert_close(v['total'],retained,rtol=0,atol=0)
        assert bool((v['grouping_blockage_weighted']>=0).all()&(v['boundary_blockage_weighted']>=0).all())
        if not has_blockage(arm):assert torch.count_nonzero(v['grouping_blockage_weighted']+v['boundary_blockage_weighted'])==0
        expected=expected+w*retained
    torch.testing.assert_close(out['total'],expected,rtol=0,atol=0)
    gradient=torch.autograd.grad(out['total'].sum(),x)[0]
    for k in range(1,13):
        if k in layer_weights(arm):assert bool(gradient[k-1].abs().sum()>0)
        else:assert torch.count_nonzero(gradient[k-1])==0
    torch.testing.assert_close(out['common_final_total'],parent(x[11],g)['total'],rtol=0,atol=0)
    return dict(original_components_exact=True,aggregate_exact=True,layer_gradient_scope=True,
        overlap_scope=True,original_GB_not_multiplied=True,common_final_preserved=True)

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def cache_gate(plan):
    identity=cache_identity(plan);key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert key==plan['required_cache_key']
    base=ROOT/'datasets/static_tensor_cache_v1'/key;assert read(base/'identity.json')==identity
    from datasets.random_reference import RandomReference
    result={}
    for role in ('train','dev_monitor500'):
        ds=RandomReference(role);blocks={}
        for i in ds.indices:blocks.setdefault(i//256,[]).append(i)
        for b,indices in blocks.items():
            p=base/ds.role/f'{b:04d}.pt';meta=read(p.with_suffix('.json'));assert p.exists() and meta['identity']==key
            assert all(meta['members'][str(i)]==ds.records[i]['member'] for i in indices)
        result[role]=len(ds.indices)
    return result

def main():
    started=time.perf_counter();result=dict(status='preparing',formal_training_started=False,dev_payloads=0,test_payloads=0,postprocessing_calls=0,arms=[])
    try:
        assert not (HERE/'runs').exists() and not read(HERE/'authorization.json')['training_enabled']
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        result['synthetic']=synthetic();print('Synthetic geometry/gradients passed',flush=True)
        plan=read(HERE/'plan.json');result['cache']=cache_gate(plan);print('Existing cache coverage passed',flush=True)
        parent=ROOT/'experiments/E140_frontend_gb4';pg=read(parent/'preflight.json')
        assert pg['status']=='passed' and pg['ctrl_c_resume_verified']
        # State serialization and safe-boundary controller code is inherited.
        for name in ('schedule.py','io.py','resident.py','captured_projection.py','augmentation.py'):
            assert (HERE/name).read_text()==(parent/name).read_text().replace('E140','E141')
        result['inherited_controller']=dict(parent_preflight_sha256=sha(parent/'preflight.json'),
            previous_signal_and_evaluation_prefix_checks=pg['continuation_proof'],new_signal_simulations=0)
        train.seed_all(0);model=GeometryModel(**train.MODEL)
        digest=hashlib.sha256(b''.join(v.contiguous().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
        assert digest==pg['initial_state_sha256'];del model
        rows=load_rows('train',plan['preflight_train_indices']);assert len({tuple(r['member']) for r in rows})==5
        result['members']=[r['member'] for r in rows]
        for arm in ARMS:
            print('Checking '+arm,flush=True)
            train.seed_all(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=GeometryModel(**train.MODEL).cuda().train();assert sum(p.numel() for p in model.parameters())==712275
            pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145)
            small=pack.batch(range(5));model.eval()
            with torch.no_grad():states=model(inputs_only(small))
            checked=objective_check(states,small,arm)
            print('Objective and layer gradients passed '+arm,flush=True)
            del small,states;gc.collect();model.train()
            opt=train.optimizer_for(model);sample=pack.batch([i%5 for i in range(64)])
            rng=train.rng_state();initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            t=time.perf_counter();cap=CapturedStep(model,opt,sample,arm);capture_seconds=time.perf_counter()-t
            print('Captured batch64 '+arm,flush=True)
            equal(model.state_dict(),initial);assert not opt.state
            train.restore_rng(rng);del initial
            t=time.perf_counter();values,norm=cap.replay();torch.cuda.synchronize();step_seconds=time.perf_counter()-t
            assert opt.state and cap.projection.steps==1
            for block in (model.initial_blocks,model.layers):
                assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in block.parameters())
            tail=EagerStep(model,opt,pack.batch([i%5 for i in range(32)]),arm,cap.projection)
            saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=train.rng_state(),projection=copy.deepcopy(cap.projection.summary()))
            path=HERE/'preparation_diagnostics'/f'{arm}_resume.pt';path.parent.mkdir(exist_ok=True);torch.save(saved,path)
            tail.replay();expected_model=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
            restored=torch.load(path,weights_only=False);model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer'])
            projection=restored['projection'];cap.projection.steps=projection['steps']
            cap.projection.clipped.copy_(torch.tensor(projection['clipped_steps_per_group'],device='cuda'))
            cap.projection.last_scales.copy_(torch.tensor(projection['last_scales'],device='cuda'))
            train.restore_rng(restored['rng']);tail.replay()
            equal(model.state_dict(),expected_model);equal(opt.state_dict(),expected_opt)
            assert cap.projection.steps==2
            # Main graph remains usable after an eager tail and checkpoint restore.
            cap.replay();assert cap.projection.steps==3
            row=dict(arm=arm,objective=checked,initial_state_sha256=digest,batch64=True,tail32=True,
                capture_no_updates=True,checkpoint_tail_replay_exact=True,main_graph_after_tail=True,
                disposable_updates_executed=4,capture_seconds=capture_seconds,first_step_seconds=step_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                loss=float(values['total']),gradient_norm=float(norm),weights=layer_weights(arm))
            result['arms'].append(row);save(HERE/'preflight_progress.json',result);print(json.dumps(row),flush=True)
            del tail,cap,opt,model,pack,sample,values,norm,saved,restored,expected_model,expected_opt,projection
            gc.collect();torch.cuda.empty_cache()
        assert 562*64+32==plan['train_count']
        result.update(status='passed',ctrl_c_resume_verified=True,initial_state_sha256=digest,
            same_five_train_members=True,formal_models_discarded=True,sources=sources())
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)

if __name__=='__main__':main()
