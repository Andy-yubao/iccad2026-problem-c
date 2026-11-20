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
    p=torch.tensor([[[2.,2.,2.,2.],[3.,2.5,2.,2.],[7.,6.,1.,1.],[80.,80.,1.,1.]]],dtype=torch.double)
    rel=torch.zeros((1,4,4),dtype=torch.double);rel[:,0,2]=rel[:,2,0]=1
    constraints=torch.zeros((1,4,5),dtype=torch.long);constraints[:,0,4]=9
    g=dict(area=p[...,2:].prod(-1),valid=torch.tensor([[True,True,True,False]]),group=rel,constraints=constraints)
    z=module_features(p,g);assert z.shape==(1,4,49) and len(FEATURE_NAMES)==49 and not z.requires_grad
    assert torch.count_nonzero(z[:,3])==0 and bool(torch.isfinite(z).all())
    torch.testing.assert_close(z[0,0,5],torch.tensor(2.,dtype=torch.double).log())
    torch.testing.assert_close(z[0,0,6],torch.tensor(.375,dtype=torch.double).log1p())
    torch.testing.assert_close(z[0,0,8:12],torch.tensor([.5,1.5,1.25,.75],dtype=torch.double).asinh())
    assert z[0,0,12]<0 and z[0,0,13]>0
    assert z[0,0,20]==1 and z[0,0,25]==1 and z[0,0,29]==0 and z[0,0,33]==0
    assert z[0,1,20]==0 and torch.count_nonzero(z[0,1,20:38])==0
    assert z[0,0,46]==1
    for factor in (.1,10.):
        gg=dict(g,area=g['area']*factor**2)
        torch.testing.assert_close(module_features(p*factor,gg),z,rtol=1e-6,atol=1e-7)
    translated=p+torch.tensor([4.2,-.7,0.,0.],dtype=torch.double)
    torch.testing.assert_close(module_features(translated,g),z,rtol=1e-6,atol=1e-7)
    perm=torch.tensor([2,3,0,1]);gg=dict(area=g['area'][:,perm],valid=g['valid'][:,perm],
        group=g['group'][:,perm][:,:,perm],constraints=g['constraints'][:,perm])
    torch.testing.assert_close(module_features(p[:,perm],gg),z[:,perm],rtol=0,atol=0)
    swap=p[...,[1,0,3,2]];c=constraints.clone();bits=constraints[...,4]
    c[...,4]=((bits&1)!=0).long()*8+((bits&2)!=0).long()*4+((bits&4)!=0).long()*2+((bits&8)!=0).long()
    zs=module_features(swap,dict(g,constraints=c));expected=z.clone()
    expected[...,:5]=z[...,[1,0,3,2,4]];expected[...,4]=-expected[...,4]
    for start in (8,12,16):expected[...,start:start+4]=z[...,[start+3,start+2,start+1,start]]
    for start in (22,26,30):expected[...,start:start+2]=z[...,[start+1,start]]
    expected[...,34:38]=z[...,[35,34,37,36]]
    expected[...,38:42]=z[...,[41,40,39,38]];expected[...,42:46]=z[...,[45,44,43,42]]
    expected[...,47:49]=z[...,[48,47]]
    torch.testing.assert_close(zs,expected,rtol=1e-6,atol=1e-7)
    singleton={k:(v[:,:1,:1] if k=='group' else v[:,:1]) for k,v in g.items()}
    one=module_features(p[:,:1],singleton);assert bool(torch.isfinite(one).all()) and one[0,0,20]==0
    empty=dict(g,valid=torch.zeros_like(g['valid']));assert torch.count_nonzero(module_features(p,empty))==0
    return dict(hand_overlap_and_direction_values=True,group_candidates_and_masks=True,
        translation_scale_axis_invariance=True,node_permutation=True,singleton_and_empty_finite=True,
        padding_zero=True,geometry_detached=True,corner_flags=True)

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

def main():
    started=time.perf_counter();result=dict(status='preparing',formal_training_started=False,dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0)
    try:
        assert not (HERE/'runs').exists() and not read(HERE/'authorization.json')['training_enabled']
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        result['synthetic']=synthetic();print('Synthetic49-feature semantics passed',flush=True)
        plan=read(HERE/'plan.json');result['cache']=cache_gate(plan)
        engine=ROOT/'experiments/E141_path_supervision';gate=read(engine/'preflight.json')
        assert gate['status']=='passed' and gate['ctrl_c_resume_verified']
        checked=[]
        for name in ('cuda_step.py','captured_projection.py','step_control.py','resident.py','schedule.py','train.py'):
            expected=(engine/name).read_text().replace('experiments.E141_path_supervision','experiments.E142_dual_geometry').replace('E141.path_supervision.v1.seed0','E142.dual_geometry.v1.seed0').replace('E141','E142')
            assert (HERE/name).read_text()==expected,name
            checked.append(name)
        result['inherited_engine']=dict(parent_preflight_sha256=sha(engine/'preflight.json'),checked_files=checked,new_signal_simulations=0)
        train.seed_all(0);m=GeometryModel(**train.MODEL)
        initial_cpu=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in m.state_dict().values())).hexdigest()
        assert sum(p.numel() for p in m.parameters())==plan['parameters']['e142']==895123
        assert all(p.requires_grad for p in m.parameters());del m
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
        for _ in range(3):
            t=time.perf_counter();values,norm=cap.replay();torch.cuda.synchronize();times.append(time.perf_counter()-t)
        for name,p in model.named_parameters():
            if any(s in name for s in ('demand_encoder','fuse','edge.','receiver','sender','relation_value')):
                assert p.grad is not None and bool(p.grad.abs().sum()>0),'Missing new-branch gradient '+name
        tail=EagerStep(model,opt,pack.batch([i%5 for i in range(32)]),ARMS[0],cap.projection)
        saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=copy.deepcopy(opt.state_dict()),rng=train.rng_state(),projection=copy.deepcopy(cap.projection.summary()))
        path=HERE/'preparation_diagnostics/resume.pt';path.parent.mkdir(exist_ok=True);torch.save(saved,path)
        tail.replay();expected_model=copy.deepcopy(model.state_dict());expected_opt=copy.deepcopy(opt.state_dict())
        restored=torch.load(path,weights_only=False);model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer'])
        projection=restored['projection'];cap.projection.steps=projection['steps']
        cap.projection.clipped.copy_(torch.tensor(projection['clipped_steps_per_group'],device='cuda'))
        cap.projection.last_scales.copy_(torch.tensor(projection['last_scales'],device='cuda'))
        train.restore_rng(restored['rng']);tail.replay();equal(model.state_dict(),expected_model);equal(opt.state_dict(),expected_opt)
        cap.replay();assert cap.projection.steps==5
        result.update(status='passed',ctrl_c_resume_verified=True,initial_state_sha256=initial_cpu,parameters=895123,
            all_new_branch_gradients=True,batch64=True,tail32=True,capture_without_updates=True,checkpoint_tail_replay_exact=True,
            main_graph_after_tail=True,disposable_updates_executed=6,formal_model_not_reused=True,capture_seconds=capture_seconds,
            step_seconds=times,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            last_short_loss=float(values['total']),last_gradient_norm=float(norm),sources=sources())
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','cache','inherited_engine','members')},ensure_ascii=False),flush=True)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)

if __name__=='__main__':main()
