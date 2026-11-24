"""Disposable exact-path preparation checks; only five registered train rows."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast,copy,gc,io,json,time,traceback
from unittest.mock import patch
import numpy as np
import torch
from . import train
from .io import HERE,ROOT,save,sources
from .model import GeometryModel
from .data import load_rows
from .resident import ResidentRows
from .public import inputs_only
from .public_preprocessing import public_batch
from .loss import per_case,mib_loss
from .base_loss import base_per_case
from .cuda_step import CapturedStep,EagerStep
from .step_control import GraphStepControl
from .augmentation import axis_swap

def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b,(a,b)

def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

def main():
    start=time.perf_counter()
    report=dict(status='preparing',formal_training_started=False,formal_training_updates=0,
                dev_payloads=0,test_payloads=0,postprocessing_calls=0,backend_updates=0)
    try:
        auth=json.loads((HERE/'authorization.json').read_text())
        assert auth['preparation_enabled'] and not auth['training_enabled']
        assert not (HERE/'runs/e158_initial_seed0').exists()
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        with patch('sys.argv',['train','--arm','e158_initial']):
            try:train.main()
            except PermissionError:pass
            else:raise AssertionError('Formal training was not blocked')
        report['launch_disabled_guard']=True
        plan=json.loads((HERE/'plan.json').read_text());train.seed_all(0)
        rows=load_rows('train',plan['preflight_train_indices']);assert len(rows)==5
        profile=json.loads((ROOT/'datasets/random_mib40k_v1/cache/profile.json').read_text())
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=profile['max_pins'],max_ports=profile['max_ports'])
        report['dataset_maximum_padding']=profile
        report['members']=[list(row['member']) for row in rows]
        batch=pack.batch(range(5));g=inputs_only(batch)
        for row in rows:
            single=inputs_only(ResidentRows([row],'cuda',max_nodes=120).batch([0]))
            equal(public_batch(row['inputs']),single)
        assert bool((batch['mask'][batch['valid']]==1).all())
        swapped=axis_swap(dict(batch,axis_swap=torch.ones(5,device='cuda',dtype=torch.bool)))
        twice=axis_swap(swapped)
        for key in batch:torch.testing.assert_close(twice[key],batch[key],rtol=0,atol=0)
        train.seed_all(0);model=GeometryModel().cuda().eval()
        count=sum(p.numel() for p in model.parameters())
        assert count==plan['parameters']['e158_initial']==plan['trainable_parameters']['e158_initial']
        assert all(p.requires_grad for p in model.parameters())
        assert len(model.initial_generator.initial_blocks)==3
        assert all(block.ff[0].in_features==64 and block.ff[0].out_features==192 and block.ff[2].out_features==64 for block in model.initial_generator.initial_blocks)
        events=[];hooks=[]
        for index,block in enumerate(model.initial_generator.initial_blocks):
            hooks.append(block.register_forward_hook(lambda module,args,out,index=index:events.append((f'block{index}',list(out.shape)))))
        hooks.append(model.initial_generator.head.register_forward_hook(lambda module,args,out:events.append(('layout_head',list(out.shape)))))
        with torch.no_grad():states=model(g)
        for hook in hooks:hook.remove()
        assert [name for name,_ in events]==['block0','block1','block2','layout_head']
        assert all(shape==[5,120,64] for _,shape in events[:3])
        assert states.shape==(1,5,120,3)
        report['hidden_coordination_path']=events
        try:model(dict(g,target=batch['target']))
        except ValueError:pass
        else:raise AssertionError('Reference accepted as model input')
        values=per_case(states,batch,'e158_initial');base=base_per_case(states[0],batch)
        expected=sum(base[k]*(.25 if k=='overlap_weighted' else 1.) for k in
            ('reference_weighted','overlap_weighted','wire_compact_weighted','grouping_weighted',
             'boundary_weighted','public_shape_weighted','public_preplaced_position_weighted'))+values['mib_weighted']
        torch.testing.assert_close(values['total'],expected,rtol=0,atol=0)
        torch.testing.assert_close(values['overlap_weighted'],.25*base['overlap_weighted'],rtol=0,atol=0)
        torch.testing.assert_close(values['reference_weighted'],base['reference_weighted'],rtol=0,atol=0)
        assert not values['group_bottleneck_weighted'].any()
        assert all(torch.isfinite(v).all() for v in values.values())
        raw=states[0].detach().clone().requires_grad_(True)
        overlap_gradient=torch.autograd.grad(per_case(raw[None],batch,'e158_initial')['overlap_weighted'].mean(),raw)[0]
        assert torch.isfinite(overlap_gradient).all() and overlap_gradient.abs().sum()>0
        zero=states[0].clone();zero[...,2]=0;assert not mib_loss(zero,batch).any()
        altered=dict(g,constraints=g['constraints'].clone())
        mask=altered['constraints'][...,2]>0;altered['constraints'][...,2][mask]+=100
        with torch.no_grad():torch.testing.assert_close(model(altered),states,rtol=0,atol=0)
        perm=torch.arange(119,-1,-1,device='cuda');mp=g['port_valid'].shape[1]
        pp=torch.arange(mp-1,-1,-1,device='cuda');keys=torch.cat((perm,120+pp));pg={}
        for k,v in g.items():
            if k in ('adjacency','mib','group'):pg[k]=v[:,perm][:,:,perm]
            elif k in ('role_hops','role_edges','role_group'):pg[k]=v[:,perm][:,:,keys]
            elif k in ('port_features','port_valid'):pg[k]=v[:,pp]
            elif k=='scale':pg[k]=v
            else:pg[k]=v[:,perm]
        with torch.no_grad():torch.testing.assert_close(model(pg),states[:,:,perm],rtol=2e-4,atol=5e-5)
        transposed=torch.stack((states[...,1],states[...,0],-states[...,2]),-1)
        torch.testing.assert_close(per_case(transposed,swapped,'e158_initial')['total'],values['total'],rtol=2e-5,atol=1e-4)
        report.update(parameters=count,trainable_parameters=count,public_path_exact=True,
            module_port_permutation=True,MIB_id_invariance=True,axis_swap=True,
            quarter_complete_overlap_exact=True,weak_overlap_has_gradient=True,
            reference_one_exact=True,unchanged_terms_sum_exact=True)
        del states,values,base,raw,overlap_gradient,pg,altered,transposed,expected
        model.train();optimizer=train.optimizer_for(model);before=cpu_state(model)
        cap=CapturedStep(model,optimizer,pack.batch([i%5 for i in range(64)]),'e158_initial')
        equal(before,cpu_state(model));assert not optimizer.state
        report['capture_without_updates']=True
        cap.replay();values,norm=cap.replay()
        branches=['initial_generator.input','initial_generator.role_input','initial_generator.port_input','initial_generator.head']
        branches += [f'initial_generator.initial_blocks.{i}.{name}' for i in range(3) for name in ('qkv','ff.0','ff.2')]
        for prefix in branches:
            grads=[p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
            assert grads and all(v is not None and torch.isfinite(v).all() for v in grads)
            assert any(v.abs().sum()>0 for v in grads),prefix
        report['gradient_branches_checked']=branches
        tail=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),'e158_initial',cap.projection)
        tail.replay();cap.replay()
        buffer=io.BytesIO()
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=train.rng_state(),projection=cap.projection.summary()),buffer)
        buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False)
        train.restore_rng(saved['rng']);tail.replay()
        expected=cpu_state(model);expected_opt=copy.deepcopy(optimizer.state_dict());expected_projection=copy.deepcopy(cap.projection.summary())
        # A rejected batch must not change Adam, parameters or accepted-step counters.
        target_before=cap.inputs['target'].clone();cap.inputs['target'].fill_(float('nan'))
        try:cap.replay()
        except RuntimeError as exc:assert 'before AdamW' in str(exc)
        else:raise AssertionError('Nonfinite batch was accepted')
        equal(expected,cpu_state(model));equal(expected_opt,optimizer.state_dict());equal(expected_projection,cap.projection.summary())
        cap.inputs['target'].copy_(target_before)
        report['nonfinite_guard_before_adam']=True
        del cap,tail,optimizer,model,values,target_before;gc.collect();torch.cuda.empty_cache()
        model=GeometryModel().cuda().train();model.load_state_dict(saved['model'])
        optimizer=train.optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
        control=GraphStepControl(model);control.steps=saved['projection']['steps']
        control.clipped.copy_(torch.tensor(saved['projection']['clipped_steps_per_group'],device='cuda'))
        control.last_scales.copy_(torch.tensor(saved['projection']['last_scales'],device='cuda'))
        resumed=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),'e158_initial',control)
        train.restore_rng(saved['rng']);resumed.replay()
        equal(expected,cpu_state(model));equal(expected_opt,optimizer.state_dict());equal(expected_projection,control.summary())
        report.update(status='passed',serialized_next_update_bitwise=True,GPU64_tail32=True,
            disposable_updates_original=5,disposable_updates_resume_replay=1,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),sources=sources(),
            elapsed_seconds=time.perf_counter()-start,formal_weights_reused=False)
        save(HERE/'queue_state.json',dict(status='prepared',active=None,completed=[],
            training_enabled=False,postprocessing_enabled=False))
        print('[PREFLIGHT] E158 passed: three attention/FFN blocks, one head, quarter overlap, GPU64/tail32 and bitwise resume',flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:save(HERE/'preflight.json',report)

if __name__=='__main__':main()
