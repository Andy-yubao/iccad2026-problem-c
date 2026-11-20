"""Disposable frozen-parent, two-S0, loss, augmentation and resume checks."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast,copy,gc,io,json,time,traceback
import numpy as np
import torch
from . import train
from .io import HERE,ROOT,save,sources
from .data import load_rows
from .resident import ResidentRows
from .public import inputs_only
from .model import GeometryModel
from .initialization import load_initial,verify_initial
from .loss import per_case,layer_loss,overlap_total,A
from .base_loss import base_per_case
from .cuda_step import CapturedStep,EagerStep
from .step_control import GraphStepControl
from .augmentation import axis_swap
from .perturbation import perturb
from .monitor_audit import audit

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

def new_model(parent):
    model=GeometryModel().cuda();model.initial_generator.load_state_dict(parent);return model

def main():
    started=time.perf_counter();report=dict(status='preparing',formal_training_updates=0,test_payloads=0,dev_payloads=0)
    try:
        assert not json.loads((HERE/'authorization.json').read_text())['training_enabled'] and not (HERE/'runs').exists()
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        plan=json.loads((HERE/'plan.json').read_text());train.seed_all(0);parent=load_initial()
        rows=load_rows('train',plan['preflight_train_indices']);assert len(rows)==5
        profile=json.loads((ROOT/'datasets/random_mib40k_v1/cache/profile.json').read_text())
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=profile['max_pins'],max_ports=profile['max_ports'])
        batch=pack.batch(range(5));g=inputs_only(batch);model=new_model(parent).eval()
        assert sum(p.numel() for p in model.parameters())==plan['parameters']['e156_s0_backend']
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==plan['trainable_parameters']['e156_s0_backend']
        with torch.no_grad():
            out=model(g);again=model(g);equal(out,again)
            clean=model.initial_generator(g)[0];equal(out[0],clean)
            assert out.shape==(5,5,120,3)
            for draw in (.5,.05,.2):
                altered=perturb(clean,g,torch.full((5,),draw,device='cuda'))
                equal(altered[...,2],clean[...,2])
                if draw==.5:equal(altered,clean)
                if draw==.05:assert bool(((altered[...,:2]-clean[...,:2]).abs()<=.10001*g['area'].clamp_min(1e-12).sqrt()[...,None]).all())
                if draw==.2:
                    for j in range(5):
                        n=int(g['valid'][j].sum());distance=(altered[j,:n,None,:2]-clean[j,None,:n,:2]).abs().sum(-1)
                        assert float(distance.min(-1).values.max())<1e-4
                        assert int(((altered[j,:n,:2]-clean[j,:n,:2]).abs().sum(-1)>0).sum())<=2
            try:model(dict(g,target=batch['target']))
            except ValueError:pass
            else:raise AssertionError('Labels accepted as public inputs')
            values=per_case(out,batch,'e156_s0_backend');r2=layer_loss(out[2],batch);r4=layer_loss(out[4],batch)
            expected=.2*r2['total']+.8*r4['total']+A*(r4['overlap_total']-overlap_total(out[0],batch)).clamp_min(0)
            torch.testing.assert_close(values['total'],expected,rtol=0,atol=0)
            torch.testing.assert_close(overlap_total(out[0],batch),base_per_case(out[0],batch)['overlap_total'],rtol=1e-6,atol=1e-6)
            full=layer_loss(out[4],batch,reference=1.)
            torch.testing.assert_close(full['total']-r4['total'],.25*full['reference_weighted'],rtol=2e-5,atol=1e-6)
            equal(r2['overlap_weighted'],torch.zeros_like(r2['overlap_weighted']))
            zero_guard=out.clone();zero_guard[4]=zero_guard[0]
            assert not per_case(zero_guard,batch,'e156_s0_backend')['relative_overlap_weighted'].any()
            swapped=axis_swap(dict(batch,axis_swap=torch.ones(5,device='cuda',dtype=torch.bool)))
            transposed=torch.stack((out[...,1],out[...,0],-out[...,2]),-1)
            torch.testing.assert_close(per_case(transposed,swapped,'e156_s0_backend')['total'],values['total'],rtol=2e-5,atol=1e-4)
            diagnostics=audit(out,batch,values);assert all(torch.isfinite(v).all() for v in diagnostics.values())
        model.train();assert not model.initial_generator.training
        optimizer=train.optimizer_for(model);before=cpu_state(model)
        cap=CapturedStep(model,optimizer,pack.batch([i%5 for i in range(64)]),'e156_s0_backend')
        equal(before,cpu_state(model));assert not optimizer.state
        cap.replay();cap.replay();verify_initial(model,parent)
        for prefix in ('problem_encoder','s0.layers.0','s0.layers.1','s0.group_move','s0.member_move','s0.move_gate'):
            grads=[p.grad for k,p in model.named_parameters() if k.startswith(prefix) and p.requires_grad]
            assert grads and all(v is not None and torch.isfinite(v).all() for v in grads) and any(v.abs().sum()>0 for v in grads),prefix
        tail=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),'e156_s0_backend',cap.projection)
        tail.replay();verify_initial(model,parent)
        buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=train.rng_state(),projection=cap.projection.summary()),buffer)
        buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False)
        train.restore_rng(saved['rng']);tail.replay();expected=cpu_state(model);expected_opt=copy.deepcopy(optimizer.state_dict());expected_projection=copy.deepcopy(cap.projection.summary())
        del cap,tail,optimizer,model;gc.collect();torch.cuda.empty_cache()
        model=new_model(parent).train();model.load_state_dict(saved['model']);optimizer=train.optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
        control=GraphStepControl(model);control.steps=saved['projection']['steps'];control.clipped.copy_(torch.tensor(saved['projection']['clipped_steps_per_group'],device='cuda'));control.last_scales.copy_(torch.tensor(saved['projection']['last_scales'],device='cuda'))
        resumed=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),'e156_s0_backend',control)
        train.restore_rng(saved['rng']);resumed.replay();equal(expected,cpu_state(model));equal(expected_opt,optimizer.state_dict());equal(expected_projection,control.summary());verify_initial(model,parent)
        report.update(status='passed',sources=sources(),parent_best51_frozen=True,eval_clean_and_deterministic=True,exact_loss_sum=True,absolute_overlap_absent=True,relative_guard_exact=True,perturbation_modes_checked=True,axis_transpose=True,all_S0_branches_have_gradients=True,capture64_and_tail32=True,serialized_next_update_bitwise=True,diagnostics_finite=True,postprocessing_calls=0,formal_weights_reused=False,elapsed_seconds=time.perf_counter()-started,peak_reserved_bytes=torch.cuda.max_memory_reserved(),members=[list(r['member']) for r in rows])
        print('[PREFLIGHT] frozen best51, shared S0, Loss, perturbation, GPU64/tail32 and exact next-update resume passed',flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:save(HERE/'preflight.json',report)

if __name__=='__main__':main()
