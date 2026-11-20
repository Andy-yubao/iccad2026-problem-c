"""Five training fixtures: exact union, gradient/update and pre-Adam abort."""
import ast
import json
import time
import traceback
import numpy as np
import torch
from .io import HERE,save,sources,checkpoint
from .data import load_rows,register_resident
from .resident import ResidentRows
from .runtime import parent_model
from .model import GeometryModel
from .batches import inputs_only
from .geometry import measures
from .loss import per_case
from .loss_config import COEFFICIENTS
from .augmentation import axis_swap
from .guards import safe,LossExplosion
from .metrics import union_area,module_overlap
from .train import seed_all,MODEL,optimizer_for,build_capture

def cpu_region(p):
    xs=np.unique(np.r_[p[:,0],p[:,0]+p[:,2]]);ys=np.unique(np.r_[p[:,1],p[:,1]+p[:,3]])
    xm=(xs[1:]+xs[:-1])/2;ym=(ys[1:]+ys[:-1])/2
    depth=np.zeros((len(xm),len(ym)),dtype=np.int16)
    for x,y,w,h in p:
        depth+=((xm>=x)&(xm<x+w))[:,None]&((ym>=y)&(ym<y+h))[None,:]
    return float((np.diff(xs)[:,None]*np.diff(ys)[None,:]*(depth>1)).sum())

def main():
    start=time.perf_counter();result=dict(status='error',unique_train_cases=5,test_read=False,dev_payloads=0)
    try:
        for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
        seed_all(0);plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        rows=load_rows('train',plan['preflight_train_indices'])
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=max(146,max(r['graph']['pin_coordinates'].shape[1] for r in rows)))
        register_resident(pack);batch=pack.batch([i%5 for i in range(32)])
        parent=parent_model()
        model=GeometryModel(**MODEL).cuda();model.load_state_dict(parent.state_dict(),strict=True)
        assert sum(p.numel() for p in model.parameters())==3490819
        assert COEFFICIENTS['position']==.1 and COEFFICIENTS['shape']==.025
        with torch.no_grad():state=parent(inputs_only(batch));geom=measures(state,batch)
        max_error=0.
        for i,row in enumerate(rows):
            n=row['graph']['n'];p=geom['positions'][i,:n].cpu().double().numpy()
            union=union_area(p);covered=np.asarray(module_overlap(p)['covered_fractions'])
            actual=geom['covered_fraction'][i,:n].cpu().numpy()
            error=float(np.max(np.abs(covered-actual)));max_error=max(max_error,error)
            assert error<5e-5,error
            assert abs(float(geom['union_area'][i])-union)<5e-6
            total=float(batch['area'][i,:n].sum())
            assert abs(float(geom['overlap_region_fraction'][i])-cpu_region(p)/total)<5e-5
        batch['axis_swap']=torch.ones(32,device='cuda',dtype=torch.bool)
        swapped=axis_swap(batch);restored=axis_swap(swapped)
        for key in batch:torch.testing.assert_close(batch[key],restored[key],rtol=0,atol=0)
        swapped_state=torch.stack((state[...,1],state[...,0],-state[...,2]),-1)
        with torch.no_grad():
            old=per_case(state,batch,state,state);new=per_case(swapped_state,swapped,swapped_state,swapped_state)
        for key in old:torch.testing.assert_close(old[key],new[key],rtol=5e-5,atol=5e-5)
        optimizer=optimizer_for(model);cap=build_capture(model,optimizer,pack)
        cap.inputs['axis_swap'].copy_(torch.arange(32,device='cuda')%2==0)
        before=model.layers[0].local.weight.detach().clone()
        values,norm=cap.replay()
        assert not torch.equal(before,model.layers[0].local.weight)
        assert cap.projection.steps==1 and all(int(v['step'])==1 for v in optimizer.state.values())
        assert all(p.grad is None and not p.requires_grad for p in cap.parent.parameters())
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        # Exercise the exact pre-Adam gate without another optimizer update.
        accepted=model.layers[0].local.weight.detach().clone();cap.ok.zero_()
        try:cap.step_if_safe()
        except LossExplosion:pass
        else:raise AssertionError('Injected unsafe batch accepted')
        assert torch.equal(accepted,model.layers[0].local.weight) and cap.projection.steps==1
        assert all(int(v['step'])==1 for v in optimizer.state.values())
        assert not bool(safe(torch.tensor([float('nan')],device='cuda'),torch.ones(1,device='cuda')))
        assert not bool(safe(torch.full((2,),11.,device='cuda'),torch.ones(2,device='cuda')))
        assert not bool(safe(torch.tensor([22.,0.,0.,0.,0.],device='cuda'),torch.ones(5,device='cuda')))
        from .predictor import Predictor
        from .evaluate import request
        from .depth_evaluation.repair.canonical_native import warmup
        from threadpoolctl import threadpool_limits
        import numba
        numba.set_num_threads(1)
        fixture=HERE/'runs/shortcheck_checkpoint.pt';checkpoint(fixture,dict(model=model.state_dict(),model_config=MODEL))
        with threadpool_limits(limits=1),torch.inference_mode():
            warmup();predictor=Predictor(fixture,rows[0]['case'].solver_inputs);record=request(predictor,rows[0])
        fixture.unlink()
        result.update(status='passed',sources=sources(),actual_updates=1,
            parameters=3490819,max_coverage_error=max_error,axis_swap_twice_exact=True,loss_axis_symmetric=True,
            reference_coefficients=dict(position=.1,shape=.025),loss=float(values['total']),max_case=float(cap.max_case),
            gradient_norm=float(norm),parent_mean=float(cap.parent_total.mean()),
            strict_abort_before_adam_verified=True,nan_and_relative_spike_rejected=True,
            parent_frozen=True,full_evaluation_hard_legal=record['metrics']['is_feasible'],postprocessing_calls=1,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            elapsed_seconds=time.perf_counter()-start)
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.perf_counter()-start)
        raise
    finally:
        save(HERE/'preflight.json',result);print({k:v for k,v in result.items() if k!='sources'},flush=True)

if __name__=='__main__':main()
