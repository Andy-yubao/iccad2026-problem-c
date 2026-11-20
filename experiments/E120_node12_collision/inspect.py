"""Necessary Node12 math/capture check on the same five cached training members."""
import ast
import json
import math
import time
import traceback
import torch
from datasets import static_tensor_cache
from . import train
from .io import OWNER,ROOT,save,sources
from .model import GeometryModel,node_geometry,AXIS_SWAP
from .e119_model import GeometryModel as E119Model
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case,base_per_case
from .cuda_step import CapturedStep
from .augmentation import axis_swap

def check_geometry(public,state):
    phi=node_geometry(state,public)
    assert phi.shape==(*state.shape[:2],12) and torch.isfinite(phi).all()
    assert (phi[~public['valid']]==0).all()
    swapped_state=torch.stack((state[...,1],state[...,0],-state[...,2]),-1)
    # Node12 uses only area and valid from g; both are invariant under axis swap.
    actual=node_geometry(swapped_state,public)
    torch.testing.assert_close(actual,phi[...,list(AXIS_SWAP)],rtol=3e-5,atol=3e-6)
    # Use one of the same train members' public areas for exact nested-square checks.
    g={k:v[:1].clone() for k,v in public.items()}
    valid_ids=torch.nonzero(g['valid'][0]).flatten()
    order=valid_ids[g['area'][0,valid_ids].argsort()]
    ids=torch.stack((order[0],order[len(order)//2],order[-1]))
    g['valid'].zero_();g['valid'][0,ids]=True
    centered=torch.zeros_like(state[:1]);centered[...,:2]=2.
    value=node_geometry(centered,g)[0,ids]
    areas=g['area'][0,ids].double()
    ratios=torch.minimum(areas[:,None],areas[None,:])/areas[:,None]
    ratios.fill_diagonal_(0.)
    expected=torch.stack((torch.ones(3,device=state.device),ratios.sum(-1).log1p(),ratios.amax(-1),
        torch.full((3,),math.log(2),device=state.device)),1).to(state.dtype)
    torch.testing.assert_close(value[:,4:8],expected,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(value[:,8:],torch.full_like(value[:,8:],math.log(1.5)),rtol=3e-5,atol=3e-6)
    separate=centered.clone()
    for i,index in enumerate(ids):separate[0,index,:2]=2.+6.*i
    zero=node_geometry(separate,g)
    assert (zero[...,4:]==0).all()
    singleton={**g,'valid':torch.zeros_like(g['valid'])}
    singleton['valid'][0,ids[0]]=True
    assert (node_geometry(centered,singleton)[...,4:]==0).all()
    differentiable=state.detach().clone().requires_grad_(True)
    node_geometry(differentiable,public)[...,5:].sum().backward()
    assert torch.isfinite(differentiable.grad).all() and differentiable.grad.abs().sum()>0

def main():
    started=time.perf_counter()
    result=dict(status='error',unique_train_cases=5,dev_payloads=0,test_read=False,
        optimizer_updates=0,formal_training_started=False,resume_tested=False)
    try:
        for path in OWNER.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        train.seed_all(0);reference=E119Model(**train.MODEL);reference_rng=torch.get_rng_state().clone()
        train.seed_all(0);model=GeometryModel(**train.MODEL)
        assert torch.equal(reference_rng,torch.get_rng_state())
        common=[k for k in reference.state_dict() if '.bias.' not in k]
        for key in common:assert torch.equal(reference.state_dict()[key],model.state_dict()[key]),key
        assert len(model.initial_blocks)==2 and len(model.layers)==12
        assert all(not hasattr(layer,'bias') for layer in model.layers)
        count=sum(p.numel() for p in model.parameters())
        assert count==plan['parameters'] and all(p.requires_grad for p in model.parameters())
        del reference
        original_builder=static_tensor_cache._build
        def forbidden(_):raise AssertionError('Existing cached members must not be preprocessed again')
        static_tensor_cache._build=forbidden
        try:rows=load_rows('train',plan['preflight_train_indices'])
        finally:static_tensor_cache._build=original_builder
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cuda',max_nodes=120);batch=pack.batch([i%5 for i in range(32)])
        public=inputs_only(batch)
        assert not {'target','reference','pin_coordinates'}&set(public)
        assert (public['mask'][public['valid']]==1).all() and (public['constraints'][...,2]==0).all()
        model.cuda().eval()
        with torch.no_grad():
            h=model.input(public['features'])*public['valid'][...,None]
            for block in model.initial_blocks:h=block(h,public)
            state=model.head(h)
        check_geometry(public,state)
        with torch.no_grad():
            output=model(public);assert output.shape==(12,32,120,3) and torch.isfinite(output).all()
            values=per_case(output,batch,'node12')
            torch.testing.assert_close(values['total'],.2*base_per_case(output[3],batch)['total']+.8*base_per_case(output[11],batch)['total'],rtol=0,atol=0)
        del output,values
        model.train();optimizer=train.optimizer_for(model);torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,'node12')
        before=model.head[0].weight.detach().clone();values,norm=cap.replay();torch.cuda.synchronize()
        assert cap.projection.steps==1 and not torch.equal(before,model.head[0].weight)
        for layer in model.layers:
            assert layer.geometry_q.weight.grad.abs().sum()>0
            assert layer.geometry_k.weight.grad.abs().sum()>0
            assert layer.geometry_scale.grad.abs().sum()>0
        for block in model.initial_blocks:assert block.qkv.weight.grad.abs().sum()>0
        assert not (OWNER/'runs').exists()
        result.update(status='passed',parameters=count,common_initial_tensor_count=len(common),
            compatible_E119_initial_tensors_exact=True,global_CPU_RNG_preserved=True,
            cached_rows_reused_without_preprocessing=True,members=[list(r['member']) for r in rows],
            containment_true_intersection_verified=True,four_way_symmetric_pressure_verified=True,
            no_overlap_and_singleton_verified=True,axis_swap_verified=True,continuous_feature_gradients_verified=True,
            node12_QK_gradients_all12=True,E119_initial_generator_gradients=True,
            optimizer_updates=1,disposable_update_only=True,total=float(values['total']),max_case_loss=float(cap.max_case),
            gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),sources=sources())
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
