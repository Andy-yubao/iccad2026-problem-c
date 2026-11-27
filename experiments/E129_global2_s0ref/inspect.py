"""Five cached train members, S0 indexing, exact objective and one disposable update."""
import ast
import hashlib
import json
import time
import traceback
import torch
from datasets import static_tensor_cache
from . import train
from .io import OWNER, save, sources
from .model import GeometryModel
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case, reference_only
from .loss_config import ARMS, COEFFICIENTS, A, B, layer_weights
from .simple_loss import components
from .cuda_step import CapturedStep

def main():
    started=time.perf_counter();arm=ARMS[0]
    result=dict(status='error',arm=arm,unique_train_cases=5,dev_payloads=0,test_read=False,
        optimizer_updates=0,formal_training_started=False,resume_tested=False)
    try:
        for p in OWNER.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        train.seed_all(0);model=GeometryModel(**train.MODEL)
        digest={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest() for k,v in model.state_dict().items()}
        assert digest==plan['initial_tensor_sha256']
        assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()==plan['initial_cpu_rng_sha256']
        count=sum(p.numel() for p in model.parameters());assert count==plan['parameters']
        assert all(p.requires_grad for p in model.parameters())
        assert len(model.initial_blocks)==2 and len(model.layers)==12
        builder=static_tensor_cache._build
        def forbidden(_):raise AssertionError('Must reuse existing CPU cache')
        static_tensor_cache._build=forbidden
        try:rows=load_rows('train',plan['preflight_train_indices'])
        finally:static_tensor_cache._build=builder
        assert len({tuple(r['member']) for r in rows})==5
        batch=ResidentRows(rows,'cuda',max_nodes=120).batch([i%5 for i in range(32)])
        public=inputs_only(batch)
        assert not {'target','reference','pin_coordinates'}&set(public)
        assert (public['mask'][public['valid']]==1).all() and (public['constraints'][...,2]==0).all()
        model.cuda().eval();readouts=[]
        handle=model.head.register_forward_hook(lambda m,a,o:readouts.append(o.detach().clone()))
        with torch.no_grad():
            output=model(public);assert output.shape==(13,32,120,3) and torch.isfinite(output).all()
            assert len(readouts)==13
            for i in range(13):assert torch.equal(output[i],readouts[i])
            weights=layer_weights(arm);assert weights=={0:.1,4:.1,12:.8}
            raw={i:components(output[i],batch,{'overlap':A,'task':B}) for i in (0,4,12)}
            initial=reference_only(output[0],batch)
            torch.testing.assert_close(initial['reference'],raw[0]['reference'],rtol=0,atol=0)
            assert set(initial)=={'position','shape','reference'}
            original4=sum(COEFFICIENTS[k]*raw[4][k] for k in COEFFICIENTS)
            final8=sum(COEFFICIENTS[k]*raw[12][k]*(.8 if k in ('position','shape') else 1.) for k in COEFFICIENTS)
            expected=.1*raw[0]['reference']+.1*original4+.8*final8
            values=per_case(output,batch,arm)
            torch.testing.assert_close(values['total'],expected,rtol=0,atol=0)
            torch.testing.assert_close(values['layer04_total'],original4,rtol=0,atol=0)
            torch.testing.assert_close(values['layer00_total'],raw[0]['reference'],rtol=0,atol=0)
            torch.testing.assert_close(values['final_total'],sum(COEFFICIENTS[k]*raw[12][k] for k in COEFFICIENTS),rtol=0,atol=0)
            torch.testing.assert_close(values['total'],values['weighted_reference']+values['weighted_overlap']+values['weighted_task'],rtol=2e-6,atol=2e-6)
            result['initial_eval_diagnostics']={k:float(values[k].mean()) for k in ('total','initial_reference','layer04_total','layer12_total','final_total')}
        handle.remove();del readouts,output,raw,initial,values,original4,final8,expected
        model.train();optimizer=train.optimizer_for(model);torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,arm);before=model.head[0].weight.detach().clone()
        try:
            values,norm=cap.replay();torch.cuda.synchronize()
        except RuntimeError:
            result['rejected_before_Adam']=dict(total=float(cap.values['total']),max_case=float(cap.max_case),
                gradient_norm=float(cap.norm),finite=bool(cap.finite),projection_steps=cap.projection.steps,
                optimizer_state_entries=len(optimizer.state))
            raise
        assert cap.projection.steps==1 and not torch.equal(before,model.head[0].weight)
        assert all(layer.qkv.weight.grad.abs().sum()>0 for layer in model.layers)
        assert all(block.qkv.weight.grad.abs().sum()>0 for block in model.initial_blocks)
        if arm=='node12':assert all(layer.geometry_q.weight.grad.abs().sum()>0 and layer.geometry_k.weight.grad.abs().sum()>0 for layer in model.layers)
        assert not (OWNER/'runs').exists()
        result.update(status='passed',parameters=count,common_initial_tensors=len(digest),parent_initial_tensors_exact=True,
            global_CPU_RNG_preserved=True,existing_CPU_cache_reused=True,members=[list(r['member']) for r in rows],
            output_states=13,geometry_blocks=12,initial_graph_blocks=2,S0_exact_readout_verified=True,
            initial_reference_only_verified=True,exact_objective_verified=True,common_final_original_loss_verified=True,
            all12_blocks_receive_gradients=True,layer_weights=weights,reference_weights={0:1.,4:1.,12:.8},
            optimizer_updates=1,disposable_update_only=True,total=float(values['total']),max_case_loss=float(cap.max_case),
            gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),sources=sources())
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
