"""Five existing cached train members; verify fixed supervision and one disposable update."""
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
from .loss import per_case
from .loss_config import ARMS, COEFFICIENTS, A, B, layer_weights, reference_weights
from .simple_loss import components
from .cuda_step import CapturedStep

def main():
    started=time.perf_counter(); arm=ARMS[0]
    result=dict(status='error', arm=arm, unique_train_cases=5, dev_payloads=0, test_read=False,
        optimizer_updates=0, formal_training_started=False, resume_tested=False)
    try:
        for path in OWNER.rglob('*.py'): ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        train.seed_all(0); model=GeometryModel(**train.MODEL)
        digests={k:hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                 for k,v in model.state_dict().items()}
        assert digests==plan['initial_tensor_sha256']
        assert hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()==plan['initial_cpu_rng_sha256']
        count=sum(p.numel() for p in model.parameters())
        assert count==plan['parameters'] and all(p.requires_grad for p in model.parameters())
        assert len(model.initial_blocks)==2 and len(model.layers)==12
        assert model.head[0].in_features==model.head[0].out_features==64
        assert model.head[2].in_features==64 and model.head[2].out_features==3
        builder=static_tensor_cache._build
        def forbidden(_): raise AssertionError('Must reuse existing CPU cache')
        static_tensor_cache._build=forbidden
        try: rows=load_rows('train',plan['preflight_train_indices'])
        finally: static_tensor_cache._build=builder
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cuda',max_nodes=120)
        batch=pack.batch([i%5 for i in range(32)]); public=inputs_only(batch)
        assert not {'target','reference','pin_coordinates'}&set(public)
        assert (public['mask'][public['valid']]==1).all() and (public['constraints'][...,2]==0).all()
        model.cuda().eval()
        with torch.no_grad():
            output=model(public)
            assert output.shape==(12,32,120,3) and torch.isfinite(output).all()
            weights=layer_weights(arm); refs=reference_weights(arm)
            assert weights=={4:.2,12:.8} and refs=={int(k):v for k,v in plan['reference_weights'].items()}
            assert abs(sum(weights[i]*refs[i] for i in weights)-.4)<1e-12
            raw={i:components(output[i-1],batch,{'overlap':A,'task':B}) for i in weights}
            # Independent scalar expansion, separate from the candidate Loss wrapper.
            expected=sum(weights[i]*sum(COEFFICIENTS[k]*raw[i][k]*(refs[i] if k in ('position','shape') else 1.)
                for k in COEFFICIENTS) for i in weights)
            original=sum(COEFFICIENTS[k]*raw[12][k] for k in COEFFICIENTS)
            values=per_case(output,batch,arm)
            torch.testing.assert_close(values['total'],expected,rtol=0,atol=0)
            torch.testing.assert_close(values['final_total'],original,rtol=0,atol=0)
            torch.testing.assert_close(values['total'],values['weighted_reference']+values['weighted_overlap']+values['weighted_task'],rtol=2e-6,atol=2e-6)
            for key in COEFFICIENTS: torch.testing.assert_close(values['final_'+key],raw[12][key],rtol=0,atol=0)
            result['initial_eval_diagnostics']={k:float(values[k].mean()) for k in
                ('total','final_total','final_own_total','layer04_total','layer12_total','weighted_reference')}
        del output,values,expected,original,raw
        model.train(); optimizer=train.optimizer_for(model); torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,arm)
        before=model.head[0].weight.detach().clone()
        try:
            values,norm=cap.replay(); torch.cuda.synchronize()
        except RuntimeError:
            result['rejected_before_Adam']=dict(total=float(cap.values['total']),max_case=float(cap.max_case),
                gradient_norm=float(cap.norm),finite=bool(cap.finite),projection_steps=cap.projection.steps,
                optimizer_state_entries=len(optimizer.state))
            raise
        assert cap.projection.steps==1 and not torch.equal(before,model.head[0].weight)
        assert all(layer.qkv.weight.grad.abs().sum()>0 for layer in model.layers)
        assert all(block.qkv.weight.grad.abs().sum()>0 for block in model.initial_blocks)
        if arm=='node12':
            assert all(layer.geometry_q.weight.grad.abs().sum()>0 and layer.geometry_k.weight.grad.abs().sum()>0
                and layer.geometry_scale.grad.abs().sum()>0 for layer in model.layers)
        assert not (OWNER/'runs').exists()
        result.update(status='passed',parameters=count,common_initial_tensors=len(digests),
            parent_initial_tensors_exact=True,global_CPU_RNG_preserved=True,existing_CPU_cache_reused=True,
            members=[list(r['member']) for r in rows], output_layers=12,initial_graph_blocks=2,
            exact_objective_verified=True,common_final_original_loss_verified=True,
            all12_blocks_receive_gradients=True,reference_weights=refs,layer_weights=weights,
            optimizer_updates=1,disposable_update_only=True,total=float(values['total']),max_case_loss=float(cap.max_case),
            gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),sources=sources())
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc()); raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
