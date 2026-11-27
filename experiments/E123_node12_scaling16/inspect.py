"""Five cached train members,16 outputs, exact objective, one disposable update."""
import ast
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
from .loss import per_case, base_per_case
from .loss_config import ARMS, layer_weights
from .cuda_step import CapturedStep

def main():
    started=time.perf_counter();arm=ARMS[0]
    result=dict(status='error',arm=arm,unique_train_cases=5,dev_payloads=0,test_read=False,
                optimizer_updates=0,formal_training_started=False,resume_tested=False)
    try:
        for path in OWNER.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        if arm=='global2':
            from .original_model import GeometryModel as Reference
            reference_config={k:v for k,v in train.MODEL.items() if k!='initial_layers'}
        elif arm=='node12':
            from .e119_model import GeometryModel as Reference
            reference_config=train.MODEL
        else:
            from .reference_model import GeometryModel as Reference
            reference_config=train.MODEL
        train.seed_all(0);reference=Reference(**reference_config)
        reference_rng=torch.get_rng_state().clone()
        train.seed_all(0);model=GeometryModel(**train.MODEL)
        assert torch.equal(reference_rng,torch.get_rng_state())
        common=[k for k in reference.state_dict() if arm!='node12' or '.bias.' not in k]
        for key in common:
            assert torch.equal(reference.state_dict()[key],model.state_dict()[key]),key
        count=sum(p.numel() for p in model.parameters())
        assert count==plan['parameters'] and all(p.requires_grad for p in model.parameters())
        assert len(model.initial_blocks)==2 and len(model.layers)==16
        assert model.head[0].in_features==model.head[0].out_features==64
        assert model.head[2].in_features==64 and model.head[2].out_features==3
        if arm=='residual':
            assert len(model.delta_heads)==16
            assert all(h.in_features==64 and h.out_features==3 for h in model.delta_heads)
        if arm=='node12':
            assert all(not hasattr(layer,'bias') for layer in model.layers)
            assert all(layer.geometry_q.in_features==layer.geometry_k.in_features==12 for layer in model.layers)
        else:
            assert all(layer.bias[0].in_features==18 for layer in model.layers)
        del reference
        builder=static_tensor_cache._build
        def forbidden(_):raise AssertionError('Must reuse existing CPU cache')
        static_tensor_cache._build=forbidden
        try:rows=load_rows('train',plan['preflight_train_indices'])
        finally:static_tensor_cache._build=builder
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cuda',max_nodes=120)
        batch=pack.batch([i%5 for i in range(32)]);public=inputs_only(batch)
        assert not {'target','reference','pin_coordinates'}&set(public)
        assert (public['mask'][public['valid']]==1).all()
        assert (public['constraints'][...,2]==0).all()
        model.cuda().eval();readouts=[];deltas=[]
        handles=[model.head.register_forward_hook(lambda m,a,o:readouts.append(o.detach().clone()))]
        if arm=='residual':
            handles += [head.register_forward_hook(lambda m,a,o:deltas.append(o.detach().clone())) for head in model.delta_heads]
        with torch.no_grad():
            output=model(public)
            assert output.shape==(16,32,120,3) and torch.isfinite(output).all()
            if arm=='residual':
                assert len(readouts)==1 and len(deltas)==16
                state=readouts[0]
                for i,delta in enumerate(deltas):
                    state=state+delta;assert torch.equal(state,output[i])
                del state
            else:
                assert len(readouts)==17
                for i in range(16):assert torch.equal(readouts[i+1],output[i])
            weights=layer_weights(arm)
            assert weights=={2:.1,8:.1,16:.8}
            expected=sum(w*base_per_case(output[i-1],batch)['total'] for i,w in weights.items())
            values=per_case(output,batch,arm)
            torch.testing.assert_close(values['total'],expected,rtol=0,atol=0)
            torch.testing.assert_close(values['final_total'],base_per_case(output[15],batch)['total'],rtol=0,atol=0)
            result['initial_eval_diagnostics']={k:float(values[k].mean()) for k in ('total','layer02_total','layer08_total','layer16_total')}
        for handle in handles:handle.remove()
        del output,readouts,deltas,values,expected
        model.train();optimizer=train.optimizer_for(model);torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,arm)
        before=model.head[0].weight.detach().clone()
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
        if arm=='node12':
            assert all(layer.geometry_q.weight.grad.abs().sum()>0 and layer.geometry_k.weight.grad.abs().sum()>0
                       and layer.geometry_scale.grad.abs().sum()>0 for layer in model.layers)
        if arm=='residual':assert all(head.weight.grad.abs().sum()>0 for head in model.delta_heads)
        assert not (OWNER/'runs').exists()
        result.update(status='passed',parameters=count,common_initial_tensors=len(common),
            matched_depth16_initial_tensors_exact=True,global_CPU_RNG_preserved=True,
            existing_CPU_cache_reused=True,members=[list(r['member']) for r in rows],
            output_layers=16,initial_graph_blocks=2,preserved_head='64->64->3',
            exact_state_readout_or_recurrence_verified=True,all16_blocks_receive_gradients=True,
            residual_eta=1 if arm=='residual' else None,
            fixed_objective_weights=weights,optimizer_updates=1,disposable_update_only=True,
            total=float(values['total']),max_case_loss=float(cap.max_case),gradient_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),sources=sources())
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
