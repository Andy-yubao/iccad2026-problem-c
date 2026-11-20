"""Five cached train members: exact S recurrence and one guarded real update."""
import ast
import json
import time
import traceback
import torch
from datasets import static_tensor_cache
from . import train
from .io import OWNER,save,sources
from .model import GeometryModel
from .reference_model import GeometryModel as E119Model
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case,base_per_case
from .cuda_step import CapturedStep

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
        common=list(reference.state_dict())
        for key,value in reference.state_dict().items():assert torch.equal(value,model.state_dict()[key]),key
        count=sum(p.numel() for p in model.parameters());assert count==plan['parameters']
        assert len(model.initial_blocks)==2 and len(model.layers)==len(model.delta_heads)==12
        assert model.layers[0].bias[0].in_features==18
        assert all(h.in_features==64 and h.out_features==3 for h in model.delta_heads)
        del reference
        original=static_tensor_cache._build
        def forbidden(_):raise AssertionError('Existing cache must be reused')
        static_tensor_cache._build=forbidden
        try:rows=load_rows('train',plan['preflight_train_indices'])
        finally:static_tensor_cache._build=original
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cuda',max_nodes=120);batch=pack.batch([i%5 for i in range(32)])
        public=inputs_only(batch)
        assert (public['mask'][public['valid']]==1).all() and (public['constraints'][...,2]==0).all()
        model.cuda().eval();initial=[];deltas=[]
        handles=[model.head.register_forward_hook(lambda m,a,o:initial.append(o.detach().clone()))]
        handles.extend(head.register_forward_hook(lambda m,a,o:deltas.append(o.detach().clone())) for head in model.delta_heads)
        with torch.no_grad():out=model(public)
        for handle in handles:handle.remove()
        assert len(initial)==1 and len(deltas)==12 and out.shape==(12,32,120,3)
        state=initial[0]
        for i,delta in enumerate(deltas):
            state=state+delta;assert torch.equal(state,out[i])
        assert torch.isfinite(out).all()
        with torch.no_grad():
            values=per_case(out,batch,'residual')
            torch.testing.assert_close(values['total'],.2*base_per_case(out[3],batch)['total']+.8*base_per_case(out[11],batch)['total'],rtol=0,atol=0)
            result['initial_eval_diagnostics']=dict(total=float(values['total'].mean()),max_case=float(values['total'].amax()),
                max_state=float(out.abs().amax()),layer4_total=float(values['layer04_total'].mean()),layer12_total=float(values['layer12_total'].mean()))
        del out,initial,deltas,values,state
        model.train();optimizer=train.optimizer_for(model);torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,'residual')
        before=model.delta_heads[-1].weight.detach().clone()
        try:
            values,norm=cap.replay();torch.cuda.synchronize()
        except RuntimeError:
            result['rejected_before_Adam']=dict(total=float(cap.values['total']),max_case=float(cap.max_case),
                gradient_norm=float(cap.norm),output_max=float(cap.output_max),finite=bool(cap.finite),
                projection_steps=cap.projection.steps,optimizer_state_entries=len(optimizer.state))
            raise
        assert cap.projection.steps==1 and not torch.equal(before,model.delta_heads[-1].weight)
        assert model.head[0].weight.grad.abs().sum()>0
        assert all(head.weight.grad.abs().sum()>0 for head in model.delta_heads)
        assert all(block.qkv.weight.grad.abs().sum()>0 for block in model.initial_blocks)
        assert not (OWNER/'runs').exists()
        result.update(status='passed',parameters=count,common_initial_tensors=len(common),
            compatible_E119_initial_tensors_exact=True,global_CPU_RNG_preserved=True,
            cached_rows_reused_without_preprocessing=True,members=[list(r['member']) for r in rows],
            original_head_calls=1,independent_delta_heads=12,exact_state_recurrence_verified=True,eta=1,
            initial_head_and_all_delta_heads_receive_gradients=True,
            optimizer_updates=1,disposable_update_only=True,total=float(values['total']),
            max_case_loss=float(cap.max_case),gradient_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),sources=sources())
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
