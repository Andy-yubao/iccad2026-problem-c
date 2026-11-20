"""Same five cached train members, one disposable guarded update, no dev/test."""
import ast
import json
import time
import traceback
import torch
from datasets import static_tensor_cache
from . import train
from .io import OWNER,ROOT,save,sources
from .model import GeometryModel
from .original_model import GeometryModel as OriginalModel
from experiments.E119_global_initial_generator.model import GeometryModel as BaselineModel
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only
from .loss import base_per_case,per_case
from .loss_config import ARMS,layer_weights
from .cuda_step import CapturedStep

def main():
    started=time.perf_counter();arm=ARMS[0]
    result=dict(status='error',arm=arm,unique_train_cases=5,dev_payloads=0,test_read=False,
                optimizer_updates=0,formal_training_started=False,resume_tested=False)
    try:
        for path in OWNER.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        train.seed_all(0)
        reference=OriginalModel(**{k:v for k,v in train.MODEL.items() if k!='initial_layers'})
        reference_rng=torch.get_rng_state().clone()
        train.seed_all(0);model=GeometryModel(**train.MODEL)
        assert torch.equal(reference_rng,torch.get_rng_state())
        common=list(reference.state_dict())
        for key,value in reference.state_dict().items():assert torch.equal(value,model.state_dict()[key]),key
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0);baseline=BaselineModel(**train.MODEL)
        assert all(torch.equal(value,baseline.state_dict()[key]) for key,value in model.state_dict().items())
        del baseline
        count=sum(p.numel() for p in model.parameters());assert count==plan['parameters']
        assert len(model.layers)==12 and model.layers[0].bias[0].in_features==18
        original_builder=static_tensor_cache._build
        def forbidden(_):raise AssertionError('Preflight attempted raw preprocessing instead of cache reuse')
        static_tensor_cache._build=forbidden
        try:rows=load_rows('train',plan['preflight_train_indices'])
        finally:static_tensor_cache._build=original_builder
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cuda',max_nodes=120)
        batch=pack.batch([i%5 for i in range(32)]);public=inputs_only(batch)
        assert not {'target','reference','pin_coordinates'}&set(public)
        assert (public['mask'][public['valid']]==1).all() and (public['constraints'][...,2]==0).all()
        model.cuda().eval()
        with torch.no_grad():
            output=model(public);assert output.shape==(12,32,120,3)
            expected=sum(weight*base_per_case(output[depth-1],batch)['total'] for depth,weight in layer_weights(arm).items())
            torch.testing.assert_close(per_case(output,batch,arm)['total'],expected,rtol=0,atol=0)
            if arm=='first12':
                reference.cuda().eval();assert torch.equal(reference(public),output)
            else:
                assert len(model.initial_blocks)==2
        del reference,output
        model.train();optimizer=train.optimizer_for(model)
        torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,arm)
        before=model.head[0].weight.detach().clone()
        values,norm=cap.replay();torch.cuda.synchronize()
        assert cap.projection.steps==1 and not torch.equal(before,model.head[0].weight)
        assert model.layers[0].qkv.weight.grad.abs().sum()>0 and model.layers[-1].qkv.weight.grad.abs().sum()>0
        if arm=='global2':
            for block in model.initial_blocks:
                assert block.qkv.weight.grad.abs().sum()>0 and block.local.weight.grad.abs().sum()>0
        assert not (OWNER/'runs').exists()
        result.update(status='passed',parameters=count,common_initial_tensors=len(common),
            common_base_initial_parameters_exact=True,global_CPU_RNG_preserved=True,shared_cache_reused_without_preprocessing=True,
            members=[list(r['member']) for r in rows],optimizer_updates=1,disposable_update_only=True,
            first_last_geometry_blocks_receive_gradients=True,initial_blocks_receive_gradients=arm=='global2',
            fixed_objective_weights=layer_weights(arm),total=float(values['total']),max_case_loss=float(cap.max_case),
            gradient_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),sources=sources())
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
