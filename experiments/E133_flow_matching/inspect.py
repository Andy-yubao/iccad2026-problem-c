"""Five original train cases only: FM mathematics, model gradients, graph RNG."""
import ast
import json
import time
import traceback
from unittest.mock import patch
import torch
from .control import OWNER,ROOT,read,sha,sources
from .io import save
from . import data,runtime
from .resident import ResidentRows
from .model import build_model,relation_descriptor
from .flow import path,objective,integrate,fixed_bank,training_statistics
from .batches import inputs_only,decode
from .addresses import attach_addresses
from .cuda_step import ProtectedStep,augmented

def main():
    started=time.perf_counter();result=dict(status='error',unique_train_cases=5,optimizer_updates=0,
        dev_tensor_payloads=0,test_read=False,raw_payload_reads=0,postprocessing_calls=0,resume_tested=False)
    try:
        for p in OWNER.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
        assert not read(OWNER/'authorization.json')['training_enabled'] and (OWNER/'STOP').exists()
        inventory=data.cache_inventory();plan=read(OWNER/'plan.json');ids=plan['preflight_train_indices']
        rows=data.load_rows('train',ids)
        historical=read(ROOT/'history/20261111_experiments_consolidation/experiments/E116_first_draft_comparison/preflight.json')
        assert [list(r['member']) for r in rows]==historical['train_members']
        result['members']=[list(r['member']) for r in rows];result['cache_inventory']=inventory
        runtime.seed_all(0);model=build_model().eval();count=sum(p.numel() for p in model.parameters())
        assert count==plan['parameters_by_arm']['fm']
        batch=ResidentRows(rows,'cpu').batch(range(5));attach_addresses(batch,rows,range(5))
        stats=training_statistics(rows)
        model.state_mean.copy_(torch.tensor(stats['mean']));model.state_std.copy_(torch.tensor(stats['std']))
        z1=model.standardize(batch['target']);valid=batch['valid'];z0,t=fixed_bank(rows,120,device='cpu')
        zt,target=path(z0,z1,t);torch.testing.assert_close(target,z1-z0)
        out=model(inputs_only(batch),zt,t);values=objective(out,target,valid)
        manual=torch.stack([(out[i,:r['graph']['n']]-target[i,:r['graph']['n']]).square().mean() for i,r in enumerate(rows)])
        torch.testing.assert_close(values['total'],manual)
        values['total'].mean().backward()
        assert model.state_input[0].weight.grad.abs().sum()>0 and model.time_input[0].weight.grad.abs().sum()>0
        assert all(l.qkv.weight.grad.abs().sum()>0 and l.modulation[1].weight.grad.abs().sum()>0 for l in model.layers)
        hard=valid&((batch['constraints'][...,0]!=0)|(batch['constraints'][...,1]!=0))
        output_grad=torch.autograd.grad(objective(out,target,valid)['total'].sum(),out)[0]
        assert output_grad[hard].abs().sum()>0
        try:relation_descriptor(dict(inputs_only(batch),reference=batch['reference']))
        except ValueError:pass
        else:raise AssertionError('Clean reference entered public encoder')
        swapped=augmented(dict(batch,axis_swap=torch.ones(5,dtype=torch.bool)))
        expected=torch.stack((z1[...,1],z1[...,0],-z1[...,2]),-1)
        torch.testing.assert_close(model.standardize(swapped['target']),expected)
        # A constant straight-path oracle must reach the prescribed endpoint.
        with patch.object(model,'velocity',lambda c,z,t,v:target*v[...,None]):
            end=integrate(model,batch,z0,8)
        torch.testing.assert_close(end[valid],batch['target'][valid],rtol=2e-5,atol=2e-6)
        model.zero_grad(set_to_none=True)
        assert torch.equal(fixed_bank(rows,120,device='cpu')[0],z0)
        result.update(parameters=count,CPU_loss=float(values['total'].mean().detach()),
            path_target_and_Euler_oracle=True,axis_symmetric_affine=True,labels_excluded_from_condition=True,
            all_F_P_channels_supervised=True,time_and_state_and_all_blocks_gradients=True)
        print('CPU FM path, objective, conditioning and Euler checks passed',flush=True)
        gpu=ResidentRows(rows,'cuda').batch([i%5 for i in range(32)])
        attach_addresses(gpu,rows,[i%5 for i in range(32)])
        runtime.seed_all(0);model=build_model().cuda().train()
        model.state_mean.copy_(torch.tensor(stats['mean'],device='cuda'));model.state_std.copy_(torch.tensor(stats['std'],device='cuda'))
        optimizer=runtime.optimizer_for(model)
        before={k:v.clone() for k,v in model.state_dict().items()}
        with patch.object(torch.optim.AdamW,'step',side_effect=AssertionError('No preparation Adam update')):
            cap=ProtectedStep(model,optimizer,gpu)
            values,norm=cap.backward_only();first=cap.z0.clone();first_t=cap.t.clone()
            cap.backward_only();torch.cuda.synchronize()
            assert not torch.equal(first,cap.z0) and not torch.equal(first_t,cap.t)
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
            assert not optimizer.state and cap.projection.steps==0
        result.update(CUDA_training_graph=False,eager_FP32=True,batch_size=32,CUDA_training_randomness_refreshes=True,
            CUDA_loss=float(values['total']),max_case=float(cap.max_case),raw_grad_norm=float(norm),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),parameters_unchanged=True,
            torch=str(torch.__version__),GPU=torch.cuda.get_device_name(0))
        del cap,optimizer,model,before
        torch.cuda.empty_cache()
        # Verify the actual deployment encoder and Euler8 graph on one of the
        # same five members; reuse its existing NPZ, never another member.
        from datasets.random_reference16k import RandomReference16k
        from .evaluation.run import public_graph,Predictor
        raw=RandomReference16k('train')[ids[0]]
        assert list(raw['member'])==list(rows[0]['member'])
        public=public_graph(raw['solver_inputs'])
        cached=ResidentRows(rows[:1],'cpu').batch([0]);attach_addresses(cached,rows[:1],[0])
        for key,value in public.items():
            torch.testing.assert_close(value,cached[key],rtol=0,atol=0)
        inference_model=build_model();inference_model.state_mean.copy_(torch.tensor(stats['mean']))
        inference_model.state_std.copy_(torch.tensor(stats['std']))
        with torch.inference_mode():
            predictor=Predictor(dict(model=inference_model.state_dict()),raw['solver_inputs'])
            a=predictor.predict(raw['solver_inputs'],list(raw['member']))
            b=predictor.predict(raw['solver_inputs'],list(raw['member']))
            assert (a==b).all()
            eager=decode(integrate(predictor.model,predictor.g,predictor.noise,8),predictor.g)
            torch.testing.assert_close(predictor.output,eager,rtol=2e-5,atol=2e-6)
        result.update(status='passed',raw_payload_reads=1,deployment_public_graph_matches_cache=True,
            Euler8_inference_capture_matches_eager=True,member_seeded_inference_repeatable=True,sources=sources())
    except BaseException as exc:
        result.update(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
