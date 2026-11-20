"""Five-member input/model gate, cache identity and bounded actual resume proofs."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ast
import copy
import hashlib
import json
import subprocess
import sys
import time
import traceback
import torch
from .io import HERE, ROOT, save, sha, sources
from . import train
from .model import GeometryModel, inputs_only, public_descriptor, PUBLIC
from .data import load_rows, cache_identity
from .resident import ResidentRows
from .addresses import attach_addresses, address_rows
from .loss import per_case, base_per_case
from .cuda_step import CapturedStep
from .selftest import compare_resume


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def cache_coverage(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if key!=plan['required_cache_key']:raise ValueError('Shared cache identity changed')
    base=ROOT/'datasets/static_tensor_cache_v1'/key
    if read(base/'identity.json')!=identity:raise ValueError('Cache producer mismatch')
    report={}
    for role in ('train','dev_monitor500'):
        dataset=RandomReference(role);blocks={}
        for index in dataset.indices:blocks.setdefault(index//256,[]).append(index)
        for block,ids in blocks.items():
            path=base/dataset.role/f'{block:04d}.pt';entry=read(path.with_suffix('.json'))
            if not path.is_file() or entry['identity']!=key or sha(path)!=entry['sha256']:
                raise ValueError('Missing or corrupt cached block')
            if any(entry['members'][str(i)]!=dataset.records[i]['member'] for i in ids):
                raise ValueError('Cached member changed')
        report[role]=dict(members=len(dataset.indices),chunks=len(blocks),all_chunk_bytes_hashed=True)
    return dict(key=key,roles=report)


def independent_checks(model,batch):
    model.eval();g=inputs_only(batch)
    with torch.no_grad():
        output=model(g);own,rel,background,_,_=public_descriptor(g)
        assert own.shape[-1]==20 and rel.shape[-1]==640 and background.shape[-1]==8
        # Independently rebuild destination lookup and scalar features in Python.
        for row in range(g['valid'].shape[0]):
            n=int(g['valid'][row].sum());addresses=g['address'][row,:n].tolist()
            assert len(set(addresses))==n and all(0<=a<128 for a in addresses)
            for i in range(n):
                assert float(rel[row,i,addresses[i]])==1
                for j in range(n):
                    w=g['wire_b2b'][row,i,j]
                    torch.testing.assert_close(rel[row,i,384+addresses[j]],w.log1p())
                    c=g['constraints'][row,:,3]
                    expected=float(i!=j and c[i]>0 and c[i]==c[j])
                    assert float(rel[row,i,512+addresses[j]])==expected
            torch.testing.assert_close(own[row,:n,0],g['area'][row,:n])
        actual=per_case(output,batch)['total'];expected=base_per_case(output,batch)['total']
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        try:model(dict(g,target=batch['target']))
        except ValueError:pass
        else:raise AssertionError('Label extra field was accepted')
        changed={k:v.clone() for k,v in batch.items()};changed['target'].fill_(100);changed['reference'].fill_(-100)
        torch.testing.assert_close(output,model(inputs_only(changed)),rtol=0,atol=0)
        perm=torch.arange(g['valid'].shape[1]-1,-1,-1)
        moved={}
        for key,value in g.items():
            moved[key]=value[:,perm][:,:,perm] if key=='wire_b2b' else (value if key=='scale' else value[:,perm])
        torch.testing.assert_close(model(moved),output[:,perm],rtol=2e-5,atol=2e-6)
        empty={k:v.clone() for k,v in g.items()}
        empty['pin_valid'].zero_();empty['pin_weights'].zero_();empty['wire_b2b'].zero_();empty['constraints'][...,3]=0
        assert torch.isfinite(model(empty)).all()
        assert (output[~g['valid']]==0).all()
        cpu={k:v.cpu() for k,v in g.items()}
        gpu_model=copy.deepcopy(model).cuda().eval()
        gpu_output=gpu_model({k:v.cuda() for k,v in cpu.items()}).cpu()
        torch.testing.assert_close(output.cpu(),gpu_output,rtol=3e-4,atol=3e-5)
        from .augmentation import axis_swap
        swap_batch={k:v.clone() for k,v in batch.items()};swap_batch['axis_swap']=torch.ones(len(g['valid']),dtype=torch.bool)
        swapped=axis_swap(swap_batch);twice=axis_swap(swapped)
        for key in batch:torch.testing.assert_close(twice[key],batch[key],rtol=0,atol=0)
        assert torch.isfinite(model(inputs_only(swapped))).all()
    return dict(label_isolation=True,independent_relation_reconstruction=True,node_permutation_equivariance=True,
        empty_sets_and_padding=True,cpu_cuda_close=True,axis_swap_involution=True,final_loss_exact=True)


def main():
    start=time.perf_counter();result=dict(status='preparing',formal_training_started=False,
        formal_optimizer_updates=0,unique_train_cases=5,dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0,
        ctrl_c_resume_verified=False)
    try:
        for path in HERE.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        auth=read(HERE/'authorization.json');plan=read(HERE/'plan.json')
        if auth['training_enabled'] or auth['allowed_arms'] or (HERE/'runs').exists():raise ValueError('Formal activation during preparation')
        for name in ('STOP','QUEUE_STOP','PAUSE_REQUEST'):
            if (HERE/name).exists():raise RuntimeError('Stop present: '+name)
        result['cache']=cache_coverage(plan)
        print('Checked complete frozen train36000/dev500 cache hashes and members.',flush=True)
        for control in plan['comparison_controls']:
            if sha(ROOT/control['result'])!=control['results_sha256']:raise ValueError('Saved control changed')
        parent=ROOT/'experiments/E136_frontend_F_linear'
        for name in ('features.py','batches.py','ports.py','simple_loss.py','group_tree.py','step_control.py','captured_projection.py'):
            if sha(HERE/name)!=sha(parent/name):raise ValueError('Copied numerical primitive changed: '+name)
        def functions(path,names):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
        names={'Objective','public_hard_components','base_per_case','masked_channels_mean'}
        if functions(HERE/'loss.py',names)!=functions(parent/'loss.py',names):raise ValueError('Final objective changed')
        train.seed_all(0);model=GeometryModel();assert sum(p.numel() for p in model.parameters())==1335363
        result['initial_state_sha256']=hashlib.sha256(b''.join(t.numpy().tobytes() for t in model.state_dict().values())).hexdigest()
        rows=load_rows('train',plan['preflight_train_indices'])
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cpu',max_nodes=120,max_pins=145)
        batch=pack.batch(range(5));attach_addresses(batch,rows,range(5),0,3)
        result['input_checks']=independent_checks(model,batch)
        # Verify raw public inference construction on precisely these five members.
        from datasets.random_reference import RandomReference
        from .evaluation.run import public_graph, Predictor, loss_components
        ds=RandomReference('train')
        for i,index in enumerate(plan['preflight_train_indices']):
            item=ds[index];reference=public_graph(item['solver_inputs'])
            expected=ResidentRows([rows[i]],'cpu',max_nodes=120,max_pins=145).batch([0])
            attach_addresses(expected,[rows[i]],[0])
            for key in PUBLIC:torch.testing.assert_close(reference[key],expected[key],rtol=0,atol=0)
        result['raw_to_cached_public_input_exact']=True
        gpu_pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145)
        gpu_batch=gpu_pack.batch([i%5 for i in range(32)]);attach_addresses(gpu_batch,rows,[i%5 for i in range(32)],0,0)
        model.cuda().train();optimizer=train.optimizer_for(model)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        cap=CapturedStep(model,optimizer,gpu_batch,'standard256')
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        values,norm=cap.replay();torch.cuda.synchronize()
        for module in (model.port_encoder,model.input_encoder,model.block.qkv,model.block.ff,model.head):
            assert sum(float(p.grad.abs().sum()) for p in module.parameters())>0
        for name,module in model.named_modules():
            params=list(module.named_parameters(recurse=False))
            if not params:continue
            base=sum(float(before[f'{name}.{key}' if name else key].square().sum()) for key,_ in params)**.5
            shift=sum(float((p-before[f'{name}.{key}' if name else key]).square().sum()) for key,p in params)**.5
            assert shift<=.005*max(base,1e-8)*(1+1e-4)
        result.update(disposable_optimizer_updates=1,disposable_loss=float(values['total']),gradient_norm=float(norm),
            capture_preserves_parameters=True,all_branches_have_gradients=True,relative_update_bound_verified=True,
            parameters=1335363,peak_allocated_bytes=torch.cuda.max_memory_allocated(),execution_path='FP32 CUDA graph')
        # Frozen training-member-only checkpoint verifies the actual evaluator input/capture/decode and Loss path.
        scratch=HERE/'preparation_diagnostics/inference.pt'
        scratch.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(model=model.state_dict(),model_config={'architecture':'standard256'}),scratch)
        with torch.inference_mode():
            predictor=Predictor(scratch,item['solver_inputs']);draft=predictor.predict(item['solver_inputs'])
            raw_loss,final_loss=loss_components(predictor.raw_state,draft,item['solver_inputs'],item)
            assert all(torch.isfinite(torch.tensor(list(value.values()))).all() for value in (raw_loss,final_loss))
        result['real_inference_path_checked_on_train_member']=True
        del cap,optimizer,predictor,model,gpu_batch,gpu_pack
        torch.cuda.empty_cache()
        print('Input/model/capture/one-step gate passed; checking disposable pause/resume and controller.',flush=True)
        result['resume_checks']=[compare_resume('standard256')]
        for mode in ('queue','evaluation'):
            completed=subprocess.run([sys.executable,'-u','-B','-m','experiments.E137_standard_transformer256.selftest',f'--{mode}-controller-test'],
                cwd=ROOT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),capture_output=True,text=True)
            log=HERE/'preparation_diagnostics'/f'{mode}_controller.log';log.write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
            if completed.returncode:raise RuntimeError('Controller short check failed: '+mode)
            result[mode+'_signal']=read(HERE/f'preparation_diagnostics/{mode}_signal_result.json')
        result.update(status='passed',ctrl_c_resume_verified=True,sources=sources(),training_model_not_reused=True,
            disposable_updates_in_resume_checks=6)
        save(HERE/'preparation_complete.json',dict(status='ready_to_start',arm='standard256',formal_updates=0,
            unique_train_cases=5,dev_payloads=0,test_payloads=0,automatic_postprocessing_prepared=True))
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-start;save(HERE/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','cache','traceback','resume_checks')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
