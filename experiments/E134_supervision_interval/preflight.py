"""Necessary implementation checks; five cached train cases, no formal run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import torch
from . import train
from .io import HERE,ROOT,save,sources,sha
from .model import GeometryModel
from .data import load_rows,cache_identity
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case,base_per_case
from .loss_config import ARMS,layer_weights
from .cuda_step import CapturedStep
from .selftest import compare_resume

def cache_coverage(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if key!=plan['required_cache_key']:raise ValueError('Cache producer identity changed')
    base=ROOT/'datasets/static_tensor_cache_v1'/key
    if json.loads((base/'identity.json').read_text(encoding='utf-8'))!=identity:raise ValueError('Cache identity mismatch')
    report={}
    for role in ('train','dev_monitor500'):
        dataset=RandomReference(role)
        chunks={}
        for index in dataset.indices:chunks.setdefault(index//256,[]).append(index)
        for block,ids in chunks.items():
            path=base/dataset.role/f'{block:04d}.pt'
            entry=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if not path.is_file() or entry['identity']!=key:raise ValueError('Missing/incompatible cache chunk')
            for i in ids:
                if entry['members'][str(i)]!=dataset.records[i]['member']:raise ValueError('Cache member mismatch')
        report[role]=dict(members=len(dataset.indices),chunks=len(chunks),metadata_coverage_complete=True,
                          tensor_payloads_read=False,chunk_sha256_rechecked_on_actual_load=True)
    return dict(key=key,roles=report)

def main():
    started=time.perf_counter()
    result=dict(status='preparing',formal_training_started=False,unique_train_cases=5,
                dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0,ctrl_c_resume_verified=False)
    try:
        for path in HERE.rglob('*.py'):
            if 'preparation_diagnostics' not in path.parts:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
        if auth['training_enabled'] or auth['allowed_arms']:raise AssertionError('Preparation must remain disabled')
        if (HERE/'runs').exists():raise AssertionError('Formal run exists')
        result['cache']=cache_coverage(plan)
        rows=load_rows('train',plan['preflight_train_indices'])
        if len({tuple(r['member']) for r in rows})!=5:raise AssertionError('Unexpected short-check members')
        result['members']=[list(r['member']) for r in rows]
        arm_results=[]
        initial_hashes={}
        for arm in ARMS:
            architecture=arm.split('_')[0]
            train.seed_all(0)
            model=GeometryModel(architecture=architecture,**train.MODEL)
            state=model.state_dict()
            digest=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in state.values())).hexdigest()
            if architecture in initial_hashes and initial_hashes[architecture]!=digest:raise AssertionError('Interval altered initialization')
            initial_hashes[architecture]=digest
            count=sum(p.numel() for p in model.parameters())
            if count!=plan['parameters'][architecture]:raise AssertionError('Parameter count changed')
            if architecture=='e119':
                from experiments.E119_global_initial_generator.model import GeometryModel as Parent
            else:
                from experiments.E120_node12_collision.model import GeometryModel as Parent
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0);parent=Parent(**train.MODEL)
            if not all(torch.equal(t,parent.state_dict()[k]) for k,t in state.items()):raise AssertionError('Copied architecture initialization changed')
            del parent
            pack=ResidentRows(rows,'cuda',max_nodes=120)
            batch=pack.batch([i%5 for i in range(32)])
            public=inputs_only(batch)
            if set(public)&{'reference','target','pin_coordinates'}:raise AssertionError('Label leaked into model inputs')
            if not (public['mask'][public['valid']]==1).all() or not (public['constraints'][...,2]==0).all():raise AssertionError('Prediction/MIB boundary changed')
            model.cuda().eval()
            with torch.inference_mode():
                output=model(public)
                actual=per_case(output,batch,arm)['total']
                weights=layer_weights(arm)
                expected=sum(v*base_per_case(output[k-1],batch)['total'] for k,v in weights.items())
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                if abs(sum(weights.values())-1)>1e-12 or weights[12]!=.8:raise AssertionError('Incorrect weights')
            model.train();optimizer=train.optimizer_for(model)
            before={k:t.detach().clone() for k,t in model.state_dict().items()}
            cap=CapturedStep(model,optimizer,batch,arm)
            if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise AssertionError('Capture updated parameters')
            values,norm=cap.replay();torch.cuda.synchronize()
            if cap.projection.steps!=1:raise AssertionError('Protected update not committed')
            if torch.equal(model.head[0].weight,before['head.0.weight']):raise AssertionError('No optimizer update')
            if not model.layers[0].qkv.weight.grad.abs().sum()>0 or not model.layers[-1].qkv.weight.grad.abs().sum()>0:
                raise AssertionError('Missing backbone gradients')
            arm_results.append(dict(arm=arm,parameters=count,initial_state_sha256=digest,weights=weights,
                disposable_optimizer_updates=1,total=float(values['total']),gradient_norm=float(norm),
                capture_no_parameter_updates=True,full_prediction_channels=True,parent_initialization_exact=True))
            print(arm_results[-1],flush=True)
            del cap,optimizer,model,pack,batch,public,output,actual,expected,before,state
            gc.collect();torch.cuda.empty_cache()
        result['arms']=arm_results
        result['resume_checks']=[]
        for arm in ('e119_stride2','e120_stride2'):
            result['resume_checks'].append(compare_resume(arm))
            print('Checkpoint/resume bitwise equivalence passed:',arm,flush=True)
        # Exercises the real queue/controller SIGINT path with simulated phase workers only.
        child=subprocess.run([sys.executable,'-u','-B','-m','experiments.E134_supervision_interval.selftest','--queue-controller-test'],
                             cwd=ROOT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),capture_output=True,text=True)
        if child.returncode:
            print(child.stdout);print(child.stderr);raise RuntimeError('Queue SIGINT/resume test failed')
        result['queue_signal']=json.loads((HERE/'preparation_diagnostics/queue_signal_result.json').read_text())
        if (HERE/'runs').exists():raise AssertionError('Preparation created formal runs')
        auth=json.loads((HERE/'authorization.json').read_text())
        if auth['training_enabled'] or auth['allowed_arms']:raise AssertionError('Preparation activated training')
        result.update(status='passed',ctrl_c_resume_verified=True,sources=sources(),
                      total_disposable_optimizer_updates=16,formal_optimizer_updates=0)
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started
        save(HERE/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','members','resume_checks','cache','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
