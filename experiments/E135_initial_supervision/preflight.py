"""Five cached train members only: objective, S0 observation and exact safe resume."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import gc
import hashlib
import json
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
from .loss import per_case,base_per_case,frontend_components,public_hard_components,PUBLIC_HARD
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
        dataset=RandomReference(role);chunks={}
        for index in dataset.indices:chunks.setdefault(index//256,[]).append(index)
        for block,ids in chunks.items():
            path=base/dataset.role/f'{block:04d}.pt'
            entry=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if not path.is_file() or entry['identity']!=key:raise ValueError('Missing cache chunk')
            for i in ids:
                if entry['members'][str(i)]!=dataset.records[i]['member']:raise ValueError('Cache member mismatch')
        report[role]=dict(members=len(dataset.indices),chunks=len(chunks),metadata_coverage_complete=True,tensor_payloads_read=False)
    return dict(key=key,roles=report)

def frontend_check():
    state=torch.zeros((2,4,3),requires_grad=True)
    target=torch.zeros_like(state)
    area=torch.ones((2,4));area[0,1]=4.
    valid=torch.zeros((2,4),dtype=torch.bool);valid[0,:3]=True
    constraints=torch.zeros((2,4,5),dtype=torch.long)
    constraints[0,1,0]=1;constraints[0,2,1]=1
    specs=torch.zeros((2,4,4));specs[0,1]=torch.tensor([0.,0.,2.,2.]);specs[0,2]=torch.tensor([4.5,5.5,1.,1.])
    with torch.no_grad():
        state[0,0]=torch.tensor([.25,-.25,2.])
        state[0,1]=torch.tensor([2.,1.,0.])
        state[0,2]=torch.tensor([5.,6.,0.]);state[0,3]=100.
        target[0,1,:2]=1.;target[0,2]=1000.
    g=dict(target=target,area=area,valid=valid,constraints=constraints,specs=specs)
    values=frontend_components(state,g)
    torch.testing.assert_close(values['center_tolerance'],torch.tensor([.05,0.]))
    torch.testing.assert_close(values['free_shape'],torch.tensor([4.,0.]))
    torch.testing.assert_close(values['total'],torch.tensor([.25,0.]))
    hard=public_hard_components(state,{k:g[k] for k in PUBLIC_HARD})
    for name in hard:torch.testing.assert_close(values[name],hard[name],rtol=0,atol=0)
    values['total'].sum().backward()
    if not torch.equal(state.grad[0,0,:2],torch.zeros(2)):raise AssertionError('Deadband has nonzero gradient')
    if not torch.equal(state.grad[0,2],torch.zeros(3)):raise AssertionError('P reference leaked into frontend')
    if not torch.equal(state.grad[0,3],torch.zeros(3)) or state.grad[1].abs().sum()!=0:raise AssertionError('Padding contributes')
    torch.testing.assert_close(state.grad[0,0,2],torch.tensor(.2))
    return dict(deadband_and_gradient=True,subset_means=True,empty_and_padding_zero=True,
        public_FP_terms_identical=True,no_P_reference=True,free_shape_coefficient=.05)

def main():
    started=time.perf_counter()
    result=dict(status='preparing',formal_training_started=False,unique_train_cases=5,dev_payloads=0,
                test_payloads=0,postprocessing_calls=0,Q_calls=0,ctrl_c_resume_verified=False)
    try:
        for path in HERE.rglob('*.py'):
            if 'preparation_diagnostics' not in path.parts:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
        if auth['training_enabled'] or auth['allowed_arms'] or (HERE/'runs').exists():raise AssertionError('Formal launch during preparation')
        baseline=ROOT/'experiments/E134_supervision_interval/runs/e119_stride3_seed0/checkpoints/best.pt'
        if sha(baseline)!=plan['baseline']['checkpoint_sha256']:raise AssertionError('Existing control changed')
        result['cache']=cache_coverage(plan)
        result['frontend_objective']=frontend_check()
        rows=load_rows('train',plan['preflight_train_indices'])
        if len({tuple(r['member']) for r in rows})!=5:raise AssertionError('Unexpected members')
        result['members']=[list(r['member']) for r in rows]
        arm_results=[];initial_digest=None
        for arm in ARMS:
            train.seed_all(0);model=GeometryModel(**train.MODEL)
            state=model.state_dict()
            digest=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in state.values())).hexdigest()
            if initial_digest is not None and initial_digest!=digest:raise AssertionError('Initialization differs across arms')
            initial_digest=digest
            count=sum(p.numel() for p in model.parameters())
            if count!=plan['parameters']['e119']:raise AssertionError('Parameter count changed')
            from experiments.E134_supervision_interval.e119_model import GeometryModel as Parent
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0);parent=Parent(**train.MODEL)
            if not all(torch.equal(t,parent.state_dict()[k]) for k,t in state.items()):raise AssertionError('Parent initialization differs')
            pack=ResidentRows(rows,'cuda',max_nodes=120)
            batch=pack.batch([i%5 for i in range(32)]);public=inputs_only(batch)
            if set(public)&{'reference','target','pin_coordinates'}:raise AssertionError('Label input leak')
            if not (public['mask'][public['valid']]==1).all() or not (public['constraints'][...,2]==0).all():raise AssertionError('Prediction/MIB boundary changed')
            model.cuda().eval();parent.cuda().eval()
            with torch.inference_mode():
                old=parent(public);output=model(public)
                if not torch.equal(old,output[1:]):raise AssertionError('Observing S0 changed S1..S12')
                actual=per_case(output,batch,arm)['total'];weights=layer_weights(arm)
                expected=sum(v*(frontend_components(output[k],batch)['total'] if arm=='e119_C' and k==0 else base_per_case(output[k],batch)['total']) for k,v in weights.items())
                # C adds S0 last to avoid calculating unused full S0 terms.
                torch.testing.assert_close(actual,expected,rtol=2e-7,atol=1e-7)
                if abs(sum(weights.values())-1)>1e-12 or weights[12]!=.8:raise AssertionError('Incorrect total weights')
                if arm=='e119_B':
                    if abs(weights[9]/weights[0]-2)>1e-12:raise AssertionError('B S9 not twice S0')
                    deltas=[weights[b]-weights[a] for a,b in ((0,3),(3,6),(6,9))]
                    if max(deltas)-min(deltas)>1e-12:raise AssertionError('B not linear')
                if {str(k):v for k,v in weights.items()}!=plan['arms'][arm]['layer_weights']:raise AssertionError('Plan/code weights differ')
            # Verify identical dropout RNG consumption and refinements in training mode too.
            model.train();parent.train();rng=train.rng_state()
            with torch.no_grad():
                old=parent(public);train.restore_rng(rng);new=model(public)
            if not torch.equal(old,new[1:]):raise AssertionError('S0 observation changes dropout path')
            del parent,old,new
            optimizer=train.optimizer_for(model)
            before={k:t.detach().clone() for k,t in model.state_dict().items()}
            cap=CapturedStep(model,optimizer,batch,arm)
            if any(not torch.equal(v,before[k]) for k,v in model.state_dict().items()):raise AssertionError('Capture updated parameters')
            values,norm=cap.replay();torch.cuda.synchronize()
            if cap.projection.steps!=1:raise AssertionError('Guarded update not committed')
            if torch.equal(model.head[0].weight,before['head.0.weight']):raise AssertionError('No update')
            for layer in (model.initial_blocks[0],model.layers[0],model.layers[-1]):
                if not layer.qkv.weight.grad.abs().sum()>0:raise AssertionError('Missing frontend/backbone gradient')
            arm_results.append(dict(arm=arm,parameters=count,initial_state_sha256=digest,weights=weights,
                disposable_optimizer_updates=1,total=float(values['total']),gradient_norm=float(norm),
                parent_initialization_exact=True,S1_to_S12_bitwise_identical_eval_and_dropout=True,all_prediction_channels=True))
            print(arm_results[-1],flush=True)
            del cap,optimizer,model,pack,batch,public,output,actual,expected,before,state,values,norm
            gc.collect();torch.cuda.empty_cache()
        result['arms']=arm_results;result['resume_checks']=[]
        for arm in ARMS:
            result['resume_checks'].append(compare_resume(arm))
            print('Exact SIGINT checkpoint/resume passed:',arm,flush=True)
        for mode,name in (('--queue-controller-test','queue'),('--evaluation-controller-test','evaluation')):
            child=subprocess.run([sys.executable,'-u','-B','-m','experiments.E135_initial_supervision.selftest',mode],
                cwd=ROOT,env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8'),capture_output=True,text=True)
            if child.returncode:
                print(child.stdout);print(child.stderr);raise RuntimeError(name+' continuation failed')
            result[name+'_signal']=json.loads((HERE/f'preparation_diagnostics/{name}_signal_result.json').read_text())
        if (HERE/'runs').exists():raise AssertionError('Preparation created formal runs')
        result.update(status='passed',ctrl_c_resume_verified=True,sources=sources(),
                      total_disposable_optimizer_updates=21,formal_optimizer_updates=0)
        save(HERE/'preparation_complete.json',dict(status='ready_to_start',arms=list(ARMS),
            ctrl_c_checkpoint_resume_verified=True,automatic_postprocessing_prepared=True,
            real_dev_payloads=0,test_payloads=0,real_postprocessing_calls=0))
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-started;save(HERE/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','members','resume_checks','cache','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
