"""Short five-training-case checks; no formal training/dev/test or repair calls."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast
import gc
import hashlib
import json
import time
import traceback
import torch
from . import train
from .io import HERE,ROOT,save,sources,sha
from .model import GeometryModel
from .batches import inputs_only
from .data import load_rows,cache_identity
from .resident import ResidentRows
from .loss import objective_forward,per_case,base_per_case,coarse_components
from .loss_config import ARMS
from .cuda_step import CapturedStep

def cache_coverage(plan):
    from datasets.random_reference import RandomReference
    identity=cache_identity(plan)
    key=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert key==plan['required_cache_key']
    base=ROOT/'datasets/static_tensor_cache_v1'/key
    assert json.loads((base/'identity.json').read_text(encoding='utf-8'))==identity
    result={}
    for role in ('train','dev_monitor500'):
        ds=RandomReference(role);blocks={}
        for i in ds.indices:blocks.setdefault(i//256,[]).append(i)
        for block,ids in blocks.items():
            path=base/ds.role/f'{block:04d}.pt';entry=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            assert path.is_file() and entry['identity']==key
            assert all(entry['members'][str(i)]==ds.records[i]['member'] for i in ids)
        result[role]=dict(members=len(ds.indices),chunks=len(blocks),metadata_only=True)
    return result

def strong_order_check():
    # Two opposite centers, no public constraints: coarse order restores the sign.
    state=torch.tensor([[[1.,0.,0.],[-1.,0.,0.]]],requires_grad=True)
    g=dict(valid=torch.ones((1,2),dtype=torch.bool),area=torch.ones((1,2)),
        target=torch.tensor([[[-1.,0.,0.],[1.,0.,0.]]]),
        reference=torch.tensor([[[-1.5,-.5,1.,1.],[.5,-.5,1.,1.]]]),
        constraints=torch.zeros((1,2,5),dtype=torch.long),specs=torch.zeros((1,2,4)))
    value=coarse_components(state,g)['strong_order'];value.sum().backward()
    assert value.item()>0 and state.grad[0,0,0]>0 and state.grad[0,1,0]<0
    return dict(wrong_order_has_restoring_gradient=True,loss=float(value))

def main():
    started=time.perf_counter();result=dict(status='preparing',formal_training_started=False,
        unique_training_cases=5,dev_payloads=0,test_payloads=0,postprocessing_calls=0,Q_calls=0)
    try:
        for path in HERE.rglob('*.py'):
            if 'runs' not in path.parts and 'preparation_diagnostics' not in path.parts:
                ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        auth=json.loads((HERE/'authorization.json').read_text(encoding='utf-8'))
        assert not auth['training_enabled'] and not auth['allowed_arms'] and not (HERE/'runs').exists()
        assert len(ARMS)==auth['maximum_training_arms']==3
        assert sha(ROOT/plan['baseline']['checkpoint'])==plan['baseline']['checkpoint_sha256']
        result['cache']=cache_coverage(plan);result['strong_order']=strong_order_check()
        rows=load_rows('train',plan['preflight_train_indices'])
        result['members']=[list(r['member']) for r in rows];assert len(set(map(tuple,result['members'])))==5
        checks=[];initial=None;result['arms']=checks
        for arm in ARMS:
            train.seed_all(0);model=GeometryModel(**train.MODEL)
            digest=hashlib.sha256(b''.join(t.contiguous().numpy().tobytes() for t in model.state_dict().values())).hexdigest()
            if initial is not None:assert digest==initial
            initial=digest
            from experiments.E134_supervision_interval.e119_model import GeometryModel as Parent
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0);parent=Parent(**train.MODEL)
            assert all(torch.equal(v,parent.state_dict()[k]) for k,v in model.state_dict().items())
            del parent
            assert sum(p.numel() for p in model.parameters())==712275
            model.cuda().eval();pack=ResidentRows(rows,'cuda',max_nodes=120);batch=pack.batch([i%5 for i in range(32)])
            public=inputs_only(batch)
            assert not (set(public)&{'reference','target','pin_coordinates'})
            assert (public['mask'][public['valid']]==1).all() and (public['constraints'][...,2]==0).all()
            with torch.no_grad():
                states=model(public);assert states.shape==(13,32,120,3)
                clean=per_case(states,batch,arm)
                if arm!='e119_C':expected=.8*base_per_case(states[12],batch)['total']+sum((.2/3)*base_per_case(states[k],batch)['total'] for k in (3,6,9))
                else:expected=.8*base_per_case(states[12],batch)['total']+sum((.2/3)*coarse_components(states[k],batch)['total'] for k in (3,6,9))
                torch.testing.assert_close(clean['total'],expected,rtol=2e-7,atol=1e-7)
                actual,_=objective_forward(model,batch,arm)
                if arm!='e119_A':
                    damaged=model(public,corrupt=True)
                    torch.testing.assert_close(actual['total'],.8*expected+.2*base_per_case(damaged[12],batch)['total'])
                torch.testing.assert_close(actual['final_total'],base_per_case(states[12],batch)['total'])
                partner,_=model.corruption_pattern(public)
                valid=public['valid'];pre=public['constraints'][...,1]!=0
                ident=torch.arange(120,device='cuda')[None,:].expand_as(partner)
                assert torch.equal(partner[pre|~valid],ident[pre|~valid])
                assert ((partner!=ident)&valid).sum()>0
                # Pairing is an involution: module-owned shapes never change.
                assert torch.equal(partner.gather(1,partner),ident)
            model.train();optimizer=train.optimizer_for(model)
            before={k:v.detach().clone() for k,v in model.state_dict().items()};rng=torch.cuda.get_rng_state_all()
            cap=CapturedStep(model,optimizer,batch,arm)
            assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
            assert all(torch.equal(x,y) for x,y in zip(rng,torch.cuda.get_rng_state_all()))
            values,norm=cap.replay();torch.cuda.synchronize()
            assert cap.projection.steps==1
            for layer in (model.initial_blocks[0],model.layers[3],model.layers[5],model.layers[-1]):
                assert layer.qkv.weight.grad is not None and layer.qkv.weight.grad.abs().sum()>0
            checks.append(dict(arm=arm,parameters=712275,initial_tensor_sha256=digest,
                disposable_updates=1,total=float(values['total']),clean_final_full_loss=float(values['final_total']),gradient_norm=float(norm),
                original_initialization_exact=True,capture_preserves_parameters_and_rng=True,
                public_input_boundary=True,corruption_involution_nonP_only=True))
            print(checks[-1],flush=True)
            del cap,optimizer,model,pack,batch,public,states,clean,actual,expected,before,values,norm,partner,ident
            gc.collect();torch.cuda.empty_cache()
        result['arms']=checks
        parent_gate=json.loads((ROOT/'experiments/E135_initial_supervision/preflight.json').read_text(encoding='utf-8'))
        assert parent_gate['ctrl_c_resume_verified']
        text=(HERE/'train.py').read_text(encoding='utf-8')
        for field in ("optimizer.state_dict()","rng=rng_state()","progress=progress","scheduler.state_dict()","restore_rng(previous['rng'])","elapsed_seconds=elapsed()","previous['projection']"):
            assert field in text
        result.update(status='passed',resume_contract_checked=True,
            resume_validation='Inherited E135 exact-resume engine; complete checkpoint fields/source paths statically checked; not exhaustively rerun',
            seconds=time.perf_counter()-started,sources=sources())
        save(HERE/'preflight.json',result)
        print('E138 three-arm preparation passed; no formal training started.',flush=True)
    except BaseException as error:
        result.update(status='failed',error=repr(error),traceback=traceback.format_exc());save(HERE/'preparation_failure.json',result);raise

if __name__=='__main__':main()
