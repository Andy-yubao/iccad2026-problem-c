"""Disposable exact-path checks, five registered train members, no formal run."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast,copy,gc,io,json,time,traceback
import numpy as np
import torch
from . import train
from .io import HERE,ROOT,save,sources
from .model import GeometryModel
from .data import load_rows
from .resident import ResidentRows
from .public import inputs_only
from .loss import per_case,layer_loss,mib_loss
from .cuda_step import CapturedStep,EagerStep
from .step_control import GraphStepControl
from .augmentation import axis_swap
from .initialization import verify_initial
from .evaluation.collect import public_batch
def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b,(a,b)
def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
def main():
    start=time.perf_counter();report=dict(status='preparing',formal_training_started=False,test_payloads=0,dev_payloads=0,formal_training_updates=0)
    try:
        assert not json.loads((HERE/'authorization.json').read_text())['training_enabled'] and not (HERE/'runs').exists()
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        plan=json.loads((HERE/'plan.json').read_text());train.seed_all(0)
        rows=load_rows('train',plan['preflight_train_indices']);assert len(rows)==5
        profile=json.loads((HERE/'data_cache_v1/profile.json').read_text(encoding='utf-8'))
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=profile['max_pins'],max_ports=profile['max_ports'])
        report['dataset_maximum_padding']=profile
        batch=pack.batch(range(5));g=inputs_only(batch)
        for i,row in enumerate(rows):
            public=public_batch(row['inputs'])
            single=inputs_only(ResidentRows([row],'cuda',max_nodes=120).batch([0]))
            equal(public,single)
        swapped=axis_swap(dict(batch,axis_swap=torch.ones(5,device='cuda',dtype=torch.bool)))
        twice=axis_swap(swapped)
        for key in batch:torch.testing.assert_close(twice[key],batch[key],rtol=0,atol=0)
        assert bool((batch['mask'][batch['valid']]==1).all())
        report['members']=[list(r['member']) for r in rows];report['stages']={}
        for phase in ('initial','backend'):
            arm='e153_'+phase;train.seed_all(0);model=GeometryModel(phase=phase).cuda().eval()
            frozen=cpu_state(model.initial_generator) if phase=='backend' else None
            assert sum(p.numel() for p in model.parameters())==plan['parameters'][arm]
            with torch.no_grad():states=model(g)
            assert states.shape==((1 if phase=='initial' else 13),5,120,3)
            try:model(dict(g,target=batch['target']))
            except ValueError:pass
            else:raise AssertionError('Reference accepted as input')
            values=per_case(states,batch,arm);assert all(torch.isfinite(v).all() for v in values.values())
            changed=dict(g,constraints=g['constraints'].clone());mask=changed['constraints'][...,2]>0;changed['constraints'][...,2][mask]+=100
            with torch.no_grad():torch.testing.assert_close(model(changed),states,rtol=0,atol=0)
            perm=torch.arange(119,-1,-1,device='cuda');mp=g['port_valid'].shape[1];pp=torch.arange(mp-1,-1,-1,device='cuda');keys=torch.cat((perm,120+pp));pg={}
            for k,v in g.items():
                if k in ('adjacency','mib','group'):pg[k]=v[:,perm][:,:,perm]
                elif k in ('role_hops','role_edges','role_group'):pg[k]=v[:,perm][:,:,keys]
                elif k in ('port_features','port_valid'):pg[k]=v[:,pp]
                elif k=='scale':pg[k]=v
                else:pg[k]=v[:,perm]
            with torch.no_grad():torch.testing.assert_close(model(pg),states[:,:,perm],rtol=2e-4,atol=5e-5)
            transposed=torch.stack((states[...,1],states[...,0],-states[...,2]),-1)
            torch.testing.assert_close(per_case(transposed,swapped,arm)['total'],values['total'],rtol=2e-5,atol=1e-4)
            if phase=='backend':
                full=layer_loss(states[4],batch,1.,False);reduced=layer_loss(states[4],batch,.75,False)
                torch.testing.assert_close(full['total']-reduced['total'],.25*full['reference_weighted'],rtol=2e-5,atol=1e-6)
                assert not reduced['overlap_weighted'].any()
            zero=states[0].clone();zero[...,2]=0;assert not mib_loss(zero,batch).any()
            del states,values,changed,pg,transposed
            model.train();optimizer=train.optimizer_for(model);before=cpu_state(model)
            sample64=pack.batch([i%5 for i in range(64)]);cap=CapturedStep(model,optimizer,sample64,arm)
            equal(before,cpu_state(model));assert not optimizer.state
            cap.replay();values,norm=cap.replay()
            branches=['initial_generator'] if phase=='initial' else ['problem_encoder','units.0.group_move','units.0.member_move','units.0.move_gate','units.1','units.2']
            for prefix in branches:
                grads=[p.grad for name,p in model.named_parameters() if name.startswith(prefix) and p.requires_grad]
                assert grads and all(v is not None and torch.isfinite(v).all() for v in grads) and any(v.abs().sum()>0 for v in grads),prefix
            tail_sample=pack.batch([i%5 for i in range(32)]);tail=EagerStep(model,optimizer,tail_sample,arm,cap.projection)
            tail.replay();cap.replay()
            if frozen is not None:verify_initial(model,frozen)
            buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=train.rng_state(),projection=cap.projection.summary()),buffer);buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False)
            train.restore_rng(saved['rng']);tail.replay();expected=cpu_state(model);expected_opt=copy.deepcopy(optimizer.state_dict());expected_projection=copy.deepcopy(cap.projection.summary())
            del cap,tail,optimizer,model,values,sample64,tail_sample;gc.collect();torch.cuda.empty_cache()
            model=GeometryModel(phase=phase).cuda().train();model.load_state_dict(saved['model']);optimizer=train.optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
            control=GraphStepControl(model);control.steps=saved['projection']['steps'];control.clipped.copy_(torch.tensor(saved['projection']['clipped_steps_per_group'],device='cuda'));control.last_scales.copy_(torch.tensor(saved['projection']['last_scales'],device='cuda'))
            resumed=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),arm,control)
            train.restore_rng(saved['rng']);resumed.replay()
            equal(expected,cpu_state(model));equal(expected_opt,optimizer.state_dict());equal(expected_projection,control.summary())
            if frozen is not None:verify_initial(model,frozen)
            report['stages'][arm]=dict(parameters=plan['parameters'][arm],trainable_parameters=plan['trainable_parameters'][arm],public_path_exact=True,MIB_id_invariance=True,module_port_permutation=True,axis_swap=True,all_new_branches_have_gradient=True,capture_without_updates=True,serialized_next_update_bitwise=True,frozen_L0=phase=='backend',disposable_updates=5)
            print(f'[PREFLIGHT] {arm} GPU64/tail32, gradients, public inputs and serialized next update passed',flush=True)
            del resumed,control,model,optimizer,saved,expected,expected_opt;gc.collect();torch.cuda.empty_cache()
        from .evaluation.solver import optimize,mib_valid
        from .evaluation.repair.backend.frozen.packing import prepare_problem,_dimensions
        from .evaluation.repair.backend.frozen.solver import hard_valid
        row=rows[0];p=prepare_problem(*row['inputs']);width,height=_dimensions(p,None,True)
        assert mib_valid(np.column_stack((np.zeros((p.n,2)),width,height)),p)
        draft=(row['graph']['reference']*row['graph']['scale']).numpy().astype(float)
        final,info=optimize(draft,row['inputs']);assert hard_valid(final,p) and mib_valid(final,p)
        from datasets.cases import ResearchCase
        from datasets.reference_layouts import public_hard_reference
        from scoring.evaluation import evaluate_case
        inputs=row['inputs'];reference=public_hard_reference(inputs[1],inputs[5],inputs[6])
        case=ResearchCase('E153.preflight',dict(member=list(row['member'])),inputs,reference,row['baseline_metrics'])
        scored=evaluate_case(case,final,info['seconds'])['metrics']
        flags=np.asarray(inputs[5]);denominator=int((flags[:,4]!=0).sum())
        for col in (2,3):
            denominator+=sum(max(0,int((flags[:,col]==group).sum())-1) for group in np.unique(flags[:,col]) if group>0)
        assert scored['is_feasible'] and scored['mib_violations']==0 and scored['max_possible_violations']==denominator
        report['full_MIB_score_denominator_checked']=True;report['train_case_score_smoke_calls']=1
        report.update(status='passed',sources=sources(),actual_postprocessing_smoke_train_cases=1,postprocessing_MIB_and_hard_checks=True,peak_reserved_bytes=torch.cuda.max_memory_reserved(),formal_weights_reused=False,elapsed_seconds=time.perf_counter()-start)
        print('[PREFLIGHT] Full MIB postprocessing smoke passed',flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:save(HERE/'preflight.json',report)
if __name__=='__main__':main()
