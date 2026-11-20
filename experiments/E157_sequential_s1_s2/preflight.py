"""Five-train-case tests for both sequential phases, without formal updates."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import ast,copy,gc,io,json,time,traceback
import numpy as np
import torch
from . import train
from .io import HERE,ROOT,save,sources
from .data import load_rows
from .resident import ResidentRows
from .public import inputs_only
from .model import GeometryModel
from .initialization import initialize_model,verify_frozen
from .loss import per_case,layer_loss,drift_terms,A,B
from .cuda_step import CapturedStep,EagerStep
from .step_control import GraphStepControl
from .augmentation import axis_swap
from .monitor_audit import audit
from .batches import decode

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

def make_model(phase,disposable):
    model=GeometryModel(phase=phase,layers=8 if phase=='s1' else 12).cuda()
    reference=initialize_model(model,'e157_'+phase,disposable_parent=disposable)
    return model,reference

def main():
    start=time.perf_counter();report=dict(status='preparing',formal_training_updates=0,test_payloads=0,dev_payloads=0,stages={})
    try:
        assert not json.loads((HERE/'authorization.json').read_text())['training_enabled'] and not (HERE/'runs').exists()
        for p in HERE.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        plan=json.loads((HERE/'plan.json').read_text());train.seed_all(0)
        rows=load_rows('train',plan['preflight_train_indices']);assert len(rows)==5
        profile=json.loads((ROOT/'datasets/random_mib40k_v1/cache/profile.json').read_text())
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=profile['max_pins'],max_ports=profile['max_ports'])
        batch=pack.batch(range(5));g=inputs_only(batch);disposable=None;parent_output=None
        for phase in ('s1','s2'):
            arm='e157_'+phase;train.seed_all(0);model,reference=make_model(phase,disposable);model.eval()
            assert sum(p.numel() for p in model.parameters())==plan['parameters'][arm]
            assert sum(p.numel() for p in model.parameters() if p.requires_grad)==plan['trainable_parameters'][arm]
            with torch.no_grad():
                out=model(g);equal(out,model(g));assert out.shape==((9 if phase=='s1' else 13),5,120,3)
                if phase=='s1':
                    # Reproduce registered historical inference solely to verify the frozen prefix.
                    from experiments.E156_two_s0_stage.model import GeometryModel as Parent
                    descriptor=json.loads((HERE/'parent_s0.json').read_text());checkpoint=torch.load(ROOT/descriptor['checkpoint'],map_location='cpu',weights_only=False)
                    historical=Parent(**checkpoint['model_config']).cuda().eval();historical.load_state_dict(checkpoint['model']);equal(out[:5],historical(g));del historical,checkpoint
                else:equal(out[:9],parent_output)
                values=per_case(out,batch,arm)
                terminal=8 if phase=='s1' else 12;first=terminal-2
                r1=layer_loss(out[first],batch);r2=layer_loss(out[terminal],batch)
                expected=.2*r1['total']+.8*r2['total']
                if phase=='s2':
                    center,ratio=drift_terms(out[8],out[12],batch);expected=expected+.8*.2*(center+.25*ratio)
                    zero=drift_terms(out[8],out[8],batch);assert not (zero[0]+zero[1]).any()
                    within=out[8].clone();within[...,0]+=.04*batch['area'].clamp_min(1e-12).sqrt();within[...,2]+=.09
                    zero=drift_terms(out[8],within,batch);assert not (zero[0]+zero[1]).any()
                torch.testing.assert_close(values['total'],expected,rtol=1e-6,atol=1e-7)
                assert not values['reference_weighted'].any() and 'group_bottleneck_weighted' not in values
                torch.testing.assert_close(r2['overlap_weighted'],A*r2['overlap_total'],rtol=0,atol=0)
                changed=dict(batch,target=batch['target']+123.)
                equal(values['total'],per_case(out,changed,arm)['total'])
                swapped=axis_swap(dict(batch,axis_swap=torch.ones(5,device='cuda',dtype=torch.bool)))
                transposed=torch.stack((out[...,1],out[...,0],-out[...,2]),-1)
                torch.testing.assert_close(per_case(transposed,swapped,arm)['total'],values['total'],rtol=2e-5,atol=1e-4)
                diagnostics=audit(out,batch,values);assert all(torch.isfinite(v).all() for v in diagnostics.values())
                prefix=out[:5 if phase=='s1' else 9].clone()
            model.train();verify_frozen(model,reference)
            with torch.no_grad():equal(model(g)[:len(prefix)],prefix)
            optimizer=train.optimizer_for(model);before=cpu_state(model)
            cap=CapturedStep(model,optimizer,pack.batch([i%5 for i in range(64)]),arm)
            equal(before,cpu_state(model));assert not optimizer.state
            cap.replay();cap.replay();verify_frozen(model,reference)
            for prefix_name in (phase+'_encoder',phase+'.layers.0',phase+'.layers.1',phase+'.heads',phase+'.demand_encoder'):
                grads=[p.grad for k,p in model.named_parameters() if k.startswith(prefix_name) and p.requires_grad]
                assert grads and all(v is not None and torch.isfinite(v).all() for v in grads) and any(v.abs().sum()>0 for v in grads),prefix_name
            tail=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),arm,cap.projection);tail.replay();verify_frozen(model,reference)
            buffer=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=train.rng_state(),projection=cap.projection.summary()),buffer);buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=False)
            train.restore_rng(saved['rng']);tail.replay();expected_model=cpu_state(model);expected_opt=copy.deepcopy(optimizer.state_dict());expected_projection=copy.deepcopy(cap.projection.summary())
            del cap,tail,optimizer,model;gc.collect();torch.cuda.empty_cache()
            model,reference=make_model(phase,disposable);model.train();model.load_state_dict(saved['model']);optimizer=train.optimizer_for(model);optimizer.load_state_dict(saved['optimizer'])
            control=GraphStepControl(model);control.steps=saved['projection']['steps'];control.clipped.copy_(torch.tensor(saved['projection']['clipped_steps_per_group'],device='cuda'));control.last_scales.copy_(torch.tensor(saved['projection']['last_scales'],device='cuda'))
            resumed=EagerStep(model,optimizer,pack.batch([i%5 for i in range(32)]),arm,control)
            train.restore_rng(saved['rng']);resumed.replay();equal(expected_model,cpu_state(model));equal(expected_opt,optimizer.state_dict());equal(expected_projection,control.summary());verify_frozen(model,reference)
            model.eval()
            with torch.no_grad():parent_output=model(g).detach().clone()
            disposable=cpu_state(model)
            report['stages'][arm]=dict(parameters=plan['parameters'][arm],trainable_parameters=plan['trainable_parameters'][arm],frozen_prefix_exact=True,registered_parent_inference_exact=True,no_reference_or_bottleneck=True,absolute_overlap_full=True,loss_sum=True,axis_transpose=True,all_new_branches_have_gradients=True,GPU64_tail32=True,serialized_next_update_bitwise=True)
            print('[PREFLIGHT] '+arm+' frozen prefix, full public Loss, GPU64/tail32 and next-update resume passed',flush=True)
            del resumed,control,optimizer,model,saved,expected_model,expected_opt;gc.collect();torch.cuda.empty_cache()
        from .evaluation.solver import optimize,mib_valid
        from .evaluation.repair.backend.frozen.packing import prepare_problem
        from .evaluation.repair.backend.frozen.solver import hard_valid
        from datasets.cases import ResearchCase
        from datasets.reference_layouts import public_hard_reference
        from scoring.evaluation import evaluate_case
        j=max(range(5),key=lambda i:int((rows[i]['inputs'][5][:,2]>0).sum()));row=rows[j];inputs=row['inputs'];n=int(inputs[0])
        draft=(decode(parent_output[-1],batch)[j,:n]*batch['scale'][j]).cpu().numpy().astype(float)
        final,info=optimize(draft,inputs);problem=prepare_problem(*inputs);assert hard_valid(final,problem) and mib_valid(final,problem)
        case=ResearchCase('E157.preflight',dict(member=list(row['member'])),inputs,public_hard_reference(inputs[1],inputs[5],inputs[6]),row['baseline_metrics'])
        metrics=evaluate_case(case,final,info['seconds'])['metrics'];assert metrics['is_feasible'] and metrics['mib_violations']==0
        flags=np.asarray(inputs[5]);denominator=int((flags[:,4]!=0).sum())
        for col in (2,3):denominator+=sum(max(0,int((flags[:,col]==label).sum())-1) for label in np.unique(flags[:,col]) if label>0)
        assert metrics['max_possible_violations']==denominator
        report.update(status='passed',sources=sources(),full_MIB_postprocessing_smoke=True,postprocessing_train_cases=1,score_denominator_exact=True,elapsed_seconds=time.perf_counter()-start,peak_reserved_bytes=torch.cuda.max_memory_reserved(),members=[list(r['member']) for r in rows])
        print('[PREFLIGHT] Independent full-MIB postprocessing and scoring smoke passed',flush=True)
    except BaseException as error:
        report.update(status='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:save(HERE/'preflight.json',report)
if __name__=='__main__':main()
