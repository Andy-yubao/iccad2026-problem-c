"""Bounded E9 checks: synthetic gradients/resume and one real cached dev case."""
import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from datasets.worksets import LayoutReader,workset_manifest
from scoring.official import official_evaluator
from . import train as training
from .batches import sample,collate,inputs_only,decode,PUBLIC_KEYS
from .inference import prepare_inference
from .model import GeometryModel
from .losses import weighted_wire,all_components,objective_loss
from .probe import build_case,worker_call,evaluate_output,DRAFTS,BASELINE
from .evaluate import load_model
from .records import save,read_rows,run_path,provenance,snapshot,sha,OLD_CHECKPOINT,OLD_CHECKPOINT_SHA


def evaluation_gate(args,output,begin):
    from . import evaluate as evaluator
    from .compatibility import inspect_mib
    original_panel=evaluator.panel_reader;original_call=evaluator.worker_call
    manifest,reader,indices,baseline,cached=original_panel('dev100')
    evaluator.panel_reader=lambda panel:(manifest,reader,indices[:1],baseline,cached)
    network=load_model(OLD_CHECKPOINT,args.device)
    identity={'checkpoint':str(OLD_CHECKPOINT),'checkpoint_sha256':OLD_CHECKPOINT_SHA,'fixture_only':True,'scope':'one cached dev case only'}
    try:
        partial=evaluator.evaluate_quality(network,output/'quality_fixture1','dev100',args.device,identity,0.)
        assert partial['status']=='resource_budget' and partial['completed_cases']==0
        complete=evaluator.evaluate_quality(network,output/'quality_fixture1','dev100',args.device,identity,120.)
        assert complete['status']=='completed' and complete['completed_cases']==1
        def forbidden(*unused,**kwargs):raise AssertionError('Completed evaluation attempted another solver call')
        evaluator.worker_call=forbidden
        reused=evaluator.evaluate_quality(network,output/'quality_fixture1','dev100',args.device,identity,120.)
        assert reused['completed_cases']==1
        with torch.no_grad():next(network.parameters()).flatten()[0]+=1.
        try:evaluator.evaluate_quality(network,output/'quality_fixture1','dev100',args.device,identity,120.)
        except ValueError:pass
        else:raise AssertionError('Changed model accepted into same evaluation')
    finally:evaluator.panel_reader=original_panel;evaluator.worker_call=original_call
    c=np.zeros((2,5));c[:,2]=1;specs=np.full((2,4),-1.)
    make=lambda area:(2,np.array(area),np.empty((0,3)),np.empty((0,3)),np.empty((0,2)),c,specs)
    assert inspect_mib(make([4,4]))['eligible_for_local_mib_compatibility']
    assert not inspect_mib(make([72,240]))['eligible_for_local_mib_compatibility']
    c[:,0]=1;specs[:,2:]=[[2,2],[3,2]]
    assert not inspect_mib(make([4,6]))['eligible_for_local_mib_compatibility']
    result={'status':'passed','fixture_only':True,'real_dev_cases':1,'test_payload_reads':0,
            'zero_budget_stops_before_solver':True,'resume_completes_same_member':True,
            'completed_evaluation_does_not_resolve':True,'changed_model_resume_rejected':True,
            'public_mib_compatibility_positive_and_negative_examples':True,
            'one_case_Q':complete['summary']['Q_mean'],'wall_seconds':time.monotonic()-begin}
    save(output/'results.json',result);print(json.dumps(result),flush=True)


def center_gate(args,output,begin):
    rows,_=fixtures();g=collate(rows,list(range(5)),args.device)
    initial=torch.load(OLD_CHECKPOINT,map_location='cpu',weights_only=False)
    retained=GeometryModel(**initial['model_config']).to(args.device);retained.load_state_dict(initial['model'])
    refined=GeometryModel(**{**initial['model_config'],'architecture':'center_refiner'}).to(args.device)
    refined.load_state_dict(initial['model'],strict=False)
    with torch.no_grad():a=retained(inputs_only(g));b=refined(inputs_only(g))
    assert torch.equal(a,b),'Zero-initialized refiner changed retained proposals'
    assert all(p.requires_grad==name.startswith('center_refiner.') for name,p in refined.named_parameters())
    del retained,refined
    fixture_initial={**initial,'fixture_only':True};manifest=workset_manifest()
    count=[0];limit=[6];real_step=torch.optim.AdamW.step
    def wrapped_step(self,*a,**kw):
        value=real_step(self,*a,**kw);count[0]+=1;return value
    torch.optim.AdamW.step=wrapped_step
    try:
        settings=SimpleNamespace(seed=0,device=args.device,batch_size=3,resume=False,arm_seconds=120.,budget_seconds=600.,max_epochs=30)
        stop=lambda:count[0]>=limit[0] or time.monotonic()-begin>=580
        for tag,resume,steps in [('uninterrupted',False,6),('resumed',False,3),('resumed',True,3)]:
            count[0]=0;limit[0]=steps;settings.resume=resume
            training.train_arm('center_refiner',output/tag,rows,rows,list(range(5)),list(range(5)),fixture_initial,
                               settings,manifest,lambda:time.monotonic()-begin,stop,stop)
        a=torch.load(output/'uninterrupted/checkpoints/latest.pt',map_location='cpu',weights_only=False)
        b=torch.load(output/'resumed/checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert a['step']==b['step']==6
        assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
        assert all(torch.equal(b['model'][key],value) for key,value in initial['model'].items())
        final=GeometryModel(**b['model_config']).to(args.device);final.load_state_dict(b['model'])
        with torch.no_grad():state=final(inputs_only(g))
        assert torch.isfinite(state).all()
        retained=GeometryModel(**initial['model_config']).to(args.device);retained.load_state_dict(initial['model'])
        with torch.no_grad():old_state=retained(inputs_only(g))
        assert torch.equal(state[...,2],old_state[...,2]),'Frozen shape outputs changed'
        assert not torch.equal(state[...,:2],old_state[...,:2]),'Position module failed to learn any change'
        assert torch.equal(decode(state,g)[...,2:],decode(old_state,g)[...,2:])
        result={'status':'passed','fixture_only':True,'real_dev_cases':0,'synthetic_cases':5,'test_payload_reads':0,
                'zero_initialization_preserves_original_output':True,'backbone_weights_unchanged':True,
                'shape_predictions_unchanged':True,'center_predictions_changed':True,'resume_model_exactly_equal':True,
                'steps':6,'trainable_parameters':sum(p.numel() for p in final.parameters() if p.requires_grad),
                'wall_seconds':time.monotonic()-begin}
        save(output/'results.json',result);print(json.dumps(result),flush=True)
    finally:torch.optim.AdamW.step=real_step


def fixtures():
    rows=[];inputs=[]
    for n in range(4,9):
        reference=torch.tensor([[2.*i,0.,2.,2.] for i in range(n)])
        c=torch.zeros(n,5);c[:2,2]=1;c[:2,3]=1;c[0,1]=1;c[-1,0]=1
        reference[-1,2]=3.
        area=reference[:,2:].prod(1)
        specs=torch.full((n,4),-1.);specs[0]=reference[0];specs[-1,2:]=reference[-1,2:]
        b2b=torch.tensor([[i,i+1,.3+.2*i] for i in range(n-1)])
        pins=torch.tensor([[0.,0.],[20.,3.]]) if n!=5 else torch.empty(0,2)
        p2b=torch.tensor([[0.,1.,1.5],[1.,float(n-1),.7]]) if len(pins) else torch.empty(0,3)
        inp=(n,area,b2b,p2b,pins,c,specs);inputs.append(inp);rows.append(sample(inp,reference))
    return rows,inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda')
    p.add_argument('--evaluation-only',action='store_true',help='Check quality evaluator budget/resume on exactly one dev member')
    p.add_argument('--center-only',action='store_true',help='Synthetic-only frozen-backbone center module and resume check')
    args=p.parse_args();begin=time.monotonic();output=run_path(args.output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    config={**provenance(),'fixture_only':True,'real_dev_cases':0 if args.center_only else 1,'synthetic_cases':5,'test_payload_reads':0,
            'budget_seconds':600,'purpose':'Finite gradients, exact objective HPWL, deterministic resume, retained checkpoint compatibility. No research training.'}
    save(output/'config.json',config);snapshot(output,config['source_sha256'])
    if args.center_only:return center_gate(args,output,begin)
    if args.evaluation_only:return evaluation_gate(args,output,begin)
    assert sha(OLD_CHECKPOINT)==OLD_CHECKPOINT_SHA
    initial=torch.load(OLD_CHECKPOINT,map_location='cpu',weights_only=False)
    model=GeometryModel(**initial['model_config']).to(args.device);model.load_state_dict(initial['model']);model.eval()
    rows,public=fixtures();g=collate(rows,list(range(5)),args.device)
    with torch.no_grad():state=model(inputs_only(g))
    assert state.shape==(5,8,3)
    try:model(g)
    except ValueError:pass
    else:raise AssertionError('Network accepted reference/target')
    changed=sample(public[0],rows[0]['reference']*rows[0]['scale']+5.)
    for key in PUBLIC_KEYS:
        if key in rows[0]:assert torch.equal(rows[0][key],changed[key]),key
    official=official_evaluator();wire=weighted_wire(g['reference'],g).cpu()
    wire_errors=[]
    for i,inp in enumerate(public):
        reference=rows[i]['reference']*rows[i]['scale']
        truth=official.calculate_hpwl_b2b(reference.tolist(),inp[2])+official.calculate_hpwl_p2b(reference.tolist(),inp[3],inp[4])
        error=abs(float(wire[i]*rows[i]['scale'])-truth);wire_errors.append(error)
        assert error<2e-5,(error,truth)
    gradients={}
    for name in training.VARIANTS:
        model.zero_grad(set_to_none=True);values=all_components(model(inputs_only(g)),g)
        loss=objective_loss(values,name,3.);loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
        gradients[name]=float(loss.detach())
    del model
    fixture_initial={**initial,'fixture_only':True}
    manifest=workset_manifest();resume_checks={}
    real_step=torch.optim.AdamW.step
    # Exercise the production loop, checkpoint/RNG restoration, monitors and
    # optimizer state using only synthetic fixtures. Stop before quality nodes.
    for name in training.VARIANTS:
        count=[0];limit=[6]
        def wrapped_step(self,*a,**kw):
            result=real_step(self,*a,**kw);count[0]+=1;return result
        torch.optim.AdamW.step=wrapped_step
        try:
            settings=SimpleNamespace(seed=0,device=args.device,batch_size=3,resume=False,arm_seconds=120.,
                                     budget_seconds=600.,max_epochs=30)
            stop=lambda:count[0]>=limit[0] or time.monotonic()-begin>=580
            full=output/(name+'_uninterrupted')
            training.train_arm(name,full,rows,rows,list(range(5)),list(range(5)),fixture_initial,settings,manifest,
                               lambda:time.monotonic()-begin,stop,stop)
            count[0]=0;limit[0]=3;partial=output/(name+'_resumed')
            training.train_arm(name,partial,rows,rows,list(range(5)),list(range(5)),fixture_initial,settings,manifest,
                               lambda:time.monotonic()-begin,stop,stop)
            count[0]=0;settings.resume=True
            training.train_arm(name,partial,rows,rows,list(range(5)),list(range(5)),fixture_initial,settings,manifest,
                               lambda:time.monotonic()-begin,stop,stop)
            a=torch.load(full/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            b=torch.load(partial/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
            assert a['step']==b['step']==6 and a['epoch']==b['epoch'] and a['cursor']==b['cursor']
            assert all(torch.equal(a['model'][key],b['model'][key]) for key in a['model'])
            assert torch.equal(a['seen'],b['seen']) and a['scheduler']==b['scheduler']
            resume_checks[name]={'model_exactly_equal':True,'steps':6,'resume_at_step':3,
                                 'scheduler_equal':True,'seen_equal':True,'epoch':a['epoch'],'cursor':a['cursor']}
        finally:torch.optim.AdamW.step=real_step
    # One already-used real development case, no gradients and no optimizer.
    reader=LayoutReader(manifest,'dev_quality');raw=reader.case(0);case=build_case(raw)
    network=load_model(OLD_CHECKPOINT,args.device);public_g=prepare_inference(raw.solver_inputs,args.device)
    with torch.no_grad():draft=(decode(network(public_g),public_g)[0]*public_g['scale'][0]).cpu().numpy().astype(float)
    c=np.asarray(raw.solver_inputs[5]);spec=np.asarray(raw.solver_inputs[6]);locked=c[:,:2].any(1)
    draft[locked,2:]=spec[locked,2:];draft[c[:,1]!=0]=spec[c[:,1]!=0]
    cached=next(r for r in read_rows(DRAFTS) if r['case_id']==case.case_id)
    delta=float(np.max(np.abs(draft-np.array(cached['draft']))));assert delta<1e-4,delta
    final=evaluate_output(case,worker_call(output,raw.solver_inputs,draft))
    assert 'error' not in final and final['final']['metrics']['is_feasible']
    reference=next(r for r in read_rows(BASELINE) if r['case_id']==case.case_id)
    old_q=reference['variants']['conic_robust']['final']['metrics']['cost']
    assert abs(final['final']['metrics']['cost']-old_q)<1e-6
    save(output/'real_case.json',{'case_id':case.case_id,'source':case.source,'variant':final,
                                'cached_E8_Q':old_q,'max_draft_difference':delta})
    try:load_model(output/(training.VARIANTS[0]+'_resumed/checkpoints/latest.pt'),args.device)
    except ValueError:pass
    else:raise AssertionError('Synthetic checkpoint accepted as a research model')
    result={'status':'passed','fixture_only':True,'real_dev_cases':1,'synthetic_cases':5,'test_payload_reads':0,
            'gradient_losses':gradients,'max_wire_error':max(wire_errors),'label_isolation':True,
            'resume':resume_checks,'retained_model_draft_max_difference':delta,
            'frozen_E8_Q_match':final['final']['metrics']['cost'],'synthetic_checkpoint_rejected_for_research':True,
            'wall_seconds':time.monotonic()-begin}
    save(output/'results.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
