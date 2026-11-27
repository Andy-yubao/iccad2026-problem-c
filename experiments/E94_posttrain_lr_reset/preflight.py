"""Five actual train cases: LR1e-3, inherited Adam, exact resume and protections."""
import ast,copy,gc,json,time
import numpy as np
import torch
from .io import ROOT,HERE,sha,save,sources,snapshot,checkpoint
from .initialization import load_parent,verify as verify_parent,provenance
from .model import GeometryModel
from .arms import model_config
from .data import load_rows,register_resident
from .resident import ResidentRows
from .batches import inputs_only
from .loss import per_case
from .loss_config import COEFFICIENTS
from .schedule import PlateauController,LR
from .train import seed_all,optimizer_for,rng_state,restore_rng,build_capture,fresh_progress,monitor
from .budget import charge_preflight

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

def verify():
    value=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    from .qualification.common import evaluator_sources,registration
    if value['status']!='passed' or value['sources']!=sources() or value['evaluation_sources']!=evaluator_sources():raise ValueError('Current-source preflight required')
    verify_parent();registration();return value

def make(lr=None):
    seed_all(0);model=GeometryModel(**model_config('simple')).cuda();opt=optimizer_for(model,'F');opt._e92_phase='F'
    parent=load_parent(model,opt);restore_rng(parent['rng'])
    if lr is not None:
        for g in opt.param_groups:g['lr']=lr
    return model,opt,parent

def scheduler_checks():
    p=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([p],lr=1e-3);s=PlateauController(opt,'F');s.step(1.,0)
    values=[]
    for epoch in range(1,12):
        s.step(1.,epoch);values.append([epoch,opt.param_groups[0]['lr'],s.drops,s.should_stop()])
        if epoch in (2,5,8):
            old=copy.deepcopy(s.state_dict());s=PlateauController(opt,'F');s.load_state_dict(old);same(s.state_dict(),old)
        assert s.should_stop()==(epoch==11)
    assert [v[1] for v in values if v[0] in (3,6,9)]==[2.5e-4,1e-4,5e-5]
    return {'protection':0,'baseline_epoch':0,'drop_epochs':[3,6,9],'earliest_stop_epoch':11,'trace':values,'recovery':True}

def main():
    begin=time.perf_counter();result={'status':'error','test_read':False};folder=HERE/'runs/preflight';folder.mkdir(parents=True,exist_ok=True)
    try:
        protected=verify_parent();result['schedule']=scheduler_checks();seed_all(0)
        for p in HERE.rglob('*.py'):
            if 'runs' not in p.relative_to(HERE).parts:ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        metadata=json.loads((HERE/'runs/sentinels.json').read_text(encoding='utf-8'));ids=[r['index'] for r in metadata['records']];assert len(set(ids))==5
        rows=load_rows('train',ids);pack=ResidentRows(rows,'cuda',max_nodes=120);register_resident(pack);batch=pack.batch([i%5 for i in range(32)])
        plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'));assert plan['lr_levels']==list(LR['F']) and plan['required_lr_drops']==3
        parent_state=torch.load(ROOT/provenance()['parent_model_and_optimizer'],map_location='cpu',weights_only=False)
        model,opt,parent=make();assert sum(p.numel() for p in model.parameters())==5829939 and all(p.requires_grad for p in model.parameters())
        same({k:v.cpu() for k,v in model.state_dict().items()},parent_state['model'])
        actual=copy.deepcopy(opt.state_dict());expected=copy.deepcopy(parent_state['optimizer'])
        for g in expected['param_groups']:g['lr']=1e-3
        for v in actual['state'].values():
            for k,item in v.items():
                if isinstance(item,torch.Tensor):v[k]=item.cpu()
        same(actual,expected);same(rng_state(),parent['rng']);assert all(float(v['step'])==36000 for v in actual['state'].values())
        model.eval()
        with torch.no_grad():
            values=per_case(model(inputs_only(batch)),batch,'F');initial=float(values['total'].mean())
        old=json.loads((ROOT/'experiments/E93_matched_posttrain/runs/preflight/results.json').read_text(encoding='utf-8'))
        assert abs(initial-old['five_case_initial_loss']['C'])<1e-7
        result['initialization']='exact P36 model/Adam step36000; only LR1e-3; P39 RNG; same five-case Loss as E93 C'
        result['five_case_initial_loss']=initial
        del model,opt,parent_state,actual,expected,values;gc.collect();torch.cuda.empty_cache()

        def trajectory(resume=False,lr=None):
            model,opt,parent=make(lr);progress=fresh_progress(32000,'F',parent);cap=build_capture(model,opt,pack,parent['projection'])
            assert progress['epoch']==39 and progress['step']==36000 and progress['phase_step']==0
            scheduler=PlateauController(opt,'F');before=rng_state();baseline=monitor(model,rows,'F','cuda');same(before,rng_state());scheduler.step(baseline['total'],0)
            for i in range(6):
                model.train();cap.replay(batch);progress['step']+=1;progress['phase_step']+=1
                if resume and i==1:
                    checkpoint(folder/'resume_check.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),'scheduler':scheduler.state_dict(),'progress':copy.deepcopy(progress),'projection':cap.projection.summary()})
                    saved=torch.load(folder/'resume_check.pt',map_location='cpu',weights_only=False);del cap,opt,model;gc.collect();torch.cuda.empty_cache()
                    model=GeometryModel(**model_config('simple')).cuda();model.load_state_dict(saved['model']);opt=optimizer_for(model,'F');opt._e92_phase='F';opt.load_state_dict(saved['optimizer'])
                    scheduler=PlateauController(opt,'F');scheduler.load_state_dict(saved['scheduler']);progress=saved['progress'];restore_rng(saved['rng'])
                    cap=build_capture(model,opt,pack,saved['projection']);same(saved['rng'],rng_state())
            assert cap.projection.steps==36006
            final={'model':{k:v.cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),
                   'rng':rng_state(),'scheduler':scheduler.state_dict(),'progress':progress,'projection':cap.projection.summary()}
            del cap,opt,model;gc.collect();torch.cuda.empty_cache();return final
        high=trajectory();resumed=trajectory(True);same(high,resumed);low=trajectory(lr=1e-4);same(high['rng'],low['rng'])
        assert any(not torch.equal(high['model'][k],low['model'][k]) for k in high['model'])
        result['actual_updates']='1e-3 six updates equal2+restore+4 bit-exact model/Adam/RNG/projection/progress; differ from1e-4 control with same RNG'
        result['high_lr_step_control']=high['projection'];result['low_lr_step_control']=low['projection']
        del high,resumed,low;gc.collect();torch.cuda.empty_cache()
        model,opt,parent=make();cap=build_capture(model,opt,pack,parent['projection']);old_model=copy.deepcopy(model.state_dict());old_opt=copy.deepcopy(opt.state_dict());old_projection=cap.projection.summary()
        bad={k:v.clone() for k,v in batch.items()};bad['target'][0,:,2]=100.
        try:cap.replay(bad)
        except RuntimeError:pass
        else:raise AssertionError('Explosion accepted')
        assert cap.max_case>=1000 and cap.values['total']<1000
        same(model.state_dict(),old_model);same(opt.state_dict(),old_opt);same(cap.projection.summary(),old_projection)
        bad['target'].fill_(float('nan'))
        try:cap.replay(bad)
        except RuntimeError:pass
        else:raise AssertionError('NaN accepted')
        same(model.state_dict(),old_model);same(opt.state_dict(),old_opt);same(cap.projection.summary(),old_projection)
        result['guards']='finite per-case>=1000 despite batch<1000 and NaN rejected before Adam, inherited state unchanged'
        from .dataset_view import ReferenceLayouts
        assert len(ReferenceLayouts('train'))==32000
        for role in ('test','local_test','tail_test'):
            try:ReferenceLayouts(role)
            except PermissionError:pass
            else:raise AssertionError('Sealed test accepted')
        from .qualification.common import registration,evaluator_sources,ready
        from .qualification.worker import STEADY
        from .qualification.run import validate_terminal
        reg=registration();oldpath=ROOT/reg['previous_results']['E90'];terminal=json.loads(oldpath.read_text(encoding='utf-8'))['records']
        assert STEADY==json.loads((oldpath.parent/'config.json').read_text(encoding='utf-8'))['steady']
        validate_terminal(terminal,ReferenceLayouts('dev_monitor500').records)
        try:validate_terminal([terminal[0],terminal[0]],ReferenceLayouts('dev_monitor500').records)
        except ValueError:pass
        else:raise AssertionError('Duplicate terminal cursor accepted')
        assert not ready()[0];verify_parent()
        result.update(status='passed',train_cases=ids,parameters=5829939,parent_hashes=protected['protected_artifacts'],sources=sources(),evaluation_sources=evaluator_sources(),postprocessing_calls=0)
        snapshot(folder/'source_snapshot.zip',{'sources':{**result['sources'],**result['evaluation_sources']}})
    except BaseException as error:result['error']=repr(error);raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-begin;charge_preflight(result['elapsed_seconds']);save(folder/'results.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','parent_hashes','evaluation_sources')},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
