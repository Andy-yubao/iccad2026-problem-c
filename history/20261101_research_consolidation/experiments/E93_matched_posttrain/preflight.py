"""Five train cases: inherited Adam, matched C/G state/RNG and exact resume."""
import copy,gc,json,time
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
    if value['status']!='passed' or value['sources']!=sources():raise ValueError('Current-source five-case preflight required')
    verify_parent();return value

def make(arm):
    seed_all(0);model=GeometryModel(**model_config('simple')).cuda()
    opt=optimizer_for(model,arm);opt._e92_phase=arm
    parent=load_parent(model,opt);restore_rng(parent['rng'])
    return model,opt,parent

def scheduler_checks():
    p=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([p],lr=1e-4);s=PlateauController(opt,'C')
    s.step(1.,0)
    s.step(1.,1);s.step(1.,2);assert s.drops==0 and s.lr_bad==2
    state=copy.deepcopy(s.state_dict());s=PlateauController(opt,'C');s.load_state_dict(state)
    s.step(1.,3);assert s.drops==1 and opt.param_groups[0]['lr']==5e-5 and not s.should_stop()
    s.step(1.,4);assert not s.should_stop()
    s.step(1.,5);assert s.should_stop()
    return {'baseline_epoch':0,'drop_epoch':3,'earliest_stop_epoch':5,'protection':0,'recovery':True}

def main():
    begin=time.perf_counter();result={'status':'error','test_read':False};folder=HERE/'runs/preflight';folder.mkdir(parents=True,exist_ok=True)
    try:
        protected=verify_parent();seed_all(0);result['schedule']=scheduler_checks()
        metadata=json.loads((HERE/'runs/sentinels.json').read_text(encoding='utf-8'))
        ids=[r['index'] for r in metadata['records']];assert len(ids)==5
        rows=load_rows('train',ids);pack=ResidentRows(rows,'cuda',max_nodes=120);register_resident(pack)
        g=pack.batch([i%5 for i in range(32)])
        source=provenance()['parent_model_and_optimizer'];parent_state=torch.load(ROOT/source,map_location='cpu',weights_only=False)
        initial={}
        plan=json.loads((HERE/'plan.json').read_text(encoding='utf-8'))
        for arm in ('C','G'):
            assert plan['overlap_weights'][arm]==COEFFICIENTS[arm]['overlap']
            assert plan['task_weight']==COEFFICIENTS[arm]['task']
            assert list(LR[arm])==plan['lr_levels']
            model,opt,parent=make(arm)
            assert sum(p.numel() for p in model.parameters())==5829939 and all(p.requires_grad for p in model.parameters())
            same({k:v.cpu() for k,v in model.state_dict().items()},parent_state['model'])
            actual=copy.deepcopy(opt.state_dict());expected=copy.deepcopy(parent_state['optimizer'])
            for group in expected['param_groups']:group['lr']=1e-4
            for v in actual['state'].values():
                for key,item in v.items():
                    if isinstance(item,torch.Tensor):v[key]=item.cpu()
            same(actual,expected);same(rng_state(),parent['rng'])
            initial[arm]={'model':{k:v.cpu().clone() for k,v in model.state_dict().items()},'optimizer':actual,'rng':rng_state()}
            del model,opt;gc.collect();torch.cuda.empty_cache()
        same(initial['C'],initial['G']);del initial,parent_state
        result['matched_initialization']='bit-identical P36 model and Adam moments/step36000; common terminal39 RNG; only LR reset1e-4'
        model,opt,parent=make('C');model.eval()
        with torch.no_grad():
            output=model(inputs_only(g));c=per_case(output,g,'C');gvalues=per_case(output,g,'G')
            for key in c:
                if key!='total':same(c[key],gvalues[key])
            torch.testing.assert_close(gvalues['total']-c['total'],COEFFICIENTS['C']['overlap']*c['overlap_total'],rtol=1e-5,atol=1e-6)
            assert c['total'].max()<1000 and gvalues['total'].max()<1000
            result['five_case_initial_loss']={'C':float(c['total'].mean()),'G':float(gvalues['total'].mean())}
        del model,opt,output,c,gvalues;gc.collect();torch.cuda.empty_cache()

        def trajectory(arm,resume=False):
            model,opt,parent=make(arm);progress=fresh_progress(32000,arm,parent)
            assert progress['epoch']==39 and progress['phase_epoch']==0 and progress['step']==36000 and progress['phase_step']==0
            cap=build_capture(model,opt,pack,parent['projection'])
            scheduler=PlateauController(opt,arm)
            before=rng_state();baseline=monitor(model,rows,arm,'cuda');same(before,rng_state())
            scheduler.step(baseline['total'],0)
            for i in range(6):
                model.train();cap.replay(g)
                progress['step']+=1;progress['phase_step']+=1
                if resume and i==1:
                    checkpoint(folder/'resume_check.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),
                        'rng':rng_state(),'scheduler':scheduler.state_dict(),'progress':copy.deepcopy(progress),'projection':cap.projection.summary()})
                    saved=torch.load(folder/'resume_check.pt',map_location='cpu',weights_only=False)
                    del cap,opt,model;gc.collect();torch.cuda.empty_cache()
                    model=GeometryModel(**model_config('simple')).cuda();model.load_state_dict(saved['model'])
                    opt=optimizer_for(model,arm);opt._e92_phase=arm;opt.load_state_dict(saved['optimizer'])
                    progress=saved['progress'];scheduler=PlateauController(opt,arm);scheduler.load_state_dict(saved['scheduler'])
                    restore_rng(saved['rng']);cap=build_capture(model,opt,pack,saved['projection']);same(saved['rng'],rng_state())
            assert cap.projection.steps==36006
            final={'model':{k:v.cpu().clone() for k,v in model.state_dict().items()},'optimizer':copy.deepcopy(opt.state_dict()),
                   'rng':rng_state(),'scheduler':scheduler.state_dict(),'progress':progress,'projection':cap.projection.summary()}
            del cap,opt,model;gc.collect();torch.cuda.empty_cache();return final
        control=trajectory('C');resumed=trajectory('C',True);same(control,resumed)
        enhanced=trajectory('G');same(control['rng'],enhanced['rng'])
        assert any(not torch.equal(control['model'][k],enhanced['model'][k]) for k in control['model'])
        result['actual_updates']='C6 equals C2+restore+4 bit-exact model/Adam/RNG/projection/progress; G6 differs as intended; C/G RNG identical after equal steps'
        del control,resumed,enhanced;gc.collect();torch.cuda.empty_cache()
        model,opt,parent=make('C');cap=build_capture(model,opt,pack,parent['projection'])
        old_model=copy.deepcopy(model.state_dict());old_opt=copy.deepcopy(opt.state_dict());old_projection=cap.projection.summary()
        bad={k:v.clone() for k,v in g.items()};bad['target'][0,:,2]=100.
        try:cap.replay(bad)
        except RuntimeError:pass
        else:raise AssertionError('Explosion not rejected')
        assert cap.max_case>=1000 and cap.values['total']<1000
        same(model.state_dict(),old_model);same(opt.state_dict(),old_opt);same(cap.projection.summary(),old_projection)
        bad['target'].fill_(float('nan'))
        try:cap.replay(bad)
        except RuntimeError:pass
        else:raise AssertionError('NaN not rejected')
        same(model.state_dict(),old_model);same(opt.state_dict(),old_opt);same(cap.projection.summary(),old_projection)
        result['guards']='per-case finite explosion and nonfinite rejected before inherited Adam; all states/counters unchanged'
        from .dataset_view import ReferenceLayouts
        assert len(ReferenceLayouts('train'))==32000
        for role in ('test','local_test','tail_test'):
            try:ReferenceLayouts(role)
            except PermissionError:pass
            else:raise AssertionError('Sealed role accepted')
        verify_parent();result.update(status='passed',train_cases=ids,parameters=5829939,parent_hashes=protected['protected_artifacts'],sources=sources())
        snapshot(folder/'source_snapshot.zip',result)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-begin;charge_preflight(result['elapsed_seconds'])
        save(folder/'results.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','parent_hashes')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
