"""At most five real train cases: objectives, real Adam, handoff and exact resume."""
import copy,gc,json,time
import numpy as np
import torch
from .io import HERE,ROOT,save,sha,sources,snapshot,checkpoint
from .initialization import verify_protected
from .sentinels import select
from .data import load_rows,register_resident
from .resident import ResidentRows
from .batches import inputs_only
from .model import GeometryModel
from .arms import model_config
from .loss import per_case
from .simple_loss import components
from .loss_config import COEFFICIENTS
from .schedule import PlateauController
from .train import seed_all,rng_state,restore_rng,optimizer_for,build_capture,fresh_progress,handoff,monitor,save_P_terminal
from .budget import charge_preflight,global_charged_seconds

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

def schedule_checks():
    p=torch.nn.Parameter(torch.zeros(()));opt=torch.optim.AdamW([p],lr=.001)
    s=PlateauController(opt,'P')
    for epoch in range(15):s.step(1.,epoch)
    assert s.best is None and s.drops==0
    for epoch in range(15,21):s.step(1.,epoch)
    assert s.drops==1 and not s.transition_pending
    saved=copy.deepcopy(s.state_dict());s=PlateauController(opt,'P');s.load_state_dict(saved)
    s.step(1.,21)
    assert s.transition_pending and s.drops==1 and opt.param_groups[0]['lr']==.00025
    f=PlateauController(opt,'F');f.step(1.,0)
    for epoch in range(1,4):f.step(1.,epoch)
    assert f.drops==1 and opt.param_groups[0]['lr']==.0001
    for epoch in range(4,7):f.step(1.,epoch)
    assert f.drops==2 and opt.param_groups[0]['lr']==.00005 and not f.should_stop()
    f.step(1.,7);assert not f.should_stop()
    f.step(1.,8);assert f.should_stop()
    g=PlateauController(opt,'F');g.step(1.,0);g.step(.9999,1)
    assert g.lr_bad==1 and g.best==1.
    g.step(.998,2);assert g.lr_bad==0 and g.best==.998
    return {'P_first_drop_epoch':18,'P_switch_epoch':21,'F_drops':[3,6],'F_stop_epoch':8,'tiny_improvement_does_not_reset':True}

def verify():
    result=json.loads((HERE/'runs/preflight/results.json').read_text(encoding='utf-8'))
    if result['status']!='passed' or result['sources']!=sources():raise ValueError('Matching current-source preflight required')
    verify_protected()
    return result

def main():
    begin=time.perf_counter();result={'status':'error','test_read':False}
    output=HERE/'runs/preflight';output.mkdir(parents=True,exist_ok=True)
    trial=output/str(time.time_ns())
    trial.mkdir()
    try:
        if global_charged_seconds()>=64800:raise RuntimeError('18h budget exhausted')
        seed_all(0);result['protected']=verify_protected();result['schedule']=schedule_checks()
        rows=select();meta=json.loads((HERE/'runs/sentinels.json').read_text(encoding='utf-8'))
        ids=[r['index'] for r in meta['records']]
        raw=load_rows('train',ids,use_cache=False)
        for a,b in zip(rows,raw):
            assert a['member']==b['member']
            for key,value in a['graph'].items():
                if isinstance(value,torch.Tensor):same(value,b['graph'][key])
        del raw
        for name,source in [('model.py','E74_collision_32k_posttrain'),('simple_loss.py','E90_auxiliary_weight_placer')]:
            assert sha(HERE/name)==sha(ROOT/'experiments'/source/name)
        # Only these five actual layouts are ever passed to a model in this check.
        pack=ResidentRows(rows,'cuda',max_nodes=120);register_resident(pack)
        g=pack.batch([i%5 for i in range(32)])
        seed_all(0);model=GeometryModel(**model_config('simple')).cuda().eval()
        assert all(p.requires_grad for p in model.parameters()) and sum(p.numel() for p in model.parameters())==5829939
        with torch.no_grad():
            state=model(inputs_only(g));reference={}
            for phase in ('P','F'):
                value=per_case(state,g,phase);original=components(state,g,COEFFICIENTS[phase]);same(value,original)
                expected=value['reference']+COEFFICIENTS[phase]['overlap']*value['overlap_total']+COEFFICIENTS[phase]['task']*value['task']
                same(value['total'],expected);assert torch.isfinite(expected).all() and expected.max()<1000
                reference[phase]={'mean':float(expected.mean()),'max':float(expected.max())}
        result['initial_loss']=reference
        del model,state;gc.collect();torch.cuda.empty_cache()

        def trajectory(resume=False,interrupt_pending=False):
            seed_all(0);m=GeometryModel(**model_config('simple')).cuda();opt=optimizer_for(m,'P');opt._e92_phase='P'
            cap=build_capture(m,opt,pack);progress=fresh_progress(5)
            for i in range(4):
                cap.replay(g);progress['step']+=1
                if i==1 and not (trial/'checkpoints/P_best.pt').exists():
                    progress['best_epoch']['P']=2
                    checkpoint(trial/'checkpoints/P_best.pt',{'model':m.state_dict(),'progress':copy.deepcopy(progress)})
                if i==1:progress['best_epoch']['P']=2
            progress['epoch']=21;progress['phase_epoch']=21
            end_rng=rng_state();projection=cap.projection.summary()
            terminal=save_P_terminal(trial,{'model':m.state_dict(),'optimizer':opt.state_dict(),
                'rng':end_rng,'progress':copy.deepcopy(progress),'scheduler':PlateauController(opt,'P').state_dict()})
            if interrupt_pending:
                # Model/RNG restored at persisted P boundary; no skipped/repeated update.
                saved=torch.load(terminal,map_location='cpu',weights_only=False)
                m.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
            del cap;gc.collect();torch.cuda.empty_cache()
            progress,opt,s=handoff(m,progress,None,trial)
            same(end_rng,rng_state());assert not opt.state and s.phase=='F' and s.best is None
            assert progress['epoch']==21 and progress['step']==4 and progress['phase_epoch']==0
            cap=build_capture(m,opt,pack,projection)
            base_rng=rng_state();baseline=monitor(m,rows,'F','cuda');same(base_rng,rng_state())
            baseline.pop('inference_seconds')  # Runtime is measured, never asserted bit-identical.
            s.step(baseline['total'],0)
            for i in range(2):
                m.train();cap.replay(g);progress['step']+=1;progress['phase_step']+=1
                if resume and i==0:
                    saved={'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),
                           'rng':rng_state(),'projection':cap.projection.summary(),'scheduler':s.state_dict()}
                    del cap,opt,m;gc.collect();torch.cuda.empty_cache()
                    m=GeometryModel(**model_config('simple')).cuda();m.load_state_dict(saved['model'])
                    opt=optimizer_for(m,'F');opt._e92_phase='F';opt.load_state_dict(saved['optimizer'])
                    s=PlateauController(opt,'F');s.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
                    cap=build_capture(m,opt,pack,saved['projection'])
                    same(saved['rng'],rng_state())
            final={'model':{k:v.cpu().clone() for k,v in m.state_dict().items()},
                   'optimizer':copy.deepcopy(opt.state_dict()),'rng':rng_state(),
                   'progress':progress,'scheduler':s.state_dict(),'projection':cap.projection.summary(),'baseline':baseline}
            del cap,opt,m;gc.collect();torch.cuda.empty_cache()
            return final
        uninterrupted=trajectory();resumed=trajectory(resume=True,interrupt_pending=True)
        same(uninterrupted,resumed)
        result['recovery']='P4 + handoff to P best2 + F2 == same handoff with F1+resume+F1, bit-exact model/Adam/RNG/projection/progress/monitor'
        del uninterrupted,resumed;gc.collect();torch.cuda.empty_cache()
        # A finite bad single case is rejected even though the batch mean is <1000.
        seed_all(0);m=GeometryModel(**model_config('simple')).cuda();opt=optimizer_for(m,'P');opt._e92_phase='P'
        cap=build_capture(m,opt,pack);before=copy.deepcopy(m.state_dict())
        bad={k:v.clone() for k,v in g.items()};bad['target'][0,:,2]=100.
        try:cap.replay(bad)
        except RuntimeError:pass
        else:raise AssertionError('Per-case guard did not reject')
        assert float(cap.max_case)>=1000 and float(cap.values['total'])<1000 and float(cap.norm)<10000
        assert not opt.state and cap.projection.steps==0;same(before,m.state_dict())
        bad['target'].fill_(float('nan'))
        try:cap.replay(bad)
        except RuntimeError:pass
        else:raise AssertionError('Nonfinite guard did not reject')
        assert not opt.state and cap.projection.steps==0;same(before,m.state_dict())
        result['guards']='finite single-case catastrophe and nonfinite rejected before Adam; weights/counters unchanged'
        from .dataset_view import ReferenceLayouts
        assert len(ReferenceLayouts('train'))==32000
        for role in ('test','local_test','tail_test'):
            try:ReferenceLayouts(role)
            except PermissionError:pass
            else:raise AssertionError('Test role accepted')
        result.update(status='passed',train_cases=ids,parameters=5829939,all_trainable=True,
                      per_epoch_draws=32000,updates_per_epoch=1000,
                      peak_gpu_allocated=torch.cuda.max_memory_allocated(),sources=sources())
        snapshot(output/'source_snapshot.zip',result)
    except BaseException as error:
        result['error']=repr(error);raise
    finally:
        seconds=time.perf_counter()-begin
        charge_preflight(seconds);result['elapsed_seconds']=seconds
        save(output/'results.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','protected')},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
