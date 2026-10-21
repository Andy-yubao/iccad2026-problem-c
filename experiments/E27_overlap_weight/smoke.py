"""Five cases only: single-factor gradients, frozen selection, resume, legality."""
import argparse,json
from pathlib import Path
import torch
from .io import ROOT,HERE,sha,save,checkpoint,sources
from .data import load_rows,collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses,weights
from .losses import objective_loss,WEIGHTS
from .train import seed_all,PlateauController,LR_LEVELS,parser,run,conflict_diagnostics,selection

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    out=args.output.resolve();assert out.is_relative_to(HERE/'runs');out.mkdir(parents=True,exist_ok=False)
    seed_all(0);device='cuda' if torch.cuda.is_available() else 'cpu'
    train=load_rows('train',indices=[0,1,2],use_cache=False)
    dev=load_rows('dev_monitor500',indices=[0,1],use_cache=False);rows=train+dev
    model=GeometryModel().to(device);assert sum(p.numel() for p in model.parameters())==497299
    assert sha(HERE/'model.py')==sha(ROOT/'experiments/E25_integrated_baseline/model.py')
    graph,_=collate([rows[i%5] for i in range(32)],device)
    state=model(inputs_only(graph));params=list(model.parameters());epoch_checks=[]
    for epoch in [0.,1.,2.]:
        full=losses(state,graph,None,'geometry',epoch,include_quality=True)
        small=losses(state,graph,None,'geometry',epoch,include_quality=False)
        assert torch.equal(full['total'],small['total'])
        base=objective_loss(full,'geometry',epoch);extra=.5*min(1.,epoch/2)*full['overlap']
        assert torch.allclose(full['total'],base+extra,rtol=1e-6,atol=1e-7)
        expected=sum(WEIGHTS[k]*full[k] for k in WEIGHTS)+full['occupancy']
        assert torch.allclose(full['repair'],expected,rtol=1e-6,atol=1e-7)
        assert selection(full)==full['repair']
        ga=torch.autograd.grad(full['total'],params,retain_graph=True)
        gb=torch.autograd.grad(base+extra,params,retain_graph=True)
        gs=torch.autograd.grad(small['total'],params,retain_graph=True)
        assert all(torch.isfinite(g).all() for g in ga)
        assert all(torch.allclose(a,b,rtol=1e-5,atol=1e-6) and torch.equal(a,c) for a,b,c in zip(ga,gb,gs))
        epoch_checks.append({'epoch':epoch,'training_overlap_weight':weights('geometry',epoch)['overlap'],
                             'total':float(full['total'].detach()),'base_total':float(base.detach())})
    assert weights('geometry',2)['overlap']==1 and WEIGHTS['overlap']==.5
    assert 'target' not in inputs_only(graph) and 'reference' not in inputs_only(graph)
    try:model(graph)
    except ValueError:pass
    else:raise AssertionError('Label-containing graph accepted')
    star=conflict_diagnostics(6,[(0,1),(0,2),(0,3)]);separate=conflict_diagnostics(6,[(0,1),(2,3),(4,5)])
    assert star['overlap_pairs']==separate['overlap_pairs']==3
    assert star['max_conflict_degree']==3 and separate['max_conflict_degree']==1
    assert star['largest_conflict_component']==4 and separate['largest_conflict_component']==2
    p=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([p],lr=LR_LEVELS[0]);ctl=PlateauController(opt)
    for epoch in range(15):ctl.step(1.,epoch)
    assert ctl.best is None
    for epoch in range(15,25):ctl.step(1.,epoch)
    assert ctl.drops==3 and not ctl.should_stop();ctl.step(1.,25);assert not ctl.should_stop()
    ctl.step(1.,26);assert ctl.should_stop()
    settings=parser().parse_args(['--output',str(out/'resume_check'),'--max-hours','.1','--device',device])
    run(settings,implementation_check=True,stop_after_steps=1,check_rows=(train,dev))
    settings.resume=True;run(settings,implementation_check=True,stop_after_steps=2,check_rows=(train,dev))
    progress=json.loads((settings.output/'progress.json').read_text(encoding='utf-8'))
    assert progress['step']==2 and progress['postprocessing_calls']==0
    assert 'raw_max_conflict_degree_mean' in progress['history'][0]['dev_monitor']
    from .evaluation import quality_monitor
    evaluated=quality_monitor(model.cpu(),dev,'cpu')
    assert evaluated['summary']['count']==2 and evaluated['summary']['error_count']==0 and evaluated['summary']['hard_legal']==2
    from .final_evaluate import freeze_once,validate_dependencies
    frozen={'candidate':'same','members':[list(r['member']) for r in dev]};assert freeze_once(out/'freeze_check',frozen)
    try:freeze_once(out/'freeze_check',{'candidate':'different'},True)
    except ValueError:pass
    else:raise AssertionError('Second candidate accepted')
    save(out/'freeze_check/results.json',{'status':'complete'});assert not freeze_once(out/'freeze_check',frozen,True)
    latest=torch.load(settings.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
    validate_dependencies(latest['identity'],sources())
    from .cache import verify
    cache_roles={role:len(verify(role)['members']) for role in ['train','dev']}
    from datasets.reference_layouts import ReferenceLayouts
    try:ReferenceLayouts('local_test')
    except PermissionError:pass
    else:raise AssertionError('Sealed tests exposed')
    save(out/'results.json',{'status':'passed','unique_real_cases':5,'parameters':497299,'batch_size':32,
        'loss_and_gradient_single_factor_verified':True,'selection_weights_unchanged':True,
        'epoch_checks':epoch_checks,'crowding_graph_counterexample':{'star':star,'separate':separate},
        'LR4_early_stop_verified':True,'resume_steps':2,'evaluation_hard_legal':2,
        'cache_members_verified':cache_roles,'formal_training_started':False,'test_read':False})
    print('E27 five-case checks passed',flush=True)
if __name__=='__main__':main()
