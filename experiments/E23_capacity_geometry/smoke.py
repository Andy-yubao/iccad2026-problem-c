"""Five real cases only: E10 geometry gradients, batch32, LR4, resume and repair."""
import argparse
import json
from pathlib import Path
import torch
from .io import HERE, save, checkpoint
from .data import load_rows, collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses, weights
from .losses import WEIGHTS
from .train import seed_all, PlateauController, LR_LEVELS, parser, run


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); output=args.output.resolve()
    if not output.is_relative_to(HERE/'runs'): raise ValueError('Use E23/runs')
    output.mkdir(parents=True,exist_ok=False); seed_all(0)
    train=load_rows('train',indices=[0,1,2],use_cache=False)
    dev=load_rows('dev_monitor500',indices=[0,1],use_cache=False)
    rows=train+dev
    graph,_=collate([rows[i%5] for i in range(32)],'cuda')
    model=GeometryModel().cuda()
    assert sum(p.numel() for p in model.parameters())==1370771
    state=model(inputs_only(graph))
    full=losses(state,graph,None,'geometry',2,include_quality=True)
    stripped=losses(state,graph,None,'geometry',2,include_quality=False)
    assert torch.equal(full['total'],stripped['total'])
    params=list(model.parameters())
    a=torch.autograd.grad(full['total'],params,retain_graph=True)
    b=torch.autograd.grad(stripped['total'],params)
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert all(torch.isfinite(x).all() for x in a)
    assert weights('geometry',3)==WEIGHTS and weights('geometry',200)==WEIGHTS
    reference_free=inputs_only(graph)
    assert 'reference' not in reference_free and 'target' not in reference_free
    try: model(graph)
    except ValueError: pass
    else: raise AssertionError('Network accepted labels')
    # Atomic cache/checkpoint serialization preserves full reference geometry.
    checkpoint(output/'tensor_roundtrip.pt',{'rows':rows})
    restored=torch.load(output/'tensor_roundtrip.pt',weights_only=False)['rows']
    assert all(torch.equal(x['graph']['reference'],y['graph']['reference']) for x,y in zip(rows,restored))
    p=torch.nn.Parameter(torch.tensor(1.)); opt=torch.optim.AdamW([p],lr=LR_LEVELS[0])
    controller=PlateauController(opt)
    for epoch in range(15): controller.step(1.,epoch)
    assert controller.best is None
    tiers=[]
    for epoch in range(15,25):
        controller.step(1.,epoch); tiers.append(opt.param_groups[0]['lr'])
    assert controller.drops==3 and not controller.should_stop()
    assert opt.param_groups[0]['lr']==5e-5
    controller.step(1.,25); assert controller.should_stop()
    clone=PlateauController(opt); clone.load_state_dict(controller.state_dict())
    assert clone.state_dict()==controller.state_dict()
    assert set(tiers)==set(LR_LEVELS)
    settings=parser().parse_args(['--output',str(output/'resume_check'),'--max-hours','.1'])
    run(settings,implementation_check=True,stop_after_steps=1,check_rows=(train,dev))
    settings.resume=True
    run(settings,implementation_check=True,stop_after_steps=2,check_rows=(train,dev))
    progress=json.loads((settings.output/'progress.json').read_text())
    assert progress['step']==2 and progress['postprocessing_calls']==0
    # Use the same two dev cases for the independently copied evaluation path.
    from .evaluation import quality_monitor
    model=model.cpu()
    evaluated=quality_monitor(model,dev,'cpu')
    assert evaluated['summary']['count']==2 and evaluated['summary']['error_count']==0
    from .final_evaluate import freeze_once, validate_dependencies
    from .io import sources
    frozen={'candidate':'fixed','members':[list(r['member']) for r in dev]}
    assert freeze_once(output/'freeze_check',frozen)
    try: freeze_once(output/'freeze_check',{'candidate':'different'},True)
    except ValueError: pass
    else: raise AssertionError('Accepted a different frozen candidate')
    save(output/'freeze_check/results.json',{'status':'complete'})
    assert not freeze_once(output/'freeze_check',frozen,True)
    state=torch.load(settings.output/'checkpoints/latest.pt',weights_only=False,map_location='cpu')
    validate_dependencies(state['identity'],sources())
    from datasets.reference_layouts import ReferenceLayouts
    try: ReferenceLayouts('local_test')
    except PermissionError: pass
    else: raise AssertionError('Test was exposed')
    save(output/'results.json',{'status':'passed','unique_real_cases':5,'batch_size_checked':32,
        'parameters':1370771,'geometry_gradient_equal_without_unused_occupancy':True,
        'lr_levels':list(LR_LEVELS),'resume_steps':progress['step'],
        'reference_roundtrip':True,'label_input_rejected':True,'frozen_candidate_guard':True,
        'evaluation_cases':2,'evaluation_errors':evaluated['summary']['error_count'],
        'evaluation_hard_legal':evaluated['summary']['hard_legal'],'test_read':False})
    print('E23 five-case checks passed',flush=True)


if __name__=='__main__': main()
