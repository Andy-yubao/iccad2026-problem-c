"""Five-case isolation, complementary-gradient and CUDA/save/resume checks."""
import gc,json,time
import torch
from .io import HERE,save,sha
from .data import load_rows,collate
from .batches import inputs_only
from .model import GeometryModel
from .arms import model_config
from .loss import losses
from .parent_loss import losses as parent_loss
from .initialization import load as initialize,SOURCE,DIGEST
from .train import seed_all,parser,run
from .cuda_step import CapturedStep

def main():
    started=time.perf_counter();seed_all(0);ids=[50,36,22,62,45]
    rows=load_rows('train',ids,use_cache=False);fixtures=[rows[i%5] for i in range(32)]
    graph,_=collate(fixtures,'cuda');model=GeometryModel(**model_config('no_reference')).cuda().eval()
    origin=initialize(model,model_config('no_reference'))
    with torch.no_grad():state=model(inputs_only(graph))
    state=state.detach().requires_grad_(True)
    a=losses(state,graph,None,'no_reference',0.,include_quality=False)
    b=losses(state,graph,None,'reference_only',100.,include_quality=True)
    old=parent_loss(state,graph,None,'geometry',0.)
    loss_delta=abs(float((a['total']+b['total']-old['total']).detach()))
    ga=torch.autograd.grad(a['total']+b['total'],state,retain_graph=True)[0]
    go=torch.autograd.grad(old['total'],state,retain_graph=True)[0]
    grad_delta=float((ga-go).abs().max());assert loss_delta<2e-6 and grad_delta<2e-5
    stripped=inputs_only(graph)
    poison=dict(graph,target=torch.full_like(graph['target'],float('nan')),reference=torch.full_like(graph['reference'],float('nan')))
    for g in (stripped,poison):
        result=losses(state,g,None,'no_reference',50.)
        assert torch.equal(result['total'],a['total'])
        assert torch.equal(torch.autograd.grad(result['total'],state,retain_graph=True)[0],torch.autograd.grad(a['total'],state,retain_graph=True)[0])
    ref_poison=dict(graph,mib=torch.full_like(graph['mib'],float('nan')),constraints=torch.zeros_like(graph['constraints']))
    assert torch.equal(losses(state,ref_poison,None,'reference_only',0.)['total'],b['total'])
    altered=dict(graph,target=graph['target']+.3,reference=graph['reference']*1.2)
    assert abs(float((losses(state,altered,None,'reference_only',0.)['total']-b['total']).detach()))>1e-4
    report={'loss_decomposition_abs_delta':loss_delta,'state_gradient_decomposition_max_abs_delta':grad_delta,
            'no_reference_deleted_or_nan_labels_same_loss_and_gradient':True,
            'reference_only_ignores_public_boundary_and_mib_objectives':True,
            'reference_only_responds_to_changed_labels':True,
            'initial_objectives':{'no_reference':float(a['total'].detach()),'reference_only':float(b['total'].detach())}}
    for arm in ('no_reference','reference_only'):
        assert torch.equal(losses(state,graph,None,arm,0.,include_quality=False)['total'],losses(state,graph,None,arm,200.,include_quality=True)['total'])
    del model,origin,state,a,b,old,ga,go,result;gc.collect();torch.cuda.synchronize()
    for arm in ('no_reference','reference_only'):
        seed_all(0);model=GeometryModel(**model_config(arm)).cuda().train();initialize(model,model_config(arm))
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=3e-4)
        captured=CapturedStep(model,optimizer,graph,arm);captured.replay()
        assert bool(captured.finite)
        del captured,optimizer,model;gc.collect();torch.cuda.synchronize()
        output=HERE/'runs/20261026_check5'/arm
        args=parser().parse_args(['--arm',arm,'--output',str(output),'--max-hours','.1','--engine','cuda_graph'])
        run(args,implementation_check=True,stop_after_steps=1,check_rows=(fixtures,rows))
        first=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert first['progress']['step']==1 and first['identity']['initialization']['sha256']==DIGEST
        args.resume=True
        run(args,implementation_check=True,stop_after_steps=2,check_rows=(fixtures,rows))
        second=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert second['progress']['step']==2 and second['elapsed_seconds']>=first['elapsed_seconds']
        assert second['optimizer']['state'] and second['identity']==first['identity']
        assert any(not torch.equal(first['model'][k],second['model'][k]) for k in first['model'])
        report[arm]={'capture_train_mode_finite':True,'real_entry_save_resume_steps':[1,2],
                     'same_initial_checkpoint':True,'optimizer_and_budget_preserved':True}
    assert sha(SOURCE)==DIGEST
    report.update(status='passed',seconds=time.perf_counter()-started,unique_real_train_cases=5,indices=ids,
                  formal_training_started=False,implementation_fixtures_only=True,test_read=False)
    save(HERE/'runs/20261026_check5/results.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
