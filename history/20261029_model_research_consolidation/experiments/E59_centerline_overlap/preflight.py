"""Five fixed train members: parity, Dropout, staged capture and exact resume."""
import json
import time
import torch
from .io import HERE,ROOT,sha,save
from .data import load_rows,collate
from .batches import inputs_only
from .resident import ResidentRows
from .arms import model_config
from .model import GeometryModel
from .loss import losses
from .loss_reference import components as original_components,WEIGHTS
from .train import seed_all,parser,run
from .cuda_step import CapturedStep


def main():
    begin=time.perf_counter();seed_all(0)
    ids=json.loads((HERE/'loss_scale.json').read_text(encoding='utf-8'))['train_indices']
    rows=load_rows('train',ids);batch=[rows[i%5] for i in range(32)]
    cpu,_=collate(batch,'cpu');gpu,_=collate(batch,'cuda')
    resident=ResidentRows(batch,'cuda',120).batch(list(range(32)))
    assert all(torch.equal(cpu[k].to('cuda'),gpu[k]) and torch.equal(gpu[k],resident[k]) for k in cpu)
    results={}
    for arm in ('original96','centerline96','repair160'):
        print('preflight arm '+arm,flush=True)
        seed_all(0)
        model=GeometryModel(**model_config(arm)).cuda().eval()
        state=model(inputs_only(gpu)).detach().requires_grad_()
        a=losses(state,gpu,None,arm,0,include_quality=False)
        b=losses(state,gpu,None,arm,200,include_quality=True)
        assert torch.equal(a['total'],b['total'])
        if arm=='original96':
            old=original_components(state,gpu);expected=sum(WEIGHTS[k]*old[k].mean() for k in WEIGHTS)
            ga=torch.autograd.grad(a['total'],state,retain_graph=True)[0]
            gb=torch.autograd.grad(expected,state,retain_graph=True)[0]
            torch.testing.assert_close(a['total'],expected,rtol=1e-6,atol=1e-7)
            torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-6)
        del state,a,b,model
        if arm=='original96':del old,expected,ga,gb
        seed_all(0)
        model=GeometryModel(**model_config(arm)).cuda().eval()
        with torch.no_grad():expected_value=float(losses(model(inputs_only(gpu)),gpu,None,arm,0)['total'])
        opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=.0003)
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        capture=CapturedStep(model,opt,gpu,arm);capture.replay(cpu)
        assert abs(float(capture.values['total'])-expected_value)<1e-5
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        results[arm]={'capture_eval_loss':expected_value,'parameters':sum(v.numel() for v in model.parameters()),'exact_input_staging':True}
        del model,opt,capture,before
        torch.cuda.empty_cache()
    # Actual entry-point interruption/recovery with Dropout and Adam, then
    # compare next update with the uninterrupted trajectory.
    for arm in ('repair160','centerline96'):
        folder=HERE/'runs'/('20261027_preflight_'+arm)
        full=folder.with_name(folder.name+'_uninterrupted')
        argv=['--arm',arm,'--dataset-size','16000','--max-hours','.15','--max-epochs','2','--engine','staged_cuda_graph']
        args=parser().parse_args(['--output',str(folder),*argv])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        one=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        two=torch.load(folder/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        full_args=parser().parse_args(['--output',str(full),*argv])
        run(full_args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        direct=torch.load(full/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert one['progress']['step']==1 and two['progress']['step']==direct['progress']['step']==2
        assert two['elapsed_seconds']>one['elapsed_seconds']
        assert all(torch.equal(two['model'][k],direct['model'][k]) for k in two['model'])
        assert any(not torch.equal(two['model'][k],one['model'][k]) for k in two['model'])
        for k,v in two['optimizer']['state'].items():
            for name,t in v.items():assert torch.equal(t,direct['optimizer']['state'][k][name])
        assert all(int(x['step'])==2 for x in two['optimizer']['state'].values())
        assert two['progress']['cursor']==64 and two['identity']==one['identity']
        results[arm]['actual_entry_exact_resume']=True
    # All test entry points reject before payload read.
    rejected=[]
    for role in ('test','local_test','tail_test'):
        try:load_rows(role)
        except PermissionError:rejected.append(role)
    assert len(rejected)==3
    save(HERE/'runs/20261027_preflight/results.json',{'status':'passed','unique_train_cases':5,'indices':ids,
        'checks':results,'sealed_roles_rejected':rejected,'test_read':False,'formal_training_started':False,
        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'seconds':time.perf_counter()-begin})
    print('E59 five-case preflight passed',flush=True)


if __name__=='__main__':main()
