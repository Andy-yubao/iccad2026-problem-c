"""Only the same five E62 training cases: Loss, CUDA and exact real resume."""
import gc
import time
import torch
from .io import HERE, ROOT, save, sha, sources
from .initialization import load, metadata, verify_protected
from .arms import ARMS, model_config
from .model import GeometryModel
from .batches import inputs_only
from .data import load_rows, collate
from .loss import losses, softened_center
from .loss_reference import components as original_components, WEIGHTS
from .train import seed_all, parser, run
from .cuda_step import CapturedStep


def same(a, b, path='state'):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a,b):
            raise AssertionError(path+' differs; max abs '+str(float((a-b).abs().max())))
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            same(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a) == len(b),path
        for i,(x,y) in enumerate(zip(a,b)):
            same(x,y,path+'/'+str(i))
    else:
        assert a == b,path


def clear():
    from . import data
    data._resident.clear()
    gc.collect(); torch.cuda.empty_cache()


def main():
    started=time.perf_counter(); seed_all(0); protected=verify_protected()
    assert sha(HERE/'loss_reference.py') == sha(ROOT/'experiments/E42_joint_regularization/losses.py')
    ids=[1683,10027,5474,9793,1814]
    rows=load_rows('train',ids,use_cache=False); batch=[rows[i%5] for i in range(32)]
    graph,_=collate(batch,'cuda')
    model=GeometryModel(**model_config('control')).cuda().eval()
    source=load(model,model_config('control'))
    assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in model.state_dict().items())
    output=model(inputs_only(graph)).detach().requires_grad_()
    original=original_components(output,graph)
    original_total=sum(WEIGHTS[k]*original[k].mean() for k in WEIGHTS)
    current=losses(output,graph,None,'control',0)
    torch.testing.assert_close(current['total'],original_total,rtol=1e-6,atol=1e-7)
    current_grad=torch.autograd.grad(current['total'],output,retain_graph=True)[0]
    original_grad=torch.autograd.grad(original_total,output,retain_graph=True)[0]
    torch.testing.assert_close(current_grad,original_grad,rtol=1e-5,atol=1e-6)
    forward_delta=float((current['total']-original_total).detach().abs())
    gradient_delta=float((current_grad-original_grad).abs().max())
    arm_checks={}
    for arm in ARMS:
        a=losses(output,graph,None,arm,0,include_quality=False)
        b=losses(output,graph,None,arm,200,include_quality=True)
        assert all(torch.equal(a[k],b[k]) for k in a)
        assert torch.isfinite(torch.stack(list(a.values()))).all()
        grad=torch.autograd.grad(a['total'],output,retain_graph=True)[0]
        assert torch.isfinite(grad).all()
        assert float(a['center_objective']) > 0
        if arm=='weight075':
            torch.testing.assert_close(a['total'],current['total']-.25*current['center_local'],rtol=1e-6,atol=1e-7)
        expected=float(a['total'].detach())
        # Capture at eval/lr0 checks exact initial weights, finite objective and no warmup updates.
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=3e-4)
        captured=CapturedStep(model,optimizer,graph,arm); captured.replay()
        assert abs(float(captured.values['total'])-expected)<1e-6
        assert not optimizer.state or all(int(s['step'])==1 for s in optimizer.state.values())
        assert all(torch.equal(v.cpu(),source['model'][k]) for k,v in model.state_dict().items())
        arm_checks[arm]={'total':expected,'finite_output_gradient':True,'positive_center_anchor':True,
                         'loss_fixed_across_epochs':True,'cuda_capture_no_weight_advance':True}
        del captured,optimizer,a,b,grad
    e=torch.tensor([-.3,-.16,-.08,-.02,.02,.08,.16,.3],device='cuda',requires_grad=True)
    soft_grad=torch.autograd.grad(softened_center(e).sum(),e)[0]
    base_grad=e.detach().clamp(-.1,.1)/.1
    assert (soft_grad.abs()>=.5*base_grad.abs()-1e-6).all()
    assert (soft_grad.abs()<=base_grad.abs()+1e-6).all()
    assert ((soft_grad-base_grad).abs()[e.detach().abs()>=.15]<1e-6).all()
    del output,original,original_total,current,current_grad,original_grad,model,graph,source
    clear()
    # Shared real Dropout/Adam path; interruption after update1 then next update2
    # must equal uninterrupted2 bit-for-bit (parameters, Adam, scheduler, RNG).
    for arm in ARMS:
        out=HERE/f'runs/20261027_preflight_{arm}_resumed'
        args=parser().parse_args(['--arm',arm,'--output',str(out),'--max-hours','.1','--max-epochs','2'])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        one=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        clear(); args.resume=True
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        resumed=torch.load(out/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        clear(); args.resume=False; args.output=HERE/f'runs/20261027_preflight_{arm}_uninterrupted'
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        uninterrupted=torch.load(args.output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert one['progress']['step']==1 and resumed['progress']['step']==uninterrupted['progress']['step']==2
        for key in ('model','optimizer','scheduler','rng'):
            if key=='rng':
                for part in ('torch','cuda'):
                    same(resumed[key][part],uninterrupted[key][part],key+'/'+part)
            else:
                same(resumed[key],uninterrupted[key],key)
        assert resumed['elapsed_seconds']>one['elapsed_seconds']
        assert one['identity']==resumed['identity']==uninterrupted['identity']
        assert all(int(s['step'])==2 for s in resumed['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in resumed['model'].values())
        arm_checks[arm].update(actual_dropout_adam_resume_bit_exact=True,optimizer_steps=[1,2],budget_preserved=True)
        del one,resumed,uninterrupted;clear()
    assert verify_protected()==protected
    save(HERE/'runs/20261027_preflight/results.json',{'status':'passed','unique_train_cases':5,'indices':ids,
        'initialization':metadata(),'E42_control_loss_abs_delta':forward_delta,'E42_control_output_gradient_max_abs_delta':gradient_delta,
        'softband_keeps_half_gradient_and_restores_large_error_slope':True,'arms':arm_checks,
        'protected_checkpoints_unchanged':protected,'sources':sources(),
        'formal_training_started':False,'test_read':False,'seconds':time.perf_counter()-started})
    print('E63 five-case preflight passed for all arms',flush=True)


if __name__=='__main__':
    main()
