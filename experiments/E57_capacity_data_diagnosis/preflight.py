"""Five real train cases only; test all Loss branches, wider capture and block resume."""
import time
import numpy as np
import torch
from .io import HERE, ROOT, save, sha
from .arms import ARMS, model_config, weight_decay
from .data import load_rows, collate
from .batches import inputs_only
from .model import GeometryModel
from .loss import losses
from .loss_reference import components as original_components, WEIGHTS
from .train import seed_all, parser, run
from .cuda_step import CapturedStep
from .subsets import indices
from .protection import verify


def main():
    started=time.perf_counter();protected=verify();seed_all(0)
    for seed in (0,1):
        sets=[indices(n,seed) for n in (2048,4096,8192,16000)]
        assert all(set(x)<set(y) for x,y in zip(sets,sets[1:]))
    assert sha(HERE/'model.py')==sha(ROOT/'experiments/E42_joint_regularization/model.py')
    ids=[50,36,22,62,45];rows=load_rows('train',ids,use_cache=False)
    batch=[rows[i%5] for i in range(32)];graph,_=collate(batch,'cuda')
    checks={};initial96=None
    for arm in ARMS:
        seed_all(0)
        model=GeometryModel(**model_config(arm)).cuda().eval()
        if arm=='reference96':initial96={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if arm=='reference96_noreg':assert all(torch.equal(v.cpu(),initial96[k]) for k,v in model.state_dict().items())
        state=model(inputs_only(graph)).detach().requires_grad_()
        a=losses(state,graph,None,arm,0,include_quality=False)
        b=losses(state,graph,None,arm,200,include_quality=True)
        assert torch.equal(a['total'],b['total'])
        if arm.startswith('geometry'):
            original=original_components(state,graph);expected=sum(WEIGHTS[k]*original[k].mean() for k in WEIGHTS)
            ga=torch.autograd.grad(a['total'],state,retain_graph=True)[0]
            gb=torch.autograd.grad(expected,state)[0]
            torch.testing.assert_close(a['total'],expected,rtol=1e-6,atol=1e-7)
            torch.testing.assert_close(ga,gb,rtol=1e-5,atol=1e-6)
            del original,expected,ga,gb
        expected_value=float(a['total'].detach())
        del state,a,b,model
        model=GeometryModel(**model_config(arm)).cuda().eval()
        # A new model has new RNG draws, so compute its own capture reference.
        with torch.no_grad():expected_value=float(losses(model(inputs_only(graph)),graph,None,arm,0)['total'])
        opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=weight_decay(arm))
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        captured=CapturedStep(model,opt,graph,arm);captured.replay()
        assert abs(float(captured.values['total'])-expected_value)<1e-6
        assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
        checks[arm]={'parameters':sum(p.numel() for p in model.parameters()),'capture_passed':True,'weight_decay':weight_decay(arm)}
        del model,opt,captured,before
        torch.cuda.empty_cache()
    del graph
    resumed=[]
    for arm,size,block in [('reference96',32,64),('geometry160',16000,500)]:
        output=HERE/'runs'/('20261026_preflight_'+arm)
        args=parser().parse_args(['--output',str(output),'--arm',arm,'--dataset-size',str(size),'--block-steps',str(block),'--max-hours','.15','--max-epochs','2'])
        run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=1)
        one=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        args.resume=True;run(args,implementation_check=True,check_rows=(batch,batch),stop_after_steps=2)
        two=torch.load(output/'checkpoints/latest.pt',map_location='cpu',weights_only=False)
        assert one['progress']['step']==1 and two['progress']['step']==2
        assert len(two['progress']['order'])==block*32 and two['progress']['cursor']==64
        assert two['identity']==one['identity'] and two['identity']['implementation_check']
        assert two['elapsed_seconds']>one['elapsed_seconds']
        assert all(int(v['step'])==2 for v in two['optimizer']['state'].values())
        assert all(torch.isfinite(v).all() for v in two['model'].values())
        assert np.array_equal(one['progress']['order'],two['progress']['order'])
        resumed.append({'arm':arm,'block_steps':block,'optimizer_steps':2,'resume_passed':True})
    assert verify()==protected
    save(HERE/'runs/20261026_preflight/results.json',{'status':'passed','unique_train_cases':5,'indices':ids,
        'nested_memberships_verified':True,'fixed_loss_checks':checks,'actual_training_resume_checks':resumed,
        'same96_initial_weights_with_and_without_regularization':True,
        'protected_checkpoints_unchanged':protected,'formal_training_started':False,'test_read':False,
        'seconds':time.perf_counter()-started})
    print('E57 five-case preflight passed',flush=True)


if __name__=='__main__':main()
