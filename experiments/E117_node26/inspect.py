"""One necessary check on five existing E114 training members; no dev/test."""
import json
import time
import traceback
import torch
from . import train
from .io import OWNER, save, sources
from .model import GeometryModel, node_geometry, AXIS_SWAP
from .original_model import GeometryModel as OriginalModel
from .data import load_rows
from .resident import ResidentRows
from .batches import inputs_only, decode
from .augmentation import axis_swap
from .cuda_step import CapturedStep

def main():
    begin=time.perf_counter()
    result=dict(status='error',unique_train_cases=5,dev_payloads=0,test_read=False,
                optimizer_updates=0,formal_training_started=False,resume_tested=False)
    try:
        plan=json.loads((OWNER/'plan.json').read_text(encoding='utf-8'))
        train.seed_all(0)
        original=OriginalModel(**train.MODEL)
        original_rng=torch.get_rng_state().clone()
        train.seed_all(0)
        model=GeometryModel(**train.MODEL)
        assert torch.equal(original_rng,torch.get_rng_state())
        shared=[k for k in original.state_dict() if '.bias.' not in k]
        for k in shared: assert torch.equal(original.state_dict()[k],model.state_dict()[k]),k
        assert not any('bias.0' in k for k in model.state_dict())
        count=sum(p.numel() for p in model.parameters())
        assert count==plan['parameters'] and all(p.requires_grad for p in model.parameters())
        del original
        rows=load_rows('train',plan['preflight_train_indices'])
        assert len({tuple(r['member']) for r in rows})==5
        pack=ResidentRows(rows,'cuda',max_nodes=120)
        batch=pack.batch([i%5 for i in range(32)])
        public=inputs_only(batch)
        assert not {'target','reference','pin_coordinates'} & set(public)
        assert (public['constraints'][...,2]==0).all() and (public['mask'][public['valid']]==1).all()
        model.cuda().eval()
        with torch.no_grad():
            state=model.head(model.input(public['features']))
            phi=node_geometry(state,public)
            swapped=axis_swap({**batch,'axis_swap':torch.ones(32,dtype=torch.bool,device='cuda')})
            state_t=torch.stack((state[...,1],state[...,0],-state[...,2]),-1)
            actual=node_geometry(state_t,inputs_only(swapped))
            expected=phi[...,list(AXIS_SWAP)].clone();expected[...,12]*=-1
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            assert phi.shape==(32,120,26) and (phi[~public['valid']]==0).all()
            empty={**public,'adjacency':torch.zeros_like(public['adjacency']),'group':torch.zeros_like(public['group'])}
            assert (node_geometry(state,empty)[...,16:]==0).all()
            out=model(public)
            assert out.shape==(12,32,120,3) and torch.isfinite(out).all()
            locked=public['valid'] & public['constraints'][...,:2].bool().any(-1)
            assert locked.any()
            changed=out[-1].clone();changed[...,0]+=.5;changed[...,2]+=.2
            assert (decode(changed,public)[locked]-decode(out[-1],public)[locked]).abs().sum()>0
        model.train()
        optimizer=train.optimizer_for(model)
        torch.cuda.reset_peak_memory_stats()
        cap=CapturedStep(model,optimizer,batch,'node26')
        # The single replay includes real AdamW and the original displacement guard.
        before={k:v.detach().clone() for k,v in model.state_dict().items()}
        tick=time.perf_counter();values,norm=cap.replay();torch.cuda.synchronize()
        elapsed=time.perf_counter()-tick
        for layer in model.layers:
            assert layer.geometry_q.weight.grad.abs().sum()>0
            assert layer.geometry_k.weight.grad.abs().sum()>0
            assert layer.geometry_scale.grad.abs().sum()>0
        assert cap.projection.steps==1 and optimizer.state
        assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
        result.update(status='passed',parameters=count,common_initial_tensors=len(shared),
                      global_CPU_RNG_preserved=True,node26_axis_swap_verified=True,padding_zero=True,
                      no_pair_MLP=True,all_module_draft_preserved=True,geometry_gradients_all12=True,
                      optimizer_updates=1,disposable_update_only=True,
                      total=float(values['total']),max_case_loss=float(cap.max_case),gradient_norm=float(norm),
                      one_replay_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      members=[list(r['member']) for r in rows],sources=sources(),
                      gpu=torch.cuda.get_device_name(0),torch=str(torch.__version__))
    except BaseException as error:
        result.update(error=repr(error),traceback=traceback.format_exc());raise
    finally:
        result['elapsed_seconds']=time.perf_counter()-begin
        save(OWNER/'preflight.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('sources','traceback')},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
