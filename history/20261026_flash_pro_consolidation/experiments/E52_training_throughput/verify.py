"""Five-case parity, padding and stochastic CUDA-graph restart checks."""
import copy
import json
import time
import torch
from .io import ROOT,HERE,save,sha
from .train import seed_all,rng_state,restore_rng,monitor_losses
from .data import load_rows,collate,register_resident
from .resident import ResidentRows
from .cuda_step import CapturedStep
from .model import GeometryModel
from .loss import losses
from .reference_loss import losses as original_losses
from .batches import inputs_only


def make_model(state,training=True):
    m=GeometryModel(**state['model_config']).cuda()
    m.load_state_dict(state['model']);m.train(training)
    return m


def params_diff(a,b):
    return max(float((p-q).abs().max()) for p,q in zip(a.parameters(),b.parameters()))


def main():
    started=time.perf_counter();seed_all(0)
    evidence=json.loads((HERE/'runs/20261025_profile5/results.json').read_text(encoding='utf-8'))
    rows=load_rows('train',evidence['indices'],use_cache=False)
    selected=[rows[i%5] for i in range(32)]
    batch,_=collate(selected,'cuda')
    pack=ResidentRows(rows,'cuda',max_nodes=120)
    ids=torch.tensor([i%5 for i in range(32)],device='cuda')
    padded=pack.batch(ids)
    checkpoint=ROOT/'experiments/E51_module_overlap/runs/20261025_rectangle_w96d12_unified_v2_main_seed0/checkpoints/best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m=make_model(state,False)
    with torch.no_grad():
        original=original_losses(m(inputs_only(batch)),batch,None,'main',0.)
        optimized=losses(m(inputs_only(batch)),batch,None,'main',0.)
        padded_values=losses(m(inputs_only(padded)),padded,None,'main',0.)
    loss_delta=max(float((original[k]-optimized[k]).abs()) for k in original)
    padding_delta=max(float((optimized[k]-padded_values[k]).abs()) for k in optimized)
    assert loss_delta<1e-6,loss_delta
    assert padding_delta<1e-5,padding_delta
    # Same training dropout stream, same original AdamW (not a fused replacement).
    a,b=make_model(state),make_model(state)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-3,weight_decay=3e-4)
    ob=torch.optim.AdamW(b.parameters(),lr=1e-3,weight_decay=3e-4)
    seed_all(7);start_rng=rng_state()
    oa.zero_grad(set_to_none=True)
    va=original_losses(a(inputs_only(batch)),batch,None,'main',0.)
    va['total'].backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.,error_if_nonfinite=True);oa.step()
    restore_rng(start_rng)
    runner=CapturedStep(b,ob,batch);vb,_=runner.replay()
    train_loss_delta=abs(float(va['total'])-float(vb['total']))
    train_update_delta=params_diff(a,b)
    save(HERE/'runs/20261025_profile5/training_parity_partial.json',
         {'training_loss_abs_delta':train_loss_delta,'one_step_parameter_max_abs_delta':train_update_delta})
    assert train_loss_delta<1e-5,train_loss_delta
    assert train_update_delta<1e-6,train_update_delta
    # Resume from an in-memory latest state; no usable training artifact is saved.
    saved_model=copy.deepcopy(b.state_dict());saved_opt=copy.deepcopy(ob.state_dict());saved_rng=rng_state()
    expected_loss=float(runner.replay()[0]['total']);torch.cuda.synchronize()
    c=make_model(state);c.load_state_dict(saved_model)
    oc=torch.optim.AdamW(c.parameters(),lr=1e-3,weight_decay=3e-4);oc.load_state_dict(saved_opt)
    restore_rng(saved_rng)
    resumed=CapturedStep(c,oc,batch)
    resumed_loss=float(resumed.replay()[0]['total']);torch.cuda.synchronize()
    resume_delta=params_diff(b,c)
    assert abs(expected_loss-resumed_loss)<1e-6,(expected_loss,resumed_loss)
    assert resume_delta<1e-7,resume_delta
    # Fast resident monitor has the same five-case Loss and aggregation as original.
    register_resident(pack)
    observed=monitor_losses(c,rows,'cuda','main',0.,32)
    gg=pack.batch(list(range(5)))
    c.eval()
    with torch.no_grad(): direct=losses(c(inputs_only(gg)),gg,None,'main',100.)
    assert abs(observed['total']-float(direct['total']))<1e-6
    result={'status':'passed','seconds':time.perf_counter()-started,'unique_cases':5,'indices':evidence['indices'],
        'original_vs_optimized_loss_max_abs':loss_delta,'global120_padding_loss_max_abs':padding_delta,
        'training_dropout_loss_abs_delta':train_loss_delta,'one_step_parameter_max_abs_delta':train_update_delta,
        'resume_loss_abs_delta':abs(expected_loss-resumed_loss),'resume_parameter_max_abs_delta':resume_delta,
        'monitor_loss_abs_delta':abs(observed['total']-float(direct['total'])),
        'temporary_optimizer_steps':5,'usable_training_started':False,'checkpoint_written':False,'test_read':False,
        'checkpoint_sha256':sha(checkpoint),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled()}
    save(HERE/'runs/20261025_profile5/verification.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
