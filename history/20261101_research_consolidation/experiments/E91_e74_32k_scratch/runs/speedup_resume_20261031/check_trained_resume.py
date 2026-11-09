"""Old/new trained-Adam trajectory and fast interruption recovery, five cases."""
import copy,gc,importlib.util,time
from pathlib import Path
import numpy as np
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save
from experiments.E91_e74_32k_scratch.model import GeometryModel
from experiments.E91_e74_32k_scratch.cuda_step import CapturedStep
from experiments.E91_e74_32k_scratch.data import load_rows
from experiments.E91_e74_32k_scratch.resident import ResidentRows
from experiments.E91_e74_32k_scratch.train import restore_rng,rng_state

def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return copy.deepcopy(v)

def same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b

def main():
    start=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    out=Path(__file__).resolve().parent
    c=torch.load(out/'before/latest.pt',map_location='cpu',weights_only=False)
    spec=importlib.util.spec_from_file_location('experiments.E91_e74_32k_scratch._speedup_reference_cuda',out/'reference_cuda_step.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    rows=load_rows('train',[1683,10027,5474,18140,25001])
    batch=[rows[i%5] for i in range(32)]
    pack=ResidentRows(batch,'cuda',max_nodes=120,max_pins=146)
    graph=pack.batch(list(range(32)))

    def branch(state,fast,count):
        m=GeometryModel(**c['model_config']).cuda().train();m.load_state_dict(state['model'])
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-4)
        opt.load_state_dict(copy.deepcopy(state['optimizer']));restore_rng(copy.deepcopy(state['rng']))
        step=(CapturedStep if fast else reference.CapturedStep)(m,opt,graph,'simple')
        tracking=state['graph_step_control']
        step.projection.steps=tracking['steps']
        step.projection.clipped.copy_(torch.tensor(tracking['clipped_steps_per_group'],device='cuda'))
        step.projection.last_scales.copy_(torch.tensor(tracking['last_scales'],device='cuda'))
        times=[];values=[]
        for _ in range(count):
            torch.cuda.synchronize();begin=time.perf_counter()
            loss,_=step.replay();torch.cuda.synchronize()
            times.append(time.perf_counter()-begin);values.append(float(loss['total']))
        result=dict(model=cpu(m.state_dict()),optimizer=cpu(opt.state_dict()),rng=cpu(rng_state()),
            graph_step_control=step.projection.summary())
        del m,opt,step;gc.collect();torch.cuda.empty_cache()
        return result,times,values

    old,old_times,old_losses=branch(c,False,6)
    new,new_times,new_losses=branch(c,True,6)
    same(old,new);assert old_losses==new_losses
    middle,_,_=branch(c,True,2)
    resumed,_,_=branch(middle,True,4)
    same(new,resumed)
    save(out/'trained_resume_checks.json',dict(status='passed',unique_real_train_cases=5,
        source_step=c['progress']['step'],current_LR=c['optimizer']['param_groups'][0]['lr'],
        six_steps_old_new_model_Adam_RNG_scales_counters_bit_exact=True,
        two_plus_four_interruption_recovery_bit_exact=True,
        old_median_step_ms=float(np.median(old_times)*1000),fast_median_step_ms=float(np.median(new_times)*1000),
        micro_speedup=float(np.median(old_times)/np.median(new_times)),
        formal_training_or_checkpoint_modified=False,test_read=False,seconds=time.perf_counter()-start))
    print('Trained checkpoint: old/fast six steps and fast2+4 recovery are bit-exact',flush=True)

if __name__=='__main__':main()
