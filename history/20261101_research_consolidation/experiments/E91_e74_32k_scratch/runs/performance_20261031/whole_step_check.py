"""Isolated same-five-case Dropout/Adam comparison; active training untouched."""
import copy,gc,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save
from experiments.E91_e74_32k_scratch.data import load_rows
from experiments.E91_e74_32k_scratch.resident import ResidentRows
from experiments.E91_e74_32k_scratch.model import GeometryModel
from experiments.E91_e74_32k_scratch.cuda_step import CapturedStep
from experiments.E91_e74_32k_scratch.train import restore_rng
from .projection_prototype import CapturedProjection

def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)

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
    begin=time.perf_counter();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    folder=Path(__file__).resolve().parent
    c=torch.load(HERE/'runs/20261031_scratch32k_seed0/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    rows=load_rows('train',[1683,10027,5474,18140,25001])
    batch=[rows[i%5] for i in range(32)]
    pack=ResidentRows(batch,'cuda',max_nodes=120,max_pins=146)
    graph=pack.batch(list(range(32)));torch.cuda.synchronize()
    results={};snapshots={}
    for mode in ('old','captured'):
        m=GeometryModel(**c['model_config']).cuda().train();m.load_state_dict(c['model'])
        opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=3e-4)
        opt.load_state_dict(copy.deepcopy(c['optimizer']))
        restore_rng(copy.deepcopy(c['rng']))
        step=CapturedStep(m,opt,graph,'simple')
        if mode=='captured':step.projection=CapturedProjection(m)
        durations=[];values=[]
        for _ in range(10):
            torch.cuda.synchronize();started=time.perf_counter()
            loss,norm=step.replay();torch.cuda.synchronize()
            durations.append(time.perf_counter()-started);values.append(float(loss['total']))
        snapshots[mode]=dict(model=cpu(m.state_dict()),optimizer=cpu(opt.state_dict()),
            clip_counts=cpu(step.projection.clipped),scales=cpu(step.projection.last_scales),
            steps=step.projection.steps,CUDA_RNG=cpu(torch.cuda.get_rng_state_all()))
        results[mode]=dict(step_median_ms=float(np.median(durations)*1000),
            step_mean_ms=float(np.mean(durations)*1000),losses=values)
        del step,opt,m;gc.collect();torch.cuda.empty_cache()
    same(snapshots['old'],snapshots['captured'])
    assert results['old']['losses']==results['captured']['losses']
    save(folder/'whole_step_results.json',dict(status='passed',unique_real_train_cases=5,
        checkpoint_step=c['progress']['step'],isolated_steps_per_path=10,
        full_step_model_Adam_RNG_clipping_bit_exact=True,same_FP32_Dropout_batch32_Loss=True,
        timings=results,whole_step_micro_speedup=results['old']['step_median_ms']/results['captured']['step_median_ms'],
        limitation='Shared GPU, hot same-five-case32 batches; excludes real32k staging, monitoring and checkpoint I/O. Not a formal whole-run benchmark.',
        active_training_source_or_checkpoint_mutated=False,test_read=False,seconds=time.perf_counter()-begin))
    print(json.dumps({k:v for k,v in json.loads((folder/'whole_step_results.json').read_text()).items() if k!='timings'}),flush=True)

if __name__=='__main__':main()
