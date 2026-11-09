"""Five-case transfer probe and isolated cloned-model projection microbenchmark."""
import copy,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.E91_e74_32k_scratch.io import HERE,save
from experiments.E91_e74_32k_scratch.model import GeometryModel
from experiments.E91_e74_32k_scratch.step_control import GraphStepControl
from experiments.E91_e74_32k_scratch.data import load_rows
from experiments.E91_e74_32k_scratch.resident import ResidentRows
from .projection_prototype import CapturedProjection

def sync():torch.cuda.synchronize()

def main():
    started=time.perf_counter();torch.set_num_threads(4)
    out=Path(__file__).resolve().parent
    c=torch.load(HERE/'runs/20261031_scratch32k_seed0/checkpoints/latest.pt',map_location='cpu',weights_only=False)
    old=GeometryModel(**c['model_config']).cuda();old.load_state_dict(c['model'])
    fast=copy.deepcopy(old)
    a=GraphStepControl(old);b=CapturedProjection(fast)
    old_params=list(old.parameters());fast_params=list(fast.parameters())
    # Both clipping and unchanged proposals; no dataset or neural forward involved.
    with torch.no_grad():
        for amount in (0.,1e-6,.1,-.05,1e-5):
            a.before();b.before()
            torch._foreach_add_(old_params,amount);torch._foreach_add_(fast_params,amount)
            a.after();b.after();sync()
            for x,y in zip(old_params,fast_params):assert torch.equal(x,y)
            assert torch.equal(a.clipped,b.clipped) and torch.equal(a.last_scales,b.last_scales)
            assert a.steps==b.steps
        timings={'old':[],'captured':[]}
        # Interleave AB/BA; GPU shared with active training, so these are micro timings.
        for order in (('old','captured'),('captured','old')):
            for name in order:
                control,parameters=(a,old_params) if name=='old' else (b,fast_params)
                for _ in range(10):
                    sync();begin=time.perf_counter()
                    control.before();torch._foreach_add_(parameters,1e-6);control.after();sync()
                    timings[name].append(time.perf_counter()-begin)
        for x,y in zip(old_params,fast_params):assert torch.equal(x,y)
        assert torch.equal(a.clipped,b.clipped) and a.steps==b.steps
    del old,fast,a,b,old_params,fast_params
    torch.cuda.empty_cache()
    rows=load_rows('train',[1683,10027,5474,18140,25001])
    ids=[i%5 for i in range(32)]
    #146 pins inferred from logged14.64MiB fixed32 buffer; verify the rounded size.
    pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=146)
    assert round(pack.row_bytes*32/2**20,2)==14.64
    batch=pack.batch(ids);sync()
    transfer=[]
    for _ in range(20):
        sync();begin=time.perf_counter();pack.batch(ids,out=batch);sync()
        transfer.append(time.perf_counter()-begin)
    cpu=ResidentRows(rows,'cpu',max_nodes=120,max_pins=146)
    target=cpu.batch(ids);assembly=[]
    for _ in range(20):
        begin=time.perf_counter();cpu.batch(ids,out=target);assembly.append(time.perf_counter()-begin)
    med={name:float(np.median(v)*1000) for name,v in timings.items()}
    result=dict(status='passed',checkpoint_step=c['progress']['step'],
        proposal_states_bit_exact=True,clipping_counters_and_scales_bit_exact=True,capture_does_not_update=True,
        projection_wall_milliseconds=med,projection_micro_speedup=med['old']/med['captured'],
        host_assembly_plus_copy_median_ms=float(np.median(transfer)*1000),
        CPU_assembly_and_copy_median_ms=float(np.median(assembly)*1000),
        fixed_batch_MiB=pack.row_bytes*32/2**20,unique_real_train_cases=5,neural_training_updates=0,
        current_training_source_or_checkpoint_changed=False,
        limitations='GPU shared with active training; projection-only synthetic proposals and hot five-case batches. Not a full training speedup or precise whole32k transfer profile.',
        seconds=time.perf_counter()-started,test_read=False)
    save(out/'results.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
