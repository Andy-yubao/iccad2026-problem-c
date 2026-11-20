"""Short no-update batch throughput probe on the original five train members."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4];sys.path.insert(0,str(ROOT))
from experiments.E140_frontend_gb4 import train
from experiments.E140_frontend_gb4.model import GeometryModel
from experiments.E140_frontend_gb4.resident import ResidentRows
from experiments.E140_frontend_gb4.data import load_rows
from experiments.E140_frontend_gb4.cuda_step import CapturedStep


def main():
    plan=json.loads((HERE.parents[2]/'plan.json').read_text(encoding='utf-8'))
    rows=load_rows('train',plan['preflight_train_indices'])
    result=dict(gpu=torch.cuda.get_device_name(0),total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
        unique_train_members=[r['member'] for r in rows],padded_nodes=120,padded_ports=145,
        formal_training=False,optimizer_updates=0,dev_payloads=0,test_payloads=0,postprocessing_calls=0,
        scope='Scratch disposable models, repeated five train cases, same E140 FP32 loss/augmentation path. Forward/backward/clip only; excludes Adam/projection, full train distribution, monitoring and checkpoints.',batches=[])
    for batch_size in (32,64,128):
        train.seed_all(0);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        model=GeometryModel(**train.MODEL).cuda().train()
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        optimizer=train.optimizer_for(model)
        pack=ResidentRows(rows,'cuda',max_nodes=120,max_pins=145)
        ids=[i%5 for i in range(batch_size)];sample=pack.batch(ids)
        started=time.perf_counter();cap=CapturedStep(model,optimizer,sample,'e119_front_gb4');capture_seconds=time.perf_counter()-started
        for _ in range(3):cap.graph.replay()
        torch.cuda.synchronize();gpu_times=[]
        for _ in range(3):
            begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(8):cap.graph.replay()
            end.record();end.synchronize();gpu_times.append(begin.elapsed_time(end)/8)
        for _ in range(2):pack.batch(ids,out=cap.inputs);cap.graph.replay();torch.cuda.synchronize()
        pipeline=[]
        for _ in range(12):
            begin=time.perf_counter();pack.batch(ids,out=cap.inputs);cap.graph.replay()
            if not bool(cap.finite & (cap.values['total']<1000) & (cap.max_case<1000) & (cap.norm<10000)):
                raise RuntimeError('Numerical guard rejected no-update probe')
            pipeline.append((time.perf_counter()-begin)*1000)
        assert all(torch.equal(v.detach().cpu(),initial[k]) for k,v in model.state_dict().items())
        assert not optimizer.state and cap.projection.steps==0
        row=dict(batch_size=batch_size,capture_seconds=capture_seconds,gpu_ms_per_batch=float(np.median(gpu_times)),
            host_copy_forward_backward_guard_ms_median=float(np.median(pipeline)),
            host_copy_forward_backward_guard_ms_p95=float(np.quantile(pipeline,.95)),
            samples_per_second=batch_size/(float(np.median(pipeline))/1000),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            weights_unchanged=True,optimizer_updates=0)
        result['batches'].append(row);print(json.dumps(row),flush=True)
        (HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        peak=row['peak_reserved_bytes']
        del cap,optimizer,model,initial,pack,sample,begin,end
        gc.collect();torch.cuda.empty_cache()
        if batch_size==64 and peak*2.15>result['total_memory_bytes']:
            result['batch128_skipped']='Conservative memory estimate exceeds physical GPU capacity'
            break
    reference=result['batches'][0]['samples_per_second']
    for row in result['batches']:row['throughput_ratio_vs_batch32']=row['samples_per_second']/reference
    result['status']='passed';(HERE/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
