"""One disposable profiled update; live training is untouched. No new Q calls."""
import sys
from pathlib import Path
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from experiments.E106_postprocessor_aware_loss.common import OWNER,RUN,read,save,setup
from experiments.E106_postprocessor_aware_loss.model import PostprocessorCritic
from experiments.E106_postprocessor_aware_loss.data import load_case,batch_cases
from experiments.E106_postprocessor_aware_loss.objective import fitting_loss,role_exp_mean


def main():
    setup()
    target=Path(__file__).resolve().parent
    config=read(OWNER/'plan.json')
    saved=torch.load(RUN/'critic/latest.pt',map_location='cpu',weights_only=False)
    picked=read(OWNER/'preflight.json')['case_indices']
    rows=[load_case(i) for i in picked]
    rows.append(rows[-1])
    # Only label/geometry data from the original five implementation cases.
    started=time.perf_counter()
    cpu=batch_cases(rows,'cpu')
    cpu_batch_ms=(time.perf_counter()-started)*1000
    tensor_bytes=sum(t.numel()*t.element_size() for t in cpu['graph'].values())
    tensor_bytes+=sum(v.numel()*v.element_size() for k,v in cpu.items() if k!='graph')
    model=PostprocessorCritic(**config['model']).cuda().train()
    model.load_state_dict(saved['model'])
    policy=config['proposed_training']
    optimizer=torch.optim.AdamW(model.parameters(),lr=policy['lr_tiers'][0],weight_decay=policy['weight_decay'])
    optimizer.load_state_dict(saved['optimizer'])
    normalization=read(RUN/'normalization.json')
    exp_mean=role_exp_mean('fit')
    # Prime CUDA library initialization without performing an optimizer update.
    warm=batch_cases(rows,'cuda')
    optimizer.zero_grad(set_to_none=True)
    fitting_loss(model(warm['graph'],warm['states'],group_size=5),warm,normalization,exp_mean)['total'].backward()
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False) as prof:
        with record_function('E106_CPU_assemble_and_H2D'):
            batch=batch_cases(rows,'cuda')
        with record_function('E106_forward_backward_clip'):
            values=fitting_loss(model(batch['graph'],batch['states'],group_size=5),batch,normalization,exp_mean)
            values['total'].backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),policy['clip_norm'])
        with record_function('E106_numeric_guard'):
            if not bool(torch.isfinite(norm)&torch.isfinite(values['total'])&(norm<1e4)&(values['total']<1e6)):
                raise FloatingPointError('Disposable profiling update rejected')
        with record_function('E106_AdamW'):
            optimizer.step()
        torch.cuda.synchronize()
    def entry(e):
        return dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,
                    device_total_us=getattr(e,'device_time_total',0),
                    self_device_us=getattr(e,'self_device_time_total',0))
    averages=prof.key_averages()
    sections=[entry(e) for e in averages if e.key.startswith('E106_')]
    top_cpu=[entry(e) for e in sorted(averages,key=lambda e:e.self_cpu_time_total,reverse=True)[:22]]
    top_gpu=[entry(e) for e in sorted(averages,key=lambda e:getattr(e,'self_device_time_total',0),reverse=True)[:22]]
    events=prof.events()
    kernels=[e for e in events if str(e.device_type).endswith('CUDA')]
    save(target/'profile.json',dict(status='complete',source_epoch=saved['progress']['epoch'],
        source_step=saved['progress']['step'],unique_fit_cases=5,case_indices=picked,
        cpu_batch_ms=cpu_batch_ms,batch_transfer_bytes=tensor_bytes,
        cuda_event_count=len(kernels),sections=sections,top_cpu=top_cpu,top_gpu=top_gpu,
        optimizer_fused=optimizer.param_groups[0].get('fused'),
        optimizer_foreach=optimizer.param_groups[0].get('foreach'),
        Loss=float(values['total']),gradient_norm=float(norm),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        concurrent_with_live_training=True,not_a_speedup_benchmark=True,
        disposable_optimizer_updates=1,formal_training_touched=False,new_Q_calls=0,test_read=False))
    print('Profiled one disposable update; live training unchanged.',flush=True)


if __name__=='__main__':
    main()
