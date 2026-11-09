"""Same five members: one legacy/optimized numerical path, real scratch updates."""
import gc,time,json,subprocess
import numpy as np
import torch
from .common import RUN,save,seed,restore_rng,cpu
from .train import baseline,configure,Engine
from .data import Data,Packer,public
IDS=[1683,10027,15001,32000,63999]

def main():
    saved=torch.load(RUN/'L/checkpoints/latest.pt',map_location='cpu',weights_only=False);data=Data('train');packer=Packer(data);ids=(IDS*7)[:32]
    report={};before_gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,power.draw,clocks.sm,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    for stage in ('L',):
        expected=None;expected_opt=None;timings={}
        for fast in (False,True):
            seed();model,cfg=baseline();part=configure(stage,model)
            if stage=='L':part.load_state_dict(saved['model'])
            model=model.cuda();params=[p for p in part.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.00025,weight_decay=.0003)
            if stage=='L':opt.load_state_dict(cpu(saved['optimizer']))
            host,_=packer.pack(ids);g={k:v.cuda() for k,v in host.items()}
            if stage=='F':
                with torch.no_grad():
                    model.placer.eval();base=model.placer(public(g));target=part(base,public(g)).detach()
                g={k:v for k,v in g.items() if k not in ('target','reference')};g.update(base=base,teacher=target,usable=torch.ones(32,device='cuda'))
            frozen={k:v.detach().cpu().clone() for k,v in model.named_parameters() if not v.requires_grad};initial=cpu(part.state_dict());opt_initial=cpu(opt.state_dict());restore_rng(saved['rng']);engine=Engine(stage,part,opt,g,fast_update=fast)
            assert all(torch.equal(v,part.state_dict()[k].detach().cpu()) for k,v in initial.items()),'Capture changed model'
            if stage=='L':
                after=cpu(opt.state_dict())
                for k,state in opt_initial['state'].items():
                    for name,v in state.items():assert torch.equal(v,after['state'][k][name]),'Capture advanced Adam'
            else:
                assert all(float(v['step'].cpu())==0 for v in opt.state.values()),'Capture advanced new Adam'
            if engine.projection:
                meta=saved['projection'];engine.projection.steps=meta['steps'];engine.projection.clipped.copy_(torch.tensor(meta['clipped_steps_per_group'],device='cuda'))
            times=[];losses=[]
            first_result=None;first_moments=None
            for step_index in range(5):
                torch.cuda.synchronize();begin=time.perf_counter();losses.append(engine.step(g));torch.cuda.synchronize();times.append(1000*(time.perf_counter()-begin))
                if step_index==0:first_result=cpu(part.state_dict());first_moments=cpu(opt.state_dict())
            result=cpu(part.state_dict());moments=cpu(opt.state_dict());assert all(torch.equal(v,dict(model.named_parameters())[k].detach().cpu()) for k,v in frozen.items())
            label='captured_update' if fast else 'legacy_update';timings[label]=dict(mean_update_ms=float(np.mean(times)),times_ms=times,losses=losses,peak_allocated_bytes=torch.cuda.max_memory_allocated(),capture_optimizer_updates=0,frozen_parameters_unchanged=True)
            if not fast:expected=result;expected_opt=moments;expected_first=first_result;expected_first_opt=first_moments
            else:
                diff=max(float((result[k]-v).abs().max()) for k,v in expected.items());moment_diff=max([float((state[n]-moments['state'][k][n]).abs().max()) for k,state in expected_opt['state'].items() for n in ('exp_avg','exp_avg_sq')] or [0.])
                loss_diff=max(abs(a-b) for a,b in zip(losses,timings['legacy_update']['losses']))
                first_diff=max(float((first_result[k]-v).abs().max()) for k,v in expected_first.items())
                first_moment_diff=max([float((state[n]-first_moments['state'][k][n]).abs().max()) for k,state in expected_first_opt['state'].items() for n in ('exp_avg','exp_avg_sq')] or [0.])
                save(RUN/'engineering/check_v2_partial.json',dict(stage=stage,paths=timings,max_parameter_abs_difference=diff,max_moment_abs_difference=moment_diff,max_loss_abs_difference=loss_diff,first_step_parameter_difference=first_diff,first_step_moment_difference=first_moment_diff))
                assert diff<3e-5 and moment_diff<3e-5 and loss_diff<3e-5,(stage,diff,moment_diff,loss_diff)
                timings.update(max_parameter_abs_difference=diff,max_moment_abs_difference=moment_diff,max_loss_abs_difference=loss_diff,observed_update_ratio=timings['legacy_update']['mean_update_ms']/timings['captured_update']['mean_update_ms'])
            del engine,opt,model,part,g,result,moments,initial,opt_initial,frozen;gc.collect();torch.cuda.empty_cache()
        report[stage]=timings
    result=dict(status='passed',unique_train_members=IDS,scratch_updates_per_path_per_stage=5,no_formal_optimizer_updates=True,no_resume_recovery_test=True,test_read=False,engine_change='same objective/batch/precision/order, GPU capturable Adam bias correction has registered FP32 operation-order differences',gpu_before=before_gpu,stages=report)
    save(RUN/'engineering/check_v2.json',result);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
    try:main()
    except BaseException as e:save(RUN/'engineering/check_v2_error.json',dict(error=repr(e)));raise
