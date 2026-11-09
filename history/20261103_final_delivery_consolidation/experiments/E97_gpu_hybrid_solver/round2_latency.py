"""Original-five single-thread anatomy and a fixed decode-in-graph control."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json,time,subprocess
import numpy as np
import torch
from threadpoolctl import threadpool_limits,threadpool_info
from .latency import HERE,ROOT,IDS,registry,public_member_inputs,Captured,sha
from .latency_fixed import padded,returned
from .student import DeploymentModel
from .nn85.base_model import decode
from .train_io.io import save

OUTPUT=HERE/'runs/20261102_round2_decode_latency'
def restored(rects,inputs):
    p=rects[0,:int(inputs[0])].cpu().numpy().astype(float)
    c,s=np.asarray(inputs[5]),np.asarray(inputs[6]);locked=c[:,:2].any(1)
    p[locked,2:]=s[locked,2:];p[c[:,1]!=0]=s[c[:,1]!=0]
    return p

@torch.inference_mode()
def main():
    OUTPUT.mkdir(exist_ok=False);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    path=HERE/'runs/20261102_candidate3_clearance_seed0/checkpoints/best.pt'
    plan=dict(source_checkpoint=path.relative_to(ROOT).as_posix(),source_sha256=sha(path),ids=IDS,repetitions=5,warmups=3,paths=['old capture=model only, decode/scaling outside graph','new capture=model+identical decode/scaling, only D2H/public exact restoration outside'],purpose='new implementation anatomy/control, not repeated identical total-only measurement; no solver, train, dev or Q',cpu_threads=1,no_power_or_clock_changes=True,source_file_sha256=sha(HERE/'round2_latency.py'),test_read=False)
    save(OUTPUT/'plan.json',plan)
    telemetry=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,power.draw,clocks.sm,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
    with threadpool_limits(1):
        saved=torch.load(path,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True);del saved
        entries=registry()['sets']['train']['records'];cases=[public_member_inputs(entries[i],'train') for i in IDS]
        g={k:v.cuda() for k,v in padded(cases[0],False).items()}
        start=time.perf_counter();old=Captured(model,g);old_cold=time.perf_counter()-start
        start=time.perf_counter();new=Captured(lambda x:decode(model(x),x)*x['scale'][:,None,None],g);new_cold=time.perf_counter()-start
        records=[]
        for index,inputs in zip(IDS,cases):
            for cap in (old,new):
                host=padded(inputs,False)
                for k,v in host.items():cap.g[k].copy_(v)
                for _ in range(3):cap.replay()
            a=returned(old.replay(),old.g,inputs,decode);b=restored(new.replay(),inputs)
            assert np.array_equal(a,b),(index,float(np.max(np.abs(a-b))))
            for repetition in range(5):
                for label,cap in ([('old',old),('decode_captured',new)] if repetition%2==0 else [('decode_captured',new),('old',old)]):
                    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True);torch.cuda.synchronize();start=time.perf_counter()
                    host=padded(inputs,False);feature_end=time.perf_counter()
                    for k,v in host.items():cap.g[k].copy_(v)
                    copied=time.perf_counter();start_event.record();output=cap.replay();end_event.record()
                    result=returned(output,cap.g,inputs,decode) if label=='old' else restored(output,inputs)
                    done=time.perf_counter();assert np.array_equal(result,a)
                    records.append(dict(train_id=index,n=int(inputs[0]),path=label,repetition=repetition,feature_ms=1000*(feature_end-start),copy_wall_ms=1000*(copied-feature_end),graph_and_return_wall_ms=1000*(done-copied),graph_cuda_ms=start_event.elapsed_time(end_event),complete_public_to_draft_ms=1000*(done-start)))
            print('latency anatomy original train',index,'done',flush=True)
        summary={label:{key:dict(mean=float(np.mean([r[key] for r in records if r['path']==label])),p95=float(np.percentile([r[key] for r in records if r['path']==label],95))) for key in ['feature_ms','copy_wall_ms','graph_and_return_wall_ms','graph_cuda_ms','complete_public_to_draft_ms']} for label in ['old','decode_captured']}
        result=dict(status='passed',summary=summary,records=records,bit_exact_physical_drafts=True,capture_seconds=dict(old=old_cold,new=new_cold),threads=threadpool_info(),gpu_before=telemetry,interpretation='old graph CUDA excludes decode while new includes it; graph+return wall includes stream waiting, not standalone decode time. Five cases only, no final-service claim.',no_solver_or_training=True,test_read=False)
        save(OUTPUT/'results.json',result);print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
