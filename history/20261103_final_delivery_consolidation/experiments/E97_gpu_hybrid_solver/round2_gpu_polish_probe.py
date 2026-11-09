"""Original five train implementation test, no dev or Q."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import time,json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .latency import HERE,ROOT,IDS,Captured,sha
from .latency_fixed import padded
from .round2_latency import restored
from .student import DeploymentModel
from .nn85.base_model import decode
from .train_io.data import load_rows
from .train_io.io import save,exclusive
from .repair85_one_thread.solver import optimize,raw_measures
from .repair85_one_thread.backend.frozen.packing import prepare_problem
from .repair85_one_thread.backend.frozen.feasible import Objective
from .repair85_one_thread.canonical_native import warmup
from .round2_gpu_polish import BatchGeometry,proposals,polish
from scoring.checks import independent_hard_checks
from datasets.reference_layouts import public_hard_reference
OUTPUT=HERE/'runs/20261102_round2_gpu_polish_probe'

def cpu_reference(x,p,mask,delta,links):
    moved=mask.copy()
    for _ in range(8):
        xy=x[None,:,:2]+moved[...,None]*delta[:,None];hi=xy+x[None,:,2:];ov=np.minimum(hi[:,:,None],hi[:,None,:])-np.maximum(xy[:,:,None],xy[:,None,:])
        blocked=((ov>1e-8).all(-1)&moved[:,:,None]&~moved[:,None,:]).any(1);moved|=(blocked.astype(np.int32)@links.astype(np.int32))>0
    candidates=np.broadcast_to(x,(len(mask),*x.shape)).copy();candidates[:,:,:2]+=moved[...,None]*delta[:,None]
    return candidates

@torch.inference_mode()
def main():
    begin=time.perf_counter();OUTPUT.mkdir(exist_ok=False);torch.manual_seed(0);torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    with exclusive(HERE/'runs/round2_execution.lock'):
        path=HERE/'runs/20261102_candidate3_clearance_seed0/checkpoints/best.pt';saved=torch.load(path,map_location='cpu',weights_only=False);model=DeploymentModel(**saved['model_config']).cuda().eval();model.load_state_dict(saved['model'],strict=True)
        rows=load_rows('train',IDS);full=json.loads((HERE/'round2_control_plan.json').read_text(encoding='utf-8'))['full_repair'];warmup();g={k:v.cuda() for k,v in padded(rows[0]['case'].solver_inputs,False).items()};cap=Captured(lambda gg:decode(model(gg),gg)*gg['scale'][:,None,None],g);records=[]
        for index,row in zip(IDS,rows):
            inputs=row['case'].solver_inputs;host=padded(inputs,False)
            for k,v in host.items():cap.g[k].copy_(v)
            draft=restored(cap.replay(),inputs);initial,stats=optimize(draft,inputs,**full);p=prepare_problem(*inputs);obj=Objective(p)
            mask,delta,links=proposals(initial,p)
            if len(mask)==0:
                mask=np.zeros((3,p.n),dtype=bool);free=np.flatnonzero(~p.preplaced)[0];mask[:,free]=True;delta=np.array([[.01,0],[-.01,0],[0,.01]])
            mask=mask[:5];delta=delta[:5];gpu=BatchGeometry(p);candidate,a,w,v,legal=gpu.evaluate(initial,mask,delta,links);candidate=candidate.cpu().numpy();reference=cpu_reference(initial,p,mask,delta,links)
            error=float(np.abs(candidate-reference).max());assert error<1e-10
            cpu_measures=np.array([raw_measures(x,p) for x in reference]);np.testing.assert_allclose(a.cpu().numpy(),cpu_measures[:,0],rtol=1e-12,atol=1e-8);np.testing.assert_allclose(w.cpu().numpy(),cpu_measures[:,1],rtol=1e-12,atol=1e-8)
            final,trace=polish(initial,p);hard=independent_hard_checks(final.tolist(),np.asarray(inputs[1]),np.asarray(inputs[5]),public_hard_reference(inputs[1],inputs[5],inputs[6]));assert hard['is_feasible'] and (final[:,:2]>=0).all()
            old_area,old_wire=raw_measures(initial,p);new_area,new_wire=raw_measures(final,p);old_soft=sum(obj.violations(initial));new_soft=sum(obj.violations(final));cert=max(1,new_area/old_area,new_wire/max(old_wire,1e-15))*np.exp(2*(new_soft-old_soft)/obj.nsoft);assert cert<=1+1e-12
            records.append(dict(train_id=index,member=list(row['member']),n=p.n,GPU_CPU_batch_position_error=error,initial_soft=obj.violations(initial),final_soft=obj.violations(final),certificate=float(cert),hard_legal=True,polish=trace,initial_positions=initial.tolist(),final_positions=final.tolist()))
            print(index,'soft',old_soft,'->',new_soft,'accepted',trace['accepted'],'gpu_polish_seconds',trace['seconds'],flush=True)
        save(OUTPUT/'results.json',dict(status='passed',records=records,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__),HERE/'round2_gpu_polish.py',HERE/'round2_gpu_polish_plan.json']},seconds=time.perf_counter()-begin,no_dev_Q_optimizer=True,test_read=False))
if __name__=='__main__':
    try:
        with threadpool_limits(1):main()
    except BaseException as e:save(OUTPUT/'error.json',dict(error=repr(e),no_automatic_retry=True));raise
