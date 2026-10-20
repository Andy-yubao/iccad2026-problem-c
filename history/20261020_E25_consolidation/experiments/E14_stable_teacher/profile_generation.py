"""Five-case timing diagnosis only; never writes to a teacher bank."""
import cProfile
import io
import json
import os
import pstats
import time
import numpy as np
import torch
from .data import HERE, reader, derived_case, save
from .pipeline import Baseline
from .frozen.inference import prepare_inference
from .frozen.batches import decode
from .frozen.packing import prepare_problem
from .targets import encode,validate
from scoring.evaluation import evaluate_case


def batch_public(inputs,device):
    singles=[prepare_inference(v,'cpu') for v in inputs]
    n=max(int(v[0]) for v in inputs);out={}
    for key,value in singles[0].items():
        if key=='scale':out[key]=torch.cat([s[key] for s in singles]);continue
        pair=key in ('adjacency','mib','group')
        shape=(len(inputs),n,n) if pair else (len(inputs),n,*value.shape[2:])
        out[key]=torch.zeros(shape,dtype=value.dtype)
        for i,s in enumerate(singles):
            ni=int(inputs[i][0])
            if pair:out[key][i,:ni,:ni]=s[key][0]
            else:out[key][i,:ni]=s[key][0]
    return {k:v.to(device) for k,v in out.items()}


def main():
    torch.set_num_threads(2);torch.manual_seed(0);np.random.seed(0)
    bank=HERE/'runs/teacher_q104_stratified_v1/accepted.json'
    records=json.loads(bank.read_text())['records'];chosen=[]
    for b in (0,1,2,4,5):chosen.append(next(r for r in records if r['stratum']==b))
    start=time.perf_counter();cpu=Baseline('cpu');model_load=time.perf_counter()-start
    rows=[];cases=[];prof=cProfile.Profile()
    for r in chosen:
        t=time.perf_counter();case,_=derived_case(reader([r['member']],'train').case(0));load=time.perf_counter()-t;cases.append(case)
        prof.enable();x,stats=cpu.solve(case.solver_inputs);prof.disable()
        t=time.perf_counter();result=evaluate_case(case,x,stats['end_to_end_seconds']);score=time.perf_counter()-t
        t=time.perf_counter();p=prepare_problem(*case.solver_inputs);target=encode(x,p);validate(target,p)
        buf=io.BytesIO();np.savez_compressed(buf,**target);label=time.perf_counter()-t
        rows.append({'member':r['member'],'n':case.block_count,'input_load_seconds':load,
                     'inference_seconds':stats['inference_seconds'],'solve_seconds':stats['end_to_end_seconds'],
                     'score_seconds':score,'encode_compress_seconds':label,'Q':result['metrics']['cost'],
                     'hard_legal':result['metrics']['is_feasible'],'stability_seconds':stats['stability']['seconds']})
        print(json.dumps(rows[-1]),flush=True)
    inputs=[c.solver_inputs for c in cases]
    # End-to-end draft timings include CPU features, transfer, decode and return.
    gpu=Baseline('cuda') if torch.cuda.is_available() else None
    cpu_drafts=[cpu.draft(v) for v in inputs]
    timings={};errors=[]
    for name,solver in [('cpu',cpu),('cuda',gpu)]:
        if solver is None:continue
        for v in inputs:solver.draft(v)
        if name=='cuda':torch.cuda.synchronize()
        t=time.perf_counter()
        for _ in range(5):drafts=[solver.draft(v) for v in inputs]
        if name=='cuda':torch.cuda.synchronize()
        timings[name+'_single_draft_per_case_seconds']=(time.perf_counter()-t)/25
        if name=='cuda':errors=[float(np.max(np.abs(a-b))) for a,b in zip(cpu_drafts,drafts)]
        public=batch_public(inputs,solver.device)
        with torch.inference_mode():
            for _ in range(3):solver.model(public)
            if name=='cuda':torch.cuda.synchronize()
            t=time.perf_counter()
            for _ in range(10):solver.model(public)
            if name=='cuda':torch.cuda.synchronize()
            timings[name+'_batch5_forward_per_case_seconds']=(time.perf_counter()-t)/50
            t=time.perf_counter()
            for _ in range(5):
                public=batch_public(inputs,solver.device)
                x=(decode(solver.model(public),public)*public['scale'][:,None,None]).cpu().numpy()
            if name=='cuda':torch.cuda.synchronize()
            timings[name+'_batch5_draft_per_case_seconds']=(time.perf_counter()-t)/25
            public=batch_public(inputs,solver.device)
            state=solver.model(public)
            together=(decode(state,public)*public['scale'][:,None,None]).cpu().numpy()
        timings[name+'_batch_vs_single_max_abs']=max(float(np.abs(together[i,:int(v[0])]-solver.draft(v)).max()) for i,v in enumerate(inputs))
    output=HERE/'runs/20261019_generation_speed_profile5'
    aggregate={k:sum(r[k] for r in rows)/len(rows) for k in ('input_load_seconds','inference_seconds','solve_seconds','score_seconds','encode_compress_seconds','stability_seconds')}
    report={'cases':rows,'mean':aggregate,'model_load_seconds':model_load,'logical_cpu':os.cpu_count(),
            'gpu':torch.cuda.get_device_name() if gpu else None,'draft_benchmarks':timings,
            'cpu_cuda_max_abs_by_case':errors,'scope':'five train cases; timing diagnosis, not throughput or quality validation; cProfile adds overhead',
            'teacher_bank_modified':False,'local_test_read':False}
    save(output/'summary.json',report)
    with (output/'cpu_profile.txt').open('w') as f:pstats.Stats(prof,stream=f).strip_dirs().sort_stats('cumtime').print_stats(40)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
