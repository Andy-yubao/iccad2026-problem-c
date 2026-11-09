"""Resident process boundary for a5-second end-to-end candidate deadline.

Validated candidates are streamed out before a later numerical call can stall.
On deadline the worker is terminated and the parent returns a checked incumbent.
Official Q stays in the parent and is never sent back to search.
"""
import multiprocessing as mp
import time
import numpy as np
from .backend.frozen.packing import prepare_problem
from .backend.frozen.feasible import row_control
from .backend.frozen.solver import hard_valid


def worker(connection,checkpoint):
    import os
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='4'
    os.environ['NUMBA_NUM_THREADS']='1'
    import torch
    from .model import GeometryModel
    from .inference import predict
    from .canonical_native import warmup
    from .monotone import optimize
    from .baseline.canonical_native import warmup as baseline_warmup
    from .backend.frozen.topology_mip import optimize as warm_mip_import
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True);warmup();baseline_warmup()
    model=None
    if checkpoint is not None:
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=GeometryModel(**state['model_config']).eval();model.load_state_dict(state['model'],strict=True)
    connection.send(('ready',{'torch_threads':2,'native_threads':4}))
    try:
        while True:
            job=connection.recv()
            if job is None:return
            inputs,raw,seconds,profile=job;begin=time.perf_counter()
            if raw is None:raw,inference=predict(model,inputs)
            else:raw=np.asarray(raw,float);inference=0.
            connection.send(('draft',{'positions':raw.tolist(),'inference_seconds':inference}))
            def observe(x,row):
                connection.send(('candidate',row))
                acknowledgement=connection.recv()
                if acknowledgement!='checked':raise RuntimeError('Parent rejected candidate or canceled evaluation')
            result,stats=optimize(raw,inputs,seconds=max(.1,seconds-(time.perf_counter()-begin)-.08),observer=observe,profile=profile)
            stats.pop('candidates',None)
            connection.send(('done',stats))
    except (EOFError,BrokenPipeError):return
    except Exception as error:
        try:connection.send(('error',repr(error)))
        except (EOFError,BrokenPipeError):pass
    finally:connection.close()


class ResidentPotential:
    def __init__(self,checkpoint=None):
        self.checkpoint=str(checkpoint) if checkpoint is not None else None
        self.process=None;self.connection=None;self.last_setup_seconds=0.;self.restarts=0
    def ensure_ready(self):
        if self.process is not None and self.process.is_alive():return
        begin=time.perf_counter();parent,child=mp.get_context('spawn').Pipe()
        process=mp.get_context('spawn').Process(target=worker,args=(child,self.checkpoint),daemon=True)
        process.start();child.close()
        if not parent.poll(60):process.terminate();process.join();raise RuntimeError('Resident solver initialization failed')
        message=parent.recv()
        if message[0]!='ready':raise RuntimeError(str(message))
        self.connection=parent;self.process=process;self.last_setup_seconds=time.perf_counter()-begin;self.restarts+=1
    def terminate(self):
        if self.process is not None:
            if self.process.is_alive():self.process.terminate()
            self.process.join(.04)
            if self.process.is_alive():self.process.kill();self.process.join(.03)
            self.process=None
        if self.connection is not None:self.connection.close();self.connection=None
    def close(self):
        self.terminate()
    def solve(self,inputs,*,draft=None,case=None,seconds=5.,force_stall=False,profile='v12'):
        self.ensure_ready()  # cold startup/restart is explicit and recorded separately
        begin=time.perf_counter();deadline=begin+seconds-.30
        values=(int(inputs[0]),*[np.asarray(v) for v in inputs[1:]])
        p=prepare_problem(*values);fallback=row_control(p)
        if not hard_valid(fallback,p):raise ValueError('Invalid public specifications')
        from scoring.evaluation import evaluate_case
        from scoring.checks import independent_hard_checks
        from datasets.reference_layouts import public_hard_reference
        reference=public_hard_reference(values[1],values[5],values[6])
        def check(row):
            x=np.asarray(row['positions'],float)
            if not hard_valid(x,p):raise AssertionError('Worker streamed an invalid candidate')
            checked=independent_hard_checks(x.tolist(),values[1],values[5],reference)
            if not checked['is_feasible']:raise AssertionError('Independent candidate check failed')
            row['independent_hard_checks']=checked
            row['score_complete_seconds']=time.perf_counter()-begin
            return row
        emergency=check({'id':0,'source':'parent_shelf_fallback','positions':fallback.tolist(),'shelf_fallback':True,
            'public_quality_cost':None,'distance':{'D':None},'found_seconds':0.})
        self.connection.send((values,None if draft is None else np.asarray(draft),seconds-.08,profile))
        candidates=[];raw=draft;inference=None;stats=None;aborted=False;error=None
        while time.perf_counter()<deadline:
            if not self.connection.poll(min(.02,max(0.,deadline-time.perf_counter()))):continue
            try:kind,payload=self.connection.recv()
            except EOFError:error='Worker exited before completion';break
            if kind=='draft':raw=np.asarray(payload['positions']);inference=payload['inference_seconds']
            elif kind=='candidate':
                row=check(payload);candidates.append(row);self.connection.send('checked')
                if force_stall:
                    # Test-only: suspend consumption until the real deadline;
                    # a valid previously streamed candidate must still be returned.
                    while time.perf_counter()<deadline:time.sleep(min(.01,max(0.,deadline-time.perf_counter())))
                    break
            elif kind=='done':stats=payload;break
            elif kind=='error':error=payload;break
        if stats is None:
            aborted=True;self.terminate()
        if candidates:
            from .monotone import certificate
            from .backend.frozen.feasible import Objective
            base=next(r for r in candidates if r['source']=='protected_ordinary');ns=Objective(p).nsoft
            safe=[r for r in candidates if certificate(r,base,ns)<=1e-10]
            selected=min(safe,key=lambda r:r['public_quality_cost']+.02*r['distance']['D'])
            public=min(candidates,key=lambda r:r['public_quality_cost'])
        else:
            selected=public=emergency;candidates=[emergency]
        if case is not None:
            scored=evaluate_case(case,selected['positions'],time.perf_counter()-begin)
            if not scored['metrics']['is_feasible']:raise AssertionError('Selected independent official check failed')
            selected['metrics']=scored['metrics']
        traces=stats['trace'] if stats is not None else [{'phase':'external_deadline_or_worker_failure','error':error,'trace_incomplete':True}]
        final={'seconds':time.perf_counter()-begin,'budget_seconds':seconds,'external_abort':aborted,'worker_error':error,
            'selected_id':selected['id'],'public_quality_best_id':public['id'],'candidates':candidates,'trace':traces,
            'shelf_selected':selected['shelf_fallback'],'raw_positions':None if raw is None else np.asarray(raw).tolist(),
            'inference_seconds':inference,'resident_startups':self.restarts,'last_setup_seconds':self.last_setup_seconds,
            'trace_incomplete':stats is None,'official_Q_returned_to_search':False,'configured_native_threads':4,
            'deadline_reserve_seconds':.30,'baseline_id':base['id'] if candidates[0] is not emergency else None,
            'safe_candidates':len(safe) if candidates[0] is not emergency else 0,'profile':profile,
            'candidate_Q_scoring':'offline after solve; final selected official scoring inside case timer'}
        final['budget_overrun']=final['seconds']>seconds
        return np.asarray(selected['positions']),final
