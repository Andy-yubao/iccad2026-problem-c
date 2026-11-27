"""One shared deadline, two independent two-thread workers, checked incumbent rollback."""
import multiprocessing as mp
import time,hashlib
import numpy as np
from .workers import worker
from .archive import raw_measures
from .selection import choose
from .backend.frozen.packing import prepare_problem
from .backend.frozen.feasible import Objective,row_control
from .backend.frozen.solver import hard_valid
from datasets.reference_layouts import public_hard_reference
from scoring.checks import independent_hard_checks
from scoring.evaluation import evaluate_case

def certificate(row,base,ns):
    k=max(1.,row['area']/base['area'],row['wire']/max(base['wire'],1e-12))
    return float(np.log(k)+2*(sum(row['soft_violations'])-sum(base['soft_violations']))/ns)

class Service:
    def __init__(self,checkpoint=None):
        self.workers={};self.model=None;self.setup_events=[]
        if checkpoint is not None:
            import torch
            from .model import GeometryModel
            torch.set_num_threads(2)
            if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            self.model=GeometryModel(**state['model_config']).eval();self.model.load_state_dict(state['model'])
    def ready(self):
        for kind in('baseline','explore'):
            if kind in self.workers and self.workers[kind][0].is_alive():continue
            begin=time.perf_counter();parent,child=mp.get_context('spawn').Pipe()
            process=mp.get_context('spawn').Process(target=worker,args=(child,kind),daemon=True);process.start();child.close()
            if not parent.poll(60):process.terminate();process.join();raise RuntimeError('Worker setup failed: '+kind)
            message=parent.recv()
            if message[0]!='ready':raise RuntimeError(str(message))
            self.workers[kind]=(process,parent);self.setup_events.append({'worker':kind,'seconds':time.perf_counter()-begin})
    def kill(self,kind):
        process,connection=self.workers.pop(kind)
        if process.is_alive():process.terminate()
        process.join(.04)
        if process.is_alive():process.kill();process.join(.03)
        connection.close()
    def close(self):
        for kind in list(self.workers):self.kill(kind)
    def solve(self,inputs,*,draft=None,case=None,seconds=5.):
        self.ready();begin=time.perf_counter();worker_deadline=begin+seconds-.50;parent_deadline=begin+seconds-.33
        values=(int(inputs[0]),*[np.asarray(v) for v in inputs[1:]])
        p=prepare_problem(*values);fallback=row_control(p);reference=public_hard_reference(values[1],values[5],values[6]);obj=Objective(p)
        from .inference import predict
        if draft is None:draft,inference=predict(self.model,values)
        else:draft=np.asarray(draft,float);inference=0.
        rows=[];seen={};errors={};completed={};aborted=[];last_hint=begin;hint_value=np.inf
        def accept(row):
            x=np.asarray(row['positions'],float)
            if not hard_valid(x,p):raise AssertionError('Worker candidate is hard illegal')
            check=independent_hard_checks(row['positions'],values[1],values[5],reference)
            if not check['is_feasible']:raise AssertionError('Independent hard-check failure')
            key=hashlib.sha256(x.tobytes()).hexdigest()
            if key in seen:
                existing=rows[seen[key]]
                if row['worker']=='baseline':existing['baseline_candidate']=True
                return existing
            row['worker_id']=row['id'];row['id']=len(rows);row['baseline_candidate']=row['worker']=='baseline'
            row['independent_checks']=check;row['checked_seconds']=time.perf_counter()-begin
            rows.append(row);seen[key]=row['id'];return row
        for kind,(process,connection) in self.workers.items():connection.send(('solve',values,draft,begin,worker_deadline))
        while time.perf_counter()<parent_deadline and len(completed)+len(errors)<2:
            received=False
            for kind,(process,connection) in list(self.workers.items()):
                if kind in completed or kind in errors:continue
                if not connection.poll():continue
                received=True
                try:message,payload=connection.recv()
                except EOFError:errors[kind]='worker exited';continue
                if message=='candidate':
                    row=accept(payload)
                    if kind=='baseline' and 'explore' not in completed and 'explore' not in errors and time.perf_counter()-last_hint>.35:
                        rank=row['public_quality_cost']+.025*row['distance']['D']
                        if rank<hint_value-.015:
                            self.workers['explore'][1].send(('hint',row['positions']));hint_value=rank;last_hint=time.perf_counter()
                elif message=='done':completed[kind]=payload
                elif message=='error':errors[kind]=payload
            if not received:time.sleep(.0005)
        for kind in list(self.workers):
            if kind not in completed:aborted.append(kind);self.kill(kind)
        # Drain any hints sent after the explore worker's last phase. It ignores
        # stale hints while awaiting its next solve command.
        base_rows=[r for r in rows if r['baseline_candidate']]
        if rows:
            base=min(base_rows,key=lambda r:r['native_proxy']) if base_rows else None
            safe=[r for r in rows if certificate(r,base,obj.nsoft)<=1e-10] if base is not None else rows
            chosen,selection=choose(rows,base,obj.area_scale,max(raw_measures(draft,p)[1],1e-12),obj.nsoft)
        else:
            assert hard_valid(fallback,p)
            chosen={'id':0,'source':'parent_shelf','worker':'parent','positions':fallback.tolist(),'shelf_fallback':True,
                'distance':{'D':None},'public_quality_cost':None,'found_seconds':time.perf_counter()-begin,
                'baseline_candidate':False};rows=[chosen];base=None;safe=rows;selection={'regime':'emergency','mathematical_Q_guarantee':False}
        if case is not None:
            final=evaluate_case(case,chosen['positions'],time.perf_counter()-begin)
            assert final['metrics']['is_feasible'];chosen['metrics']=final['metrics']
        elapsed=time.perf_counter()-begin
        return np.asarray(chosen['positions']),{'candidates':rows,'selected_id':chosen['id'],'baseline_id':None if base is None else base['id'],
            'safe_candidates':len(safe),'seconds':elapsed,'budget_overrun':elapsed>seconds,'worker_aborts':aborted,
            'worker_errors':errors,'worker_stats':completed,'raw_positions':draft.tolist(),'inference_seconds':inference,
            'configured_threads':{'baseline':2,'explore':2,'total_numerical':4},'setup_events':list(self.setup_events),
            'shelf_selected':chosen['shelf_fallback'],'official_Q_returned_to_search':False,
            'public_certificate_available':base is not None,'selection':selection,
            'selected_certificate_log_bound':certificate(chosen,base,obj.nsoft) if base is not None else None,
            'final_scoring_inside_budget':True,'candidate_Q_offline':True}
