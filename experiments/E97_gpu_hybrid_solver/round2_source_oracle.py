"""Read-only source-selection diagnostic; all geometry was already solved."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math,hashlib
from pathlib import Path
import numpy as np
from .train_io.data import load_rows
from .repair85_one_thread.backend.frozen.packing import prepare_problem
from .repair85_one_thread.backend.frozen.feasible import Objective
from .train_io.io import save
from threadpoolctl import threadpool_limits
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    plan=read(HERE/'round2_source_oracle_plan.json');output=HERE/'runs/20261102_round2_saved_source_oracle';output.mkdir(exist_ok=False)
    folders=[HERE/'runs/20261102_fullrepair_singlethread_dev500/E85/terminal',HERE/'runs/20261102_round2_current_control/C3/terminal',HERE/'runs/20261102_round2_candidate2_dynamicmst_seed0/evaluation/terminal']
    arms=[[read(p) for p in sorted(f.glob('*.json'))] for f in folders];assert all(len(a)==500 for a in arms)
    rows=load_rows('dev_monitor500');records=[]
    for i,row in enumerate(rows):
        problem=prepare_problem(*row['case'].solver_inputs);objective=Objective(problem);values=[]
        for arm in arms:
            r=arm[i];assert r['member']==list(row['member']);p=np.array(r['positions']);centers=p[:,:2]+p[:,2:]/2
            area=float(np.prod((p[:,:2]+p[:,2:]).max(0)-p[:,:2].min(0)));wire=0.
            if len(objective.bi):wire+=float((np.abs(centers[objective.bi[:,0]]-centers[objective.bi[:,1]]).sum(-1)*problem.b2b[:,2]).sum())
            if len(objective.pi):wire+=float((np.abs(problem.pins[objective.pi[:,0]]-centers[objective.pi[:,1]]).sum(-1)*problem.p2b[:,2]).sum())
            values.append((area,wire,sum(objective.violations(p))))
        a,w,s=values[1];rho=[max(1,v[0]/a,v[1]/max(w,1e-12))*math.exp(2*(v[2]-s)/objective.nsoft) for v in values]
        choice=min(range(3),key=lambda k:(rho[k],k)) if min(rho)<1-1e-10 else 1
        assert arms[choice][i]['metrics']['cost']<=arms[1][i]['metrics']['cost']+1e-9
        records.append(dict(index=i,member=list(row['member']),n=problem.n,choice=choice,rho=rho,public_components=values,selected_saved_Q=arms[choice][i]['metrics']['cost'],control_saved_Q=arms[1][i]['metrics']['cost']))
    mass=sum(math.exp((r['n']-120)/12) for r in records)
    result=dict(status='complete',plan=plan,counts={str(k):sum(r['choice']==k for r in records) for k in range(3)},selected_saved_weighted_Q=sum(math.exp((r['n']-120)/12)*r['selected_saved_Q'] for r in records)/mass,control_saved_weighted_Q=sum(math.exp((r['n']-120)/12)*r['control_saved_Q'] for r in records)/mass,records=records,not_deployable=True,no_new_inference_solver_scorer_or_optimizer=True,test_read=False)
    save(output/'results.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('records','plan')},indent=2))
if __name__=='__main__':
    with threadpool_limits(1):main()
